import time
from dataclasses import asdict
import gc
import wandb
from collections import defaultdict
from transformers import (
    AutoTokenizer,
    set_seed,
    get_scheduler,
    AutoModelForCausalLM,
)
import torch
from torch import nn, optim
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from accelerate import Accelerator
import json
import tyro
import sys
sys.path.append('/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/benchmark/RLHF-training')
from dataset.data_utils import *
from config import Config
import os
from utils import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True


if __name__=="__main__":
    config = tyro.cli(Config)
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
    world_size = accelerator.num_processes
    
    set_seed(config.seed)
    if config.debug == False and accelerator.is_main_process:
        wandb.init(
            project=config.project_name,
            config=asdict(config),
            name=config.exp_name
        )

    policy_kwargs = {
        "torch_dtype": torch.bfloat16 if "gpt2" not in config.model_path else torch.float32,
        "attn_implementation": "flash_attention_2" if "gpt2" not in config.model_path else "eager",
        "use_cache": False
    }
    accelerator.print(policy_kwargs)

    if config.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        policy = AutoLigerKernelForCausalLM.from_pretrained(
            config.model_path,
            **policy_kwargs
        )
    else:
        policy = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            **policy_kwargs
        )
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    policy.resize_token_embeddings(len(tokenizer))
    
    train_dataset = globals()[f'get_{config.dataset_name}']('train')
    try:
        import random
        random.shuffle(train_dataset)
    except:
        train_dataset = train_dataset.shuffle(config.seed)
    test_dataset = globals()[f'get_{config.dataset_name}']('test')
    test_dataset = test_dataset.shuffle(config.seed)
    test_dataset = test_dataset.select(range(256))
    train_dataset = TorchDataset(train_dataset, tokenizer, max_length=config.max_length, max_prompt_length=config.max_prompt_length)
    accelerator.print(config)
    # test_dataset = TorchDataset(test_dataset, tokenizer, max_length=config.max_length, max_prompt_length=config.max_prompt_length)

    accelerator.print(f"Number of training data: {len(train_dataset)}")
    # accelerator.print(f"Number of testing data: {len(test_dataset)}")

    if config.use_packing:
        accelerator.print('Use packing collator')
    else:
        accelerator.print('Use padding collator')

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size // world_size, collate_fn=train_dataset.packing_collate_fn if config.use_packing else train_dataset.padding_collate_fn, pin_memory=True, num_workers=8, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=config.eval_batch_size // world_size, collate_fn=train_dataset.packing_collate_fn if config.use_packing else train_dataset.padding_collate_fn, pin_memory=True, num_workers=8, shuffle=True)

    if config.gradient_checkpointing:
        policy.gradient_checkpointing_enable(dict(use_reentrant=False))

    optimizer = getattr(torch.optim, config.optimizer)(policy.parameters(), lr=config.lr, weight_decay=0.0, eps=1e-5)

    lr_scheduler = get_scheduler(
        'cosine_with_min_lr',
        optimizer,
        num_warmup_steps=config.warm_up_steps,
        num_training_steps=(len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps)) * config.num_train_epochs,
        scheduler_specific_kwargs={"min_lr": config.lr * 0.1},
    )
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (config.warm_up_steps + 1)))
    policy, optimizer, train_dataloader = accelerator.prepare(policy, optimizer, train_dataloader)
    # eval_dataloader = accelerator.prepare(test_dataloader)

    example_counter = 0
    accumulated = 0
    batch_metrics= defaultdict(list)
    policy.train()
    average_log_prob = True if config.average_log_prob else False
    print(f"Total Epochs: {config.num_train_epochs}")

    for epoch in range(config.num_train_epochs):
        for batch in train_dataloader:
            if example_counter % config.eval_every == 0 and example_counter > 0:
                # all_eval_metrics = defaultdict(list)
                # for batch in eval_dataloader:
                #     with torch.inference_mode():
                #         logits = policy(
                #             input_ids=batch['target_combined_input_ids'],
                #             attention_mask=batch['target_combined_attention_mask'],
                #         ).logits.to(torch.float32)
                #         policy_logps = get_batch_logps(logits, batch['target_labels'], average_log_prob=True)
                #         losses = (-policy_logps)
                #     losses = accelerator.gather_for_metrics(losses)
                #     all_eval_metrics['eval/loss'].extend(losses.float().cpu().numpy().tolist())
                
                # mean_eval_metrics = {}
                # for k, v in all_eval_metrics.items():
                #     mean_eval_metrics[k] = sum(v) / len(v)
                # accelerator.print(f'eval stats after {example_counter} examples: {formatted_dict(mean_eval_metrics)}')
                # if accelerator.is_main_process and config.debug == False:
                #     wandb.log(mean_eval_metrics, step=example_counter)

                save(policy, example_counter, accelerator, tokenizer, config)

            start_time = time.time()
            example_counter += config.batch_size
            with accelerator.accumulate(policy):
                with accelerator.autocast():
                        # position_ids=batch['target_combined_position_ids']
                    logits = policy(
                        input_ids=batch['target_combined_input_ids'],
                        attention_mask=batch['target_combined_attention_mask'],
                    ).logits.to(torch.float32)
                    policy_logps = get_batch_logps(logits, batch['target_labels'], average_log_prob=average_log_prob)
                    loss = (-policy_logps).mean()   
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    accumulated += 1
            
            step_time = time.time() - start_time
            examples_per_second = config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)

            with torch.no_grad():
                train_losses = accelerator.gather_for_metrics(-policy_logps).float().cpu().numpy().tolist()
                batch_metrics['train/loss'].extend(train_losses)
            
            if accumulated >= config.gradient_accumulation_steps:
                lr_scheduler.step()
                with torch.no_grad():
                    mean_metrics = {}
                    for k, v in batch_metrics.items():
                        mean_metrics[k] = sum(v) / len(v)
                    mean_metrics['train/lr'] = lr_scheduler.get_last_lr()[0]
                    accelerator.print(f'train stats after {example_counter} examples: {formatted_dict(mean_metrics)}')
                    batch_metrics = defaultdict(list)
                if accelerator.is_main_process and config.debug == False:
                    wandb.log(mean_metrics, step=example_counter)

                delete_dicts(batch, batch_metrics, mean_metrics)
                accumulated = 0
                free_memory()

    save(policy, len(train_dataset), accelerator, tokenizer, config)