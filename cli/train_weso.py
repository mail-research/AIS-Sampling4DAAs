import time
import torch.nn.functional as F
import deepspeed
import yaml
from accelerate.state import AcceleratorState
from dataclasses import asdict
import gc
import wandb
from collections import defaultdict
from transformers import (
    AutoTokenizer,
    get_scheduler,
    set_seed,
    AutoModelForCausalLM,
)
import torch
from accelerate.utils import BnbQuantizationConfig
from torch import nn, optim
from accelerate import Accelerator, DistributedDataParallelKwargs
import json
import tyro
import sys
sys.path.append('/cluster/tufts/laolab/kdoan02/RLHF-training')
from dataset.data_utils import *
from config import WESOConfig
import os
from model import EFTWrapper
from utils import *
import transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True


if __name__=="__main__":
    config = tyro.cli(WESOConfig)
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
    world_size = accelerator.num_processes
    set_seed(config.seed)
    
    if config.debug == False and accelerator.is_main_process:
        wandb.init(
            project=config.project_name,
            config=asdict(config),
            name=config.exp_name
        )

    config_jsonl = asdict(config)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir, exist_ok=True)

    if not os.path.exists(config.local_run_dir):
        os.makedirs(config.local_run_dir, exist_ok=True)
    
    accelerator.print(config.local_run_dir)

    config_path = os.path.join(config.output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_jsonl, f)
    
    policy_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "use_cache": False
    }
    
    if config.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        policy = AutoLigerKernelForCausalLM.from_pretrained(
            config.model_path,
            **policy_kwargs
        )
        ref_policy = AutoLigerKernelForCausalLM.from_pretrained(
            config.model_path,
            **policy_kwargs
        )
    else:
        policy = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            **policy_kwargs
        )

        ref_policy = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            **policy_kwargs
        )
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    weak_tokenizer = AutoTokenizer.from_pretrained(config.base_weak_model_path)
    base_weak_model = AutoModelForCausalLM.from_pretrained(config.base_weak_model_path, **policy_kwargs)
    aligned_weak_model = AutoModelForCausalLM.from_pretrained(config.aligned_weak_model_path, **policy_kwargs)

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        policy.resize_token_embeddings(len(tokenizer))
        ref_policy.resize_token_embeddings(len(tokenizer))

    if weak_tokenizer.pad_token_id is None:
        weak_tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        base_weak_model.resize_token_embeddings(len(weak_tokenizer))
        aligned_weak_model.resize_token_embeddings(len(weak_tokenizer))

    train_dataset = globals()[f'get_{config.dataset_name}']('train')
    test_dataset = globals()[f'get_{config.dataset_name}']('test')
    train_dataset = W2SDataset(train_dataset, tokenizer, max_length=config.max_length, max_prompt_length=config.max_prompt_length, weak_tokenier=weak_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size // world_size, collate_fn=train_dataset.packing_collate_fn if config.use_packing else train_dataset.padding_collate_fn, pin_memory=True, num_workers=4, shuffle=True)

    if config.gradient_checkpointing:
        policy.gradient_checkpointing_enable(dict(use_reentrant=False))
        ref_policy.gradient_checkpointing_enable(dict(use_reentrant=False))

    optimizer = getattr(torch.optim, config.optimizer)(policy.parameters(), lr=config.lr, weight_decay=0.0, eps=1e-5)
    reference_accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )


    reference_accelerator.print("precomputing logprobs ...")
    reference_model = EFTWrapper(
        reference_accelerator, 
        base_weak_model,
        ref_policy, 
        weak_tokenizer,
        tokenizer, 
        config, 
        iterators=train_dataloader,
        algined_weak_model=aligned_weak_model
    )

    lr_scheduler = get_scheduler(
        'cosine_with_min_lr',
        optimizer,
        num_warmup_steps=config.warm_up_steps,
        num_training_steps=(len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps)) * config.num_train_epochs,
        scheduler_specific_kwargs={"min_lr": config.lr * 0.1},
    )
    print("Use cosine lr")

    accelerator.print(policy)
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (config.warm_up_steps + 1)))

    policy, optimizer, train_dataloader = accelerator.prepare(policy, optimizer, train_dataloader)
    deepspeed_states = AcceleratorState().deepspeed_plugin
    deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = config.batch_size // world_size
    eval_ds_config = {
        "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"],
        "bf16": {"enabled": True},
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }
    eval_ds_config["zero_optimization"] = {
        "stage": 3,
        "stage3_param_persistence_threshold": 1e4,
    }
    ref_policy, *_ = deepspeed.initialize(model=ref_policy, config=eval_ds_config)
    ref_policy.eval()

    example_counter = 0
    accumulated = 0
    batch_metrics= defaultdict(list)
    policy.train()

    for epoch in range(config.num_train_epochs):
        accelerator.print(f'Epoch: {epoch+1}')
        for batch in train_dataloader:
            if example_counter % config.eval_every == 0 and example_counter > 0:
                save(policy, example_counter, accelerator, tokenizer, config)

            start_time = time.time()
            example_counter += config.batch_size
            with torch.no_grad():
                weights = reference_model.get_weight(batch["target_combined_input_ids"]).to(accelerator.device)
            
            with accelerator.accumulate(policy):
                with accelerator.autocast():
                    logits = policy(
                        input_ids=batch['target_combined_input_ids'],
                        attention_mask=batch['target_combined_attention_mask'],
                        position_ids=batch['target_combined_position_ids']
                    ).logits.to(torch.float32)
                    policy_logps = get_batch_logps(logits, batch['target_labels'], average_log_prob=True)
                    losses = (-weights * policy_logps)
                    loss = losses.mean()
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    accumulated += 1
            
            step_time = time.time() - start_time
            examples_per_second = config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)

            with torch.no_grad():
                train_losses = accelerator.gather_for_metrics(losses).detach().float().cpu().numpy().tolist()
                gathered_logps = accelerator.gather_for_metrics(policy_logps).detach().cpu().numpy().tolist()
                gathered_weights = accelerator.gather_for_metrics(weights).detach().cpu().numpy().tolist()

                batch_metrics['loss/train'].extend(train_losses)
                batch_metrics['logps_train/policy_logps'].extend(gathered_logps)
                batch_metrics['logps_train/weights'].extend(gathered_weights)

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