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
from accelerate import Accelerator
import json
import tyro
import sys
from dataset.reward_dataset import *
from config import DPOConfig
import os
from utils import *
import transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True

def forward(model, batch, tokenizer):
    concatenated_batch = concatneted_inputs(batch, tokenizer)

    all_logits = model(
        concatenated_batch['concatenated_combined_input_ids'], 
        attention_mask=concatenated_batch['concatenated_combined_attention_mask'],
    ).logits.to(torch.float32)
    
    all_logps = get_batch_logps(all_logits, concatenated_batch['concatenated_labels'])
    chosen_logps = all_logps[:batch['chosen_combined_input_ids'].shape[0], ...]
    rejected_logps = all_logps[batch['chosen_combined_input_ids'].shape[0]:, ...]
    return chosen_logps, rejected_logps

if __name__=="__main__":
    config = tyro.cli(DPOConfig)
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

    config_path = os.path.join(config.output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_jsonl, f)
    
    policy_kwargs = {
        "torch_dtype": torch.bfloat16 if "gpt2" not in config.model_path else torch.float32,
        "attn_implementation": "flash_attention_2" if "gpt2" not in config.model_path else "eager",
        "use_cache": False
    }

    if not config.use_liger:
        policy = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            **policy_kwargs
        )
        ref_policy = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            **policy_kwargs
        )
    else:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        policy = AutoLigerKernelForCausalLM.from_pretrained(
            config.model_path,
            **policy_kwargs
        )
        ref_policy = AutoLigerKernelForCausalLM.from_pretrained(
            config.model_path,
            **policy_kwargs
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        policy.resize_token_embeddings(len(tokenizer))
        ref_policy.resize_token_embeddings(len(tokenizer))

    train_dataset = globals()[f'get_{config.dataset_name}']('train')
    test_dataset = globals()[f'get_{config.dataset_name}']('test')
    accelerator.print(f"Number of training data: {len(train_dataset)}")
    accelerator.print(f"Number of testing data: {len(test_dataset)}")

    train_dataset = RewardDataset(train_dataset, tokenizer, max_length=config.max_length, max_prompt_length=config.max_prompt_length)
    test_dataset = RewardDataset(test_dataset, tokenizer, max_length=config.max_length, max_prompt_length=config.max_prompt_length)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size // world_size, collate_fn=train_dataset.packing_collate_fn if config.use_packing else train_dataset.padding_collate_fn, pin_memory=True, num_workers=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.eval_batch_size // world_size, collate_fn=train_dataset.packing_collate_fn if config.use_packing else train_dataset.padding_collate_fn, pin_memory=True, num_workers=4, shuffle=True)

    if config.gradient_checkpointing:
        policy.gradient_checkpointing_enable(dict(use_reentrant=False))
        ref_policy.gradient_checkpointing_enable(dict(use_reentrant=False))

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
    eval_dataloader = accelerator.prepare(test_dataloader)
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
            if "ref_chosen_logps" in batch.keys() or "ref_rejected_logps" in batch.keys():
                ref_chosen_logps = torch.tensor(batch['ref_chosen_logps'], device=accelerator.device)
                ref_rejected_logps = torch.tensor(batch['ref_rejected_logps'], device=accelerator.device)

            else:
                with torch.inference_mode():
                    ref_chosen_logps, ref_rejected_logps = forward(ref_policy, batch, tokenizer)
            
            with accelerator.accumulate(policy):
                with accelerator.autocast():
                    chosen_logps, rejected_logps = forward(policy, batch, tokenizer)
                    pi_logratios = chosen_logps - rejected_logps
                    ref_logratios = ref_chosen_logps - ref_rejected_logps
                    logits = pi_logratios - ref_logratios 
                    losses = -F.logsigmoid(config.beta * logits)
                    # losses = (logits - 1 / (2 * config.beta)) ** 2
                    loss = losses.mean()
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    accumulated += 1
            
            step_time = time.time() - start_time
            examples_per_second = config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)

            with torch.no_grad():
                chosen_rewards = config.beta * (chosen_logps - ref_chosen_logps).detach()
                rejected_rewards = config.beta * (rejected_logps - ref_rejected_logps).detach()

                train_losses = accelerator.gather_for_metrics(losses).detach().float().cpu().numpy().tolist()
                gathered_chosen_logps = accelerator.gather_for_metrics(chosen_logps).detach().cpu().numpy().tolist()
                gathered_rejected_logps = accelerator.gather_for_metrics(rejected_logps).detach().cpu().numpy().tolist()   
                margins = accelerator.gather_for_metrics((chosen_rewards - rejected_rewards)).detach().float().cpu().numpy().tolist()
                train_accuracy = (chosen_rewards > rejected_rewards).detach().float().cpu().numpy().tolist()
                gathered_chosen_rewards = accelerator.gather_for_metrics(config.beta * (chosen_logps - ref_chosen_logps)).detach().float().cpu().numpy().tolist()
                gathered_rejected_rewards = accelerator.gather_for_metrics(config.beta * (rejected_logps - ref_rejected_logps)).detach().float().cpu().numpy().tolist()

                batch_metrics['loss/train'].extend(train_losses)
                batch_metrics['rewards_train/margins'].extend(margins)
                batch_metrics['rewards_train/chosen_rewards'].extend(gathered_chosen_rewards)
                batch_metrics['rewards_train/rejected_rewards'].extend(gathered_rejected_rewards)
                batch_metrics['rewards_train/accuracy'].extend(train_accuracy)
                batch_metrics['logps_train/chosen_logps'].extend(gathered_chosen_logps)
                batch_metrics['logps_train/rejected_logps'].extend(gathered_rejected_logps)
                del margins, chosen_rewards, rejected_rewards

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