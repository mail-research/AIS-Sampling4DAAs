import time
import torch.nn.functional as F
import deepspeed
from accelerate.state import AcceleratorState
from dataclasses import asdict
import gc
import numpy as np
import wandb
import bitsandbytes
from collections import defaultdict
from transformers import (
    AutoTokenizer,
    set_seed,
    get_scheduler,
    AutoModelForSequenceClassification,
)
import torch
from accelerate.utils import BnbQuantizationConfig
from torch import nn, optim
from accelerate import Accelerator
import json
import tyro
import sys
sys.path.append('/home/phuc/project/rlhf_training')
from dataset.reward_dataset import *
from config import Config, DPOConfig
import os
from utils import *
import transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True


def forward(model, batch, is_packing: bool = True):
    if not is_packing: 
        batch = concatneted_inputs(batch)
    
    all_logits = model(
        batch['concatenated_input_ids'], 
        attention_mask=batch['concatenated_attention_mask'],
        position_ids=batch['concatenated_position_ids']
    ).logits.to(torch.float32)
    if not is_packing:
        all_logps = get_batch_logps(all_logits, batch['concatenated_labels'])
    else:
        all_logps = packed_get_batch_logps(all_logits, batch['concatenated_labels'], batch['seq_len'])

    chosen_logps = all_logps[:len(batch['seq_len'])//2, ...]
    rejected_logps = all_logps[len(batch['seq_len'])//2:, ...]
    return chosen_logps, rejected_logps

if __name__=="__main__":
    config = tyro.cli(Config)
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
    world_size = accelerator.num_processes
    set_seed(config.seed)

    if config.debug == False and accelerator.is_main_process:
        wandb.init(
            project='rlhf-training',
            config=asdict(config),
            name=config.exp_name
        )

    policy_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "num_labels": 1,
        "use_cache": False
    }

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.model_path,
        **policy_kwargs
    )

    reward_model.score = layer_init(
        reward_model.score,
        std=1 / np.sqrt(reward_model.config.hidden_size + 1),
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    accelerator.print(tokenizer.pad_token)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    reward_model.resize_token_embeddings(len(tokenizer))
    accelerator.print(tokenizer.pad_token)
    accelerator.print(reward_model)
    
    reward_model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = globals()[f'get_{config.dataset_name}']('train')
    test_dataset = globals()[f'get_{config.dataset_name}']('test')
    train_dataset = RewardDataset(train_dataset, tokenizer, max_length=config.max_length, max_prompt_length=config.max_prompt_length)
    test_dataset = RewardDataset(test_dataset, tokenizer, max_length=config.max_length, max_prompt_length=config.max_prompt_length)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size // world_size, collate_fn=train_dataset.packing_collate_fn if config.use_packing else train_dataset.padding_collate_fn, pin_memory=True, num_workers=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.eval_batch_size // world_size, collate_fn=train_dataset.packing_collate_fn if config.use_packing else train_dataset.padding_collate_fn, pin_memory=True, num_workers=8, shuffle=True)

    if config.gradient_checkpointing:
        reward_model.gradient_checkpointing_enable(dict(use_reentrant=False))

    optimizer = getattr(torch.optim, config.optimizer)(reward_model.parameters(), lr=config.lr, weight_decay=0.0, eps=1e-5)
    
    lr_scheduler = get_scheduler(
        'cosine_with_min_lr',
        optimizer,
        num_warmup_steps=config.warm_up_steps,
        num_training_steps=(len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps)) * config.num_train_epochs,
        scheduler_specific_kwargs={"min_lr": config.lr * 0.1},
    )

    reward_model, optimizer, train_dataloader = accelerator.prepare(reward_model, optimizer, train_dataloader)
    eval_dataloader = accelerator.prepare(test_dataloader)
    example_counter = 0
    accumulated = 0
    batch_metrics= defaultdict(list)
    reward_model.train()

    for batch in train_dataloader:
        if example_counter % config.eval_every == 0 and example_counter > 0:
            save(reward_model, example_counter, accelerator, tokenizer, config)
        
        start_time = time.time()
        example_counter += config.batch_size
        concatnated_batch = concatneted_inputs(batch, tokenizer)

        with accelerator.accumulate(reward_model):
            with accelerator.autocast():
                reward_logits = reward_model(
                    input_ids=concatnated_batch['concatenated_combined_input_ids'],
                    attention_mask=concatnated_batch['concatenated_combined_attention_mask'],
                ).logits.squeeze(-1).to(torch.float32)
                chosen_logits = reward_logits[:reward_logits.size(0) // 2]
                rejected_logits = reward_logits[reward_logits.size(0) // 2:]
                losses = -F.logsigmoid(chosen_logits - rejected_logits)
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
            margins = accelerator.gather_for_metrics((chosen_logits - rejected_logits)).detach().float().cpu().numpy().tolist()
            train_accuracy = (chosen_logits > rejected_logits).detach().float().cpu().numpy().tolist()
            gathered_chosen_rewards = accelerator.gather_for_metrics(chosen_logits).detach().float().cpu().numpy().tolist()
            gathered_rejected_rewards = accelerator.gather_for_metrics(rejected_logits).detach().float().cpu().numpy().tolist()

            batch_metrics['loss/train'].extend(train_losses)
            batch_metrics['rewards_train/margins'].extend(margins)
            batch_metrics['rewards_train/chosen_rewards'].extend(gathered_chosen_rewards)
            batch_metrics['rewards_train/rejected_rewards'].extend(gathered_rejected_rewards)
            batch_metrics['rewards_train/accuracy'].extend(train_accuracy)

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

    save(reward_model, len(train_dataset), accelerator, tokenizer, config)