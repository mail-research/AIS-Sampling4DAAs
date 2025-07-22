from transformers import AutoModelForCausalLM, AutoTokenizer
import jsonlines
import time
import torch.nn.functional as F
import deepspeed
from accelerate.state import AcceleratorState
from dataclasses import asdict
import gc
import numpy as np
import wandb
from accelerate.utils import gather_object
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
from dataset.data_utils import *
from config import Config, GRPOConfig
import os
from utils import *
import transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True

def safe_zip(*lists):
    """Zips lists together, ensuring they have the same length."""
    if not lists:
        return  # Handle empty input

    first_length = len(lists[0])
    for lst in lists[1:]:
        if len(lst) != first_length:
            raise ValueError("All lists must have the same length.")

    return zip(*lists)



if __name__=="__main__":
    config = tyro.cli(GRPOConfig)
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
        "use_cache": True
    }

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path,
        num_labels=1,
        **policy_kwargs
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    accelerator.print(tokenizer.pad_token)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    reward_model.resize_token_embeddings(len(tokenizer))
    reward_model.config.pad_token_id = tokenizer.pad_token_id

    accelerator.print(tokenizer.pad_token)

    policy = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        **policy_kwargs
    )

    ref_policy = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        **policy_kwargs
    )

    train_dataset = globals()[f'get_{config.dataset_name}']('train')
    test_dataset = globals()[f'get_{config.dataset_name}']('test')
    import random
    random.shuffle(train_dataset)
    train_dataset = TorchDataset(train_dataset, tokenizer, max_length=config.max_length, max_prompt_length=config.max_prompt_length)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size // world_size, collate_fn=train_dataset.packing_collate_fn if config.use_packing else train_dataset.padding_collate_fn, pin_memory=True, num_workers=4, shuffle=True)

    # if config.gradient_checkpointing:
    #     reward_model.gradient_checkpointing_enable(dict(use_reentrant=False))

    optimizer = getattr(torch.optim, config.optimizer)(policy.parameters(), lr=config.lr, weight_decay=0.0, eps=1e-5)
    policy, optimizer, train_dataloader = accelerator.prepare(policy, optimizer, train_dataloader)
    # eval_dataloader = accelerator.prepare(test_dataloader)
    example_counter = 0
    accumulated = 0
    batch_metrics= defaultdict(list)
    policy.train()
    ref_policy = prepare_deepspeed(ref_policy, config)
    reward_model = prepare_deepspeed(reward_model, config)

    assert config.rloo_k == 2

    generation_kwargs = {
        "temperature": 0.9,
        "top_p": 1.0,
        "max_new_tokens": 128,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    MAX_TOKENS = generation_kwargs['max_new_tokens']
    device = accelerator.device

    def repeat_generator():
        while True:
            yield from train_dataloader

    iter_dataloader = iter(repeat_generator())

    n_iter = 0
    batch_metrics = defaultdict(list)
    all_items = []
    eval_every = 60 * config.batch_size * config.gradient_accumulation_steps
    accelerator.print(config.num_iters)

    with open("samples/online_ipo_samples.jsonl", mode="w") as f:
        writer = StreamingJSONWriter(f)

        while n_iter < config.num_iters:
            batch = next(iter_dataloader)
            if example_counter % eval_every == 0 and example_counter > 0:
                save(policy, example_counter, accelerator, tokenizer, config)

            prompts = batch['prompt_text']
            start_time = time.time()
            batch_size = len(prompts)
            example_counter += config.batch_size
            prompt_ids = batch['prompt_input_ids']
            batch['prompt_input_ids'] = batch['prompt_input_ids'].repeat(2, 1)
            batch['prompt_attention_mask'] = batch['prompt_attention_mask'].repeat(2, 1)
            prompt_mask = batch['prompt_attention_mask']
            # prompt_mask = prompt_mask.repeat(2, 1)
            context_length = prompt_ids.size(1)

            with torch.no_grad():
                with unwrap_model_for_generation(policy, accelerator) as unwrapped_model:
                    policy_output = get_batch_samples(batch, unwrapped_model, tokenizer, return_tensors=True, **generation_kwargs)
                    completion_ids = policy_output[:, context_length:]
                    completion_mask = get_completion_mask(completion_ids, tokenizer, device)
                    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
                    position_ids = torch.cumsum(attention_mask, dim=-1) - 1
                    position_ids.masked_fill_(~attention_mask.bool(), 0)
                    sequence_lengths = torch.eq(completion_ids, tokenizer.pad_token_id).int().argmax(-1) - 1
                    sequence_lengths = sequence_lengths % completion_ids.shape[-1]
                    sequence_lengths = sequence_lengths.to(accelerator.device) + context_length

                ref_logits = ref_policy(policy_output, attention_mask=attention_mask, position_ids=position_ids).logits.to(torch.float32)
                ref_logits = ref_logits[:, context_length-1:-1, :]
                ref_per_logprobs = selective_log_softmax(ref_logits, completion_ids)
                transformer_output = reward_model.model(policy_output, attention_mask=attention_mask, position_ids=position_ids)
                hidden_states = transformer_output[0]
                logits = reward_model.score(hidden_states).squeeze(-1) # (B, S)
                rewards = logits[torch.arange(policy_output.size(0), device=accelerator.device), sequence_lengths]
                loss_mask = (completion_ids != tokenizer.pad_token_id).long()
                first_half, second_half = rewards.split(batch_size)
                # Get the indices of the chosen and rejected examples
                mask = first_half >= second_half
                batch_range = torch.arange(batch_size, device=device)
                chosen_indices = batch_range + (~mask * batch_size)
                rejected_indices = batch_range + (mask * batch_size)
                cr_indices = torch.cat((chosen_indices, rejected_indices), dim=0)  # cr = chosen and rejected
                completion_mask = completion_mask[cr_indices]
                ref_logprobs = ref_per_logprobs[cr_indices]
                ref_logprobs = (ref_logprobs * completion_mask).sum(dim=1)
                torch.cuda.empty_cache()
            
            with accelerator.accumulate(policy):
                with accelerator.autocast():
                    logits = policy(policy_output, attention_mask=attention_mask, position_ids=position_ids).logits.to(torch.float32)
                    logits = logits[:, context_length-1:-1, :]
                    per_logprobs = selective_log_softmax(logits, completion_ids)
                    logprobs = per_logprobs[cr_indices]
                    logprobs = (logprobs * completion_mask).sum(dim=1)
                    chosen_logprobs, rejected_logprobs = torch.split(logprobs, batch_size)
                    ref_chosen_logprobs, ref_rejected_logprobs = torch.split(ref_logprobs, batch_size)
                    pi_logratios = chosen_logprobs - rejected_logprobs
                    ref_logratios = ref_chosen_logprobs - ref_rejected_logprobs
                    logits = pi_logratios - ref_logratios
                    losses = (logits - 1 / (2 * config.beta)) ** 2
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
                kl_sequences = logprobs - ref_logprobs
                chosen_rewards = config.beta * (chosen_logprobs - ref_chosen_logprobs)
                rejected_rewards = config.beta * (rejected_logprobs - ref_rejected_logprobs)
                reward_margins = chosen_rewards - rejected_rewards
                scores_margins = rewards[chosen_indices] - rewards[rejected_indices]
                accuracy = reward_margins > 0.0

                kl_sequences = accelerator.gather_for_metrics(kl_sequences).detach().float().cpu().numpy().tolist()
                total_rewards = accelerator.gather_for_metrics(rewards).detach().float().cpu().numpy().tolist()
                accuracy = accelerator.gather_for_metrics(accuracy).detach().float().cpu().numpy().tolist()
                reward_margins = accelerator.gather_for_metrics(reward_margins).detach().float().cpu().numpy().tolist()
                scores_margins = accelerator.gather_for_metrics(scores_margins).detach().float().cpu().numpy().tolist()
                
                chosen_rewards = accelerator.gather_for_metrics(chosen_rewards).detach().float().cpu().numpy().tolist()
                rejected_rewards = accelerator.gather_for_metrics(rejected_rewards).detach().float().cpu().numpy().tolist()

                rejected_logprobs  = accelerator.gather_for_metrics(rejected_logprobs).detach().float().cpu().numpy().tolist()
                chosen_logprobs = accelerator.gather_for_metrics(chosen_logprobs).detach().float().cpu().numpy().tolist()
                ref_rejected_logprobs  = accelerator.gather_for_metrics(ref_rejected_logprobs).detach().float().cpu().numpy().tolist()
                ref_chosen_logprobs = accelerator.gather_for_metrics(ref_chosen_logprobs).detach().float().cpu().numpy().tolist()

                completion_length = accelerator.gather_for_metrics(completion_mask.sum(dim=1)).detach().float().cpu().numpy().tolist()

                batch_metrics['loss/train'].extend(train_losses)
                batch_metrics['metrics/kl'].extend(kl_sequences)
                batch_metrics['metrics/chosen_logps'].extend(chosen_logprobs)
                batch_metrics['metrics/rejected_logps'].extend(rejected_logprobs)
                batch_metrics['rewards/reward_margins'].extend(reward_margins)
                batch_metrics['rewards/scores_margins'].extend(scores_margins)
                batch_metrics['rewards/chosen_rewards'].extend(chosen_rewards)
                batch_metrics['rewards/scores'].extend(total_rewards)
                batch_metrics['rewards/accuracy'].extend(accuracy)
                batch_metrics['rewards/rejected_rewards'].extend(rejected_rewards)
                batch_metrics['metrics/completion_length'].extend(completion_length)

                completion_ids = completion_ids[cr_indices]
                chosen_completion_ids, rejected_completion_ids = torch.split(completion_ids, batch_size)

                chosen_completions = gather_object(tokenizer.batch_decode(chosen_completion_ids, skip_special_tokens=True))
                rejected_completions = gather_object(tokenizer.batch_decode(rejected_completion_ids, skip_special_tokens=True))
                prompt_text = gather_object(tokenizer.batch_decode(prompt_ids, skip_special_tokens=True))

                if accelerator.is_main_process:
                    for prompt, chosen_completion, rejected_completion, chosen_logprob, rejected_logprob, ref_chosen_logp, ref_rejected_logp in safe_zip(prompt_text, chosen_completions, rejected_completions, chosen_logprobs, rejected_logprobs, ref_chosen_logprobs, ref_rejected_logprobs):
                        item = {
                            "prompt": prompt,
                            "chosen_completion": chosen_completion,
                            "rejected_completion": rejected_completion,
                            "chosen_logprobs": chosen_logprob,
                            "rejected_logprobs": rejected_logprob,
                            "ref_chosen_logprobs": ref_chosen_logp,
                            "ref_rejected_logprobs": ref_rejected_logp
                        }
                        all_items.append(item)
            
            if accumulated >= config.gradient_accumulation_steps:
                n_iter += 1
                with torch.no_grad():
                    mean_metrics = {}
                    for k, v in batch_metrics.items():
                        mean_metrics[k] = sum(v) / len(v)
                    accelerator.print(f'train stats after iteration {n_iter} with {example_counter} examples: {formatted_dict(mean_metrics)}')
                    batch_metrics = defaultdict(list)
                if accelerator.is_main_process:
                    if config.debug == False:
                        wandb.log(mean_metrics, step=example_counter)
                    # writer.write_all(all_items)
                    for item in all_items:
                        writer.write_item(item)

                all_items = []
                delete_dicts(batch, batch_metrics, mean_metrics)
                accumulated = 0
                free_memory()
        writer.close()

    save(policy, len(train_dataset), accelerator, tokenizer, config)
