from accelerate import Accelerator
import sys
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
import torch
sys.path.append('/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/benchmark/RLHF-training')
from dataset.reward_dataset import *
from config import EvaluateConfig
from model import ImplicitReward
import itertools
from utils import StreamingJSONWriter
from config import Config
import os
from utils import *
from tqdm import tqdm
import tyro
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.autograd.grad_mode.set_grad_enabled(False)


if __name__=="__main__":
    config = tyro.cli(EvaluateConfig)
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(config.judge_path)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    if not config.implicit:
        reward_model = AutoModelForSequenceClassification.from_pretrained(config.judge_path, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map='auto')
    else:
        aligned_weak_model = AutoModelForCausalLM.from_pretrained(config.model_path, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map='auto')
        base_weak_model = AutoModelForCausalLM.from_pretrained(config.ref_model_path, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map='auto')
        reward_model = ImplicitReward(accelerator, base_weak_model, aligned_weak_model)
    # eval_dataset = globals()[f'get_{config.dataset_name}'](config.split)
    eval_dataset = globals()[f'get_{config.dataset_name}'](config.split, flatten=False)
    pref_dataset = []
    
    def generate_k_pairs(response_lst, K=4):
        result = []
        for pair in itertools.combinations(response_lst, 2):
            result.append(pair)
            # Stop when we have K pairs
            if len(result) == K:
                return result
        
    for sample in tqdm(eval_dataset):
        prompt = sample['prompt']
        response_pairs = generate_k_pairs(sample['completion'], 4)
        pref_dataset.extend([{"prompt": prompt, "chosen": response[0], "rejected": response[1]} for response in response_pairs])
    eval_dataset = RewardDataset(pref_dataset, tokenizer, max_prompt_length=1024, max_length=2048)
    
    # eval_dataset = RewardDataset(eval_dataset, tokenizer)
    world_size = accelerator.num_processes
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size // world_size, collate_fn=eval_dataset.padding_collate_fn, pin_memory=True, num_workers=8, shuffle=False)
    if not config.implicit:
        reward_model.config.pad_token_id = tokenizer.pad_token_id
    total_acc = []
    reward_model, eval_dataloader = accelerator.prepare(reward_model, eval_dataloader)

    with open(f'samples/label/{config.dataset_name}_{config.judge_path.split("/")[1]}_{config.split}.jsonl', 'w') as f:
        writer = StreamingJSONWriter(f)
        tqdm_loader = tqdm(eval_dataloader)
        for batch in tqdm_loader:
            chosen_scores = reward_model(
                batch['chosen_combined_input_ids'],
                attention_mask=batch['chosen_combined_attention_mask'],
                labels=batch['chosen_labels']
            ).logits.squeeze(-1).to(torch.float32)

            rejected_scores = reward_model(
                batch['rejected_combined_input_ids'],
                attention_mask=batch['rejected_combined_attention_mask'],
                labels=batch['rejected_labels']
            ).logits.squeeze(-1).to(torch.float32)

            accuracy = (chosen_scores > rejected_scores).float().cpu().numpy().tolist()
            total_acc.extend(accuracy)
            chosen_scores = chosen_scores.cpu().numpy().tolist()
            rejected_scores = rejected_scores.cpu().numpy().tolist()
            for prompt, chosen_txt, rejected_txt, chosen_score, rejected_score in zip(batch['prompt_text'], batch['chosen_text'], batch['rejected_text'], chosen_scores, rejected_scores):
                item = {
                    "prompt": prompt,
                    "chosen": chosen_txt if chosen_score > rejected_score else rejected_txt,
                    "rejected": rejected_txt if chosen_score > rejected_score else chosen_txt,
                    "chosen_score": max(chosen_score, rejected_score),
                    "rejected_score": min(chosen_score, rejected_score)
                }
                writer.write_item(item)
            tqdm_loader.set_description_str(f"Accuracy: {sum(total_acc) / len(total_acc)}")
        print(f"Accuracy: {sum(total_acc) / len(total_acc)}")
        writer.close()

                    # "chosen_logps": chosen_logp if chosen_score > rejected_score else rejected_logp,
                    # "rejected_logps": rejected_logp if chosen_score > rejected_score else chosen_logp,
                    # "ref_chosen_logps": ref_chosen_logp if chosen_score > rejected_score else ref_rejected_logp,
                    # "ref_rejected_logps": ref_rejected_logp if chosen_score > rejected_score else ref_chosen_logp,

