import torch.utils
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from datasets import Dataset as HfDataset
import itertools
from train import dataloader
from accelerate import Accelerator
import transformers
from train.dataloader import DPODataset, collate_fn
from transformers import set_seed
from train.dataloader import get_dataset, get_flat_data
from train import models
from train.utils import disable_dropout, move_batch_to_device, StreamingJSONWriter
from omegaconf import OmegaConf, DictConfig
from datetime import datetime
import torch
import hydra
import json
torch.backends.cuda.matmul.allow_tf32 = True
torch.autograd.grad_mode.set_grad_enabled(False)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    set_seed(config.seed)

    print('=' * 80)
    print(config.seed)

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Loading reward model from {config.reward_model_path}")
    
    print(f'Loading tokenizer at {config.reward_model_path}')
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.reward_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print('building reward model')
    reward_model_dtype = torch.bfloat16 
    reward_class = AutoModelForSequenceClassification
    reward_model = reward_class.from_pretrained(config.reward_model_path, torch_dtype=reward_model_dtype)
    
    reward_model = accelerator.prepare(reward_model)
    print(f'Loading dataloader')
    dataset = get_dataset(config.dataset, split=config.split)
    pref_dataset = []
    def generate_k_pairs(response_lst, K=16):
        result = []
        for pair in itertools.combinations(response_lst, 2):
            result.append(pair)
            # Stop when we have K pairs
            if len(result) == K:
                return result

    for sample in tqdm(dataset):
        prompt = sample['ctx']
        response_pairs = generate_k_pairs(sample['target'], 16)
        pref_dataset.extend([{"ctx": prompt, "chosen": response[0], "rejected": response[1]} for response in response_pairs])
    
    pref_dataset = HfDataset.from_list(pref_dataset)
    collate_func = partial(collate_fn, tokenizer=tokenizer)

    eval_dataset = DPODataset(tokenizer, pref_dataset, max_length=640)
    eval_iterator = torch.utils.data.DataLoader(eval_dataset, batch_size=config.model.eval_batch_size, shuffle=True, pin_memory=True, collate_fn=collate_func)
    eval_iterator = accelerator.prepare(eval_iterator)
    
    i = 0
    all_samples = []
    for batch in tqdm(eval_iterator):
        chosen_rewards = reward_model(
            batch['chosen_combined_input_ids'],
            batch['chosen_combined_attention_mask'],
            batch['chosen_labels']
        ).logits

        if i <= 5:
            print(chosen_rewards)
            i += 1
        
        rejected_rewards = reward_model(
            batch['rejected_combined_input_ids'],
            batch['rejected_combined_attention_mask'],
            batch['rejected_labels']
        ).logits

        for prompt, chosen, rejected, chosen_score, rejected_score in zip(batch['prompt_text'], batch['chosen_text'], batch['rejected_text'], chosen_rewards.tolist(), rejected_rewards.tolist()):
            data = {
                "prompt": prompt,
                "chosen": chosen if chosen_score > rejected_score else rejected,
                'rejected': rejected if chosen_score > rejected_score else chosen,
                'rejected_score': rejected_score,
                "chosen_score": chosen_score
            }
            all_samples.append(data)
        
    if accelerator.is_main_process:
        print(f"Writing feedback to {config.output_path}")
        output_file = open(config.output_path, 'w')
        writer = StreamingJSONWriter(output_file)
        for item in all_samples:
            writer.write_item(item)
        writer.close()
        output_file.close()
    
    accelerator.end_training()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    
if __name__=="__main__":
    main()