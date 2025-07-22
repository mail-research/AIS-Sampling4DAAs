from torch.utils.data import Dataset, DataLoader
import random
import json
from typing import List
import numpy as np
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset as hf_dataset
from collections import defaultdict
from typing import Dict
from torch.nn.utils.rnn import pad_sequence
import torch

class TorchDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        max_length=640,
        max_prompt_length=512,
        truncation_mode='keep_start',
        *args, 
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode

    def tokenize_batch_element(self, prompt: str, generation: str, truncation_mode: str, prefix: str='target') -> Dict:
        """
        Tokenize a single batch element and truncate if prompt + generation is too long. Batch element is turned into Pytorch 
        tensors in self.collate. Create the labels for the generation, which are of length equal to the sum of the length of 
        the prompt and the generation, with -100 for the prompt tokens.

        Args:
        - prompt: the input/instruction text
        - generation: output text
        - truncation_mode: one of 'keep_start'/'keep_end' (truncate end/beginning of combined text respectively)
        - prefix: the prefix corresponding to the generation (e.g., 'chosen', 'rejected', 'target')

        Returns:
            A dict of the tokenized prompt, tokenized generation, and the concatenation of the two on all relevant elements
            (e.g., tokens, attention mask, etc.). The generation elements will have keys starting with '{prefix}_' and the
            concatenated elements will have keys starting with '{prefix}_combined_'.
        """
        prompt_token_ids = self.tokenizer.encode(prompt)
        generation_token_ids = self.tokenizer.encode(generation)

        # clip EOS token at end of input
        if len(prompt_token_ids) > 0 and prompt_token_ids[-1] == self.tokenizer.eos_token_id:
            prompt_token_ids.pop()

        # clip BOS token at start of output
        if len(generation_token_ids) > 0 and generation_token_ids[0] == self.tokenizer.bos_token_id:
            generation_token_ids.pop(0)

        # clip EOS at end of output since it will be added later anyway
        if len(generation_token_ids) > 0 and generation_token_ids[-1] == self.tokenizer.eos_token_id:
            generation_token_ids.pop()

        # if combined sequence is too long, first truncate prompt
        if (len(prompt_token_ids) + len(generation_token_ids) > self.max_length) and (len(prompt_token_ids) > self.max_prompt_length):
            if truncation_mode == 'keep_start':
                prompt_token_ids = prompt_token_ids[:self.max_prompt_length]
            elif truncation_mode == 'keep_end':
                prompt_token_ids = prompt_token_ids[-self.max_prompt_length:]
            else:
                raise ValueError(f'Unknown truncation mode: {truncation_mode}')

        # then truncate generation if needed
        if (len(prompt_token_ids) + len(generation_token_ids) > self.max_length):
            generation_token_ids = generation_token_ids[:(self.max_length - len(prompt_token_ids))]

        # reconstitute the prompt and generation
        prompt = self.tokenizer.decode(prompt_token_ids, skip_special_tokens=True)
        generation = self.tokenizer.decode(generation_token_ids, skip_special_tokens=True) + ' ' + self.tokenizer.eos_token

        batch_element = { 'prompt_text' : prompt, f'{prefix}_text': generation }

        for k,v in self.tokenizer(prompt).items():
            batch_element[f'prompt_{k}'] = v

        for k,v in self.tokenizer(generation).items():
            batch_element[f'{prefix}_{k}'] = v

        # combine the prompt and generation belonging to the same example
        batch_element.update(self.combine_prompt_and_generation(batch_element, batch_element, prefix=prefix))
  
        return batch_element

    def combine_prompt_and_generation(self, prompt_dict: Dict, generation_dict: Dict, prefix: str='target') -> Dict:
        """
        Tokenize the concatenated prompt and generation. 
        
        Note that you cannot just concatenate the input ids, attention mask, etc. after the fact -- as done 
        in the DPO repo -- because of subtle differences. For example, the ID for 'Well' corresponds to no 
        space ('Well') when at the start of a text but a space ('\n Well) when succeeding a newline. Therefore
        we could not get the correct token ID for '\nWell' by first tokenizing '\n' then 'Well' then concatenating
        the resulting tokens together.

        The prefix for each concantenated element will be f'{prefix}_combined_'.

        Args:
        - prompt_dict: dict of the prompt text, tokens, attention mask, etc.
        - generation_dict: dict of the generation text, tokens, attention mask, etc.
        - prefix: str to prepend to the the keys of the tokenized (prompt + generation)

        Returns:
            A dict of the (prompt + generation) text, tokens, attention mask, etc, along with the labels for the
            joint sequence, where the prompt token labels have been set to -100.
        """
        combined_dict = { f'{prefix}_combined_text' : prompt_dict['prompt_text'] + generation_dict[f'{prefix}_text'] }

        for k,v in self.tokenizer(prompt_dict['prompt_text'] + generation_dict[f'{prefix}_text']).items():
            combined_dict[f'{prefix}_combined_{k}'] = v

        combined_dict[f'{prefix}_labels'] = combined_dict[f'{prefix}_combined_input_ids'][:]  # contains both input and response (unpadded)
        combined_dict[f'{prefix}_labels'][:len(prompt_dict['prompt_input_ids'])] = [-100] * len(prompt_dict['prompt_input_ids'])

        return combined_dict
    
    def __getitem__(self, index):
        element = self.dataset[index]
        batch_element =  self.tokenize_batch_element(
            element['prompt'],
            element['completion'],
            truncation_mode='keep_start' if '\n\nHuman:' not in element['prompt'] else 'keep_end'
        )
        if "logprob" in element.keys():
            batch_element["logprob"] = element["logprob"]
        return batch_element
    
    def __len__(self):
        return len(self.dataset)

    def padding_collate_fn(self, batch):
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:
                    # flip prompt so that you are padding to the beginning
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                if k.endswith('_input_ids'):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                # Always pad to max_length for consistency across processes
                max_len = self.max_prompt_length if 'prompt' in k else self.max_length
                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        if 'target_combined_attention_mask' in padded_batch:
            position_ids = torch.cumsum(padded_batch['target_combined_attention_mask'], dim=-1) - 1
            position_ids.masked_fill_(~padded_batch['target_combined_attention_mask'].bool(), 0)
            padded_batch['target_combined_position_ids'] = position_ids
        else:
            chosen_position_ids = torch.cumsum(padded_batch['chosen_combined_attention_mask'], dim=-1) - 1
            chosen_position_ids.masked_fill_(~padded_batch['chosen_combined_attention_mask'].bool(), 0)
            padded_batch['chosen_combined_position_ids'] = chosen_position_ids

            rejected_position_ids = torch.cumsum(padded_batch['rejected_combined_attention_mask'], dim=-1) - 1
            rejected_position_ids.masked_fill_(~padded_batch['rejected_combined_attention_mask'].bool(), 0)
            padded_batch['rejected_combined_position_ids'] = rejected_position_ids
        return padded_batch
    
    def packing_collate_fn(self, batch):
        packed_batch = {}
        packed_batch['target_combined_position_ids'] = []
        
        index = 1
        for example in batch:
            for k, v in example.items():
                if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                    max_len = self.max_prompt_length if 'prompt' in k else self.max_length
                    if k.endswith('_attention_mask') and k.startswith('prompt') == False:
                        if k not in packed_batch:
                            packed_batch[k] = []
                        value = torch.full_like(torch.tensor(example['target_combined_input_ids'][:max_len]), index)
                        packed_batch[k].append(value.flatten())
                    else:
                        if k not in packed_batch:
                            packed_batch[k] = []
                        v = v[:max_len]
                        packed_batch[k].append(torch.tensor(v).flatten())
                # else:
                #     packed_batch[k].append(v)
            packed_batch['target_combined_position_ids'].append(torch.tensor(list(range(len(example['target_combined_input_ids'][:max_len])))))
            index += 1
        
        for k, v in packed_batch.items():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels') or k.endswith('_position_ids'):
                packed_batch[k] = torch.cat(packed_batch[k], dim=0).unsqueeze(0)
        
        return packed_batch
    
class W2SDataset(TorchDataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        weak_tokenier,
        max_length=640,
        max_prompt_length=512,
        truncation_mode='keep_start',
        *args, 
        **kwargs
    ):
        super().__init__(dataset, tokenizer, max_length, max_prompt_length, truncation_mode, *args, **kwargs)
        self.weak_tokenizer = weak_tokenier
    
    def tokenize_batch_element(self, prompt: str, generation: str, truncation_mode: str, tokenizer, prefix: str='target') -> Dict:
        """
        Tokenize a single batch element and truncate if prompt + generation is too long. Batch element is turned into Pytorch 
        tensors in self.collate. Create the labels for the generation, which are of length equal to the sum of the length of 
        the prompt and the generation, with -100 for the prompt tokens.

        Args:
        - prompt: the input/instruction text
        - generation: output text
        - truncation_mode: one of 'keep_start'/'keep_end' (truncate end/beginning of combined text respectively)
        - prefix: the prefix corresponding to the generation (e.g., 'chosen', 'rejected', 'target')

        Returns:
            A dict of the tokenized prompt, tokenized generation, and the concatenation of the two on all relevant elements
            (e.g., tokens, attention mask, etc.). The generation elements will have keys starting with '{prefix}_' and the
            concatenated elements will have keys starting with '{prefix}_combined_'.
        """
        prompt_token_ids = tokenizer.encode(prompt)
        generation_token_ids = tokenizer.encode(generation)

        # clip EOS token at end of input
        if len(prompt_token_ids) > 0 and prompt_token_ids[-1] == tokenizer.eos_token_id:
            prompt_token_ids.pop()

        # clip BOS token at start of output
        if len(generation_token_ids) > 0 and generation_token_ids[0] == tokenizer.bos_token_id:
            generation_token_ids.pop(0)

        # clip EOS at end of output since it will be added later anyway
        if len(generation_token_ids) > 0 and generation_token_ids[-1] == tokenizer.eos_token_id:
            generation_token_ids.pop()

        # if combined sequence is too long, first truncate prompt
        if (len(prompt_token_ids) + len(generation_token_ids) > self.max_length) and (len(prompt_token_ids) > self.max_prompt_length):
            if truncation_mode == 'keep_start':
                prompt_token_ids = prompt_token_ids[:self.max_prompt_length]
            elif truncation_mode == 'keep_end':
                prompt_token_ids = prompt_token_ids[-self.max_prompt_length:]
            else:
                raise ValueError(f'Unknown truncation mode: {truncation_mode}')

        # then truncate generation if needed
        if (len(prompt_token_ids) + len(generation_token_ids) > self.max_length):
            generation_token_ids = generation_token_ids[:(self.max_length - len(prompt_token_ids))]

        # reconstitute the prompt and generation
        prompt = tokenizer.decode(prompt_token_ids, skip_special_tokens=True)
        generation = tokenizer.decode(generation_token_ids, skip_special_tokens=True) + ' ' + tokenizer.eos_token

        batch_element = { 'prompt_text' : prompt, f'{prefix}_text': generation }

        for k,v in tokenizer(prompt).items():
            batch_element[f'prompt_{k}'] = v

        for k,v in tokenizer(generation).items():
            batch_element[f'{prefix}_{k}'] = v

        # combine the prompt and generation belonging to the same example
        batch_element.update(self.combine_prompt_and_generation(batch_element, batch_element, tokenizer=tokenizer, prefix=prefix))
        return batch_element

    def __getitem__(self, index):
        element = self.dataset[index]
        batch_element = self.tokenize_batch_element(
            element['prompt'],
            element['completion'],
            truncation_mode='keep_start' if '\n\nHuman:' not in element['prompt'] else 'keep_end',
            tokenizer=self.tokenizer
        )
        batch_element.update(self.tokenize_batch_element(
            element['prompt'],
            element['completion'],
            truncation_mode='keep_start' if '\n\nHuman:' not in element['prompt'] else 'keep_end',
            tokenizer=self.weak_tokenizer,
            prefix="weak_target"
        )
        )
        return batch_element

    def combine_prompt_and_generation(self, prompt_dict: Dict, generation_dict: Dict, tokenizer, prefix: str='target') -> Dict:
        combined_dict = { f'{prefix}_combined_text' : prompt_dict['prompt_text'] + generation_dict[f'{prefix}_text'] }
        for k,v in tokenizer(prompt_dict['prompt_text'] + generation_dict[f'{prefix}_text']).items():
            combined_dict[f'{prefix}_combined_{k}'] = v

        combined_dict[f'{prefix}_labels'] = combined_dict[f'{prefix}_combined_input_ids'][:]  # contains both input and response (unpadded)
        combined_dict[f'{prefix}_labels'][:len(prompt_dict['prompt_input_ids'])] = [-100] * len(prompt_dict['prompt_input_ids'])

        return combined_dict
    
    def padding_collate_fn(self, batch):
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:
                    # flip prompt so that you are padding to the beginning
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                if k.endswith('_input_ids'):
                    if k.startswith("weak"):
                        padding_value = self.weak_tokenizer.pad_token_id
                    else:
                        padding_value = self.tokenizer.pad_token_id

                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                # Always pad to max_length for consistency across processes
                max_len = self.max_prompt_length if 'prompt' in k else self.max_length
                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        if 'target_combined_attention_mask' in padded_batch:
            position_ids = torch.cumsum(padded_batch['target_combined_attention_mask'], dim=-1) - 1
            position_ids.masked_fill_(~padded_batch['target_combined_attention_mask'].bool(), 0)
            padded_batch['target_combined_position_ids'] = position_ids
        else:
            chosen_position_ids = torch.cumsum(padded_batch['chosen_combined_attention_mask'], dim=-1) - 1
            chosen_position_ids.masked_fill_(~padded_batch['chosen_combined_attention_mask'].bool(), 0)
            padded_batch['chosen_combined_position_ids'] = chosen_position_ids

            rejected_position_ids = torch.cumsum(padded_batch['rejected_combined_attention_mask'], dim=-1) - 1
            rejected_position_ids.masked_fill_(~padded_batch['rejected_combined_attention_mask'].bool(), 0)
            padded_batch['rejected_combined_position_ids'] = rejected_position_ids
        return padded_batch

def get_human_prompt():
    return "\n\nHuman: "

def get_assistant_prompt():
    return "\n\nAssistant: "

def get_separate_prompt(i: int):
    return get_human_prompt() if i % 2 == 0 else get_assistant_prompt()

def get_tldr_sft(split):
    tldr_ds = load_dataset('vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144', split=split)
    tldr_ds = tldr_ds.select(range(len(tldr_ds)-8000))
    tldr_ds = tldr_ds.select_columns(['query', 'reference_response'])
    def remove_eos_token(ex):
        ex['reference_response'] = ex['reference_response'].replace("<|endoftext|>", "")
        return ex

    tldr_ds = tldr_ds.map(remove_eos_token, batched=False, num_proc=8)
    tldr_ds = tldr_ds.rename_column('query', 'prompt')
    tldr_ds = tldr_ds.rename_column('reference_response', 'completion')
    return tldr_ds

def preprocess_dialogue(ex):
    chosen = ex['chosen']
    context = ex['chosen'][:-1]
    chosen = ex['chosen'][-1]
    rejected = ex['rejected'][-1]

    context = [get_separate_prompt(i + (len(context) + 1) % 2) + s for i, s in enumerate(context)] 
    context = ''.join(context)
    context += get_assistant_prompt().rstrip()

    chosen = " " + chosen.strip()
    rejected = " " + rejected.strip()

    example = {
        "prompt": context,
        "chosen": chosen,
        "rejected": rejected
    }
    return example


def get_full_sft(split):
    tldr_ds = get_tldr_sft(split)
    anthropic = get_hh(split=split)
    anthropic = anthropic.rename_column('chosen', 'completion')
    anthropic = anthropic.remove_columns(['rejected'])
    ds = concatenate_datasets([tldr_ds, anthropic])
    ds = ds.shuffle(seed=42)
    return ds

def get_full_pref(split):
    tldr_ds = get_tldr_pref(split)
    anthropic = get_hh(split)
    ds = concatenate_datasets([tldr_ds, anthropic])
    ds = ds.shuffle(seed=42)
    return ds

def get_tldr_pref(split):
    tldr_ds = load_dataset('vwxyzjn/summarize_from_feedback_oai_preprocessing_1706381144', split='validation' if split == 'test' else split)
    tldr_ds = tldr_ds.select_columns(['query', 'chosen', 'rejected'])
    tldr_ds = tldr_ds.rename_column('query', 'prompt')
    def remove_eos_token(ex):
        ex['chosen'] = ex['chosen'].replace("<|endoftext|>", "")
        ex['rejected'] = ex['rejected'].replace("<|endoftext|>", "")
        return ex
    tldr_ds = tldr_ds.map(remove_eos_token, batched=False, num_proc=8)
    return tldr_ds

def get_hh(split):
    ds = load_dataset("Dahoas/full-hh-rlhf", split=split)
    ds = ds.select_columns(['prompt', 'chosen', 'rejected'])
    ds = ds.filter(lambda ex: ex['chosen'] != ex['rejected'])
    return ds

def get_full_hh_sft(split: str = "train"):
    dataset = load_dataset("Dahoas/full-hh-rlhf", split=split)
    dataset = dataset.rename_column("response", "completion")
    dataset = dataset.shuffle()
    return dataset

def get_tldr_llama3_8b(split):
    dataset = load_dataset('DatPySci/Llama-3.1-8B-rm-tldr-pref', split=split)
    return dataset

def get_alpaca_pref(split: str):
    dataset = load_dataset("tatsu-lab/alpaca_farm", "alpaca_human_preference", split="preference")
    def format_example(ex):
        if ex['input'] != "":
            ex['prompt'] = ex['instruction'] + "\n" + ex["input"]
        else:
            ex['prompt'] = ex["instruction"]
        
        ex["chosen"] = ex["output_1"] if ex['preference'] == 1 else ex["output_2"]
        ex["rejected"] = ex["output_1"] if ex['preference'] == 2 else ex["output_2"]
        return ex
    
    dataset = dataset.map(format_example)
    return dataset

def get_ultrachat_w2s(split: str):
    if split == "train":
        split = "train_sft[:25%]"
        
    elif split == "test":
        split = "test_sft"
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    def get_response(ex):
        ex["completion"] = ex["messages"][1]["content"]
        return ex
    dataset = dataset.map(get_response, num_proc=8)
    dataset = dataset.shuffle()
    return dataset

def get_ultrachat_sft(split: str):
    if split == "train":
        split = "train_sft"
    
    elif split == "test":
        split = "test_sft"
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    def get_response(ex):
        ex["completion"] = ex["messages"][1]["content"]
        return ex
    
    dataset = dataset.map(get_response, num_proc=8)
    dataset = dataset.shuffle()
    return dataset

def get_ultrachat_pref(split: str):
    if split == "train":
        split = "train_prefs"
        
    elif split == "test":
        split = "test_prefs"
    
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    def get_response(ex):
        ex["completion"] = ex["chosen"][1]["content"]
        ex["chosen"] = ex["chosen"][1]["content"]
        ex["rejected"] = ex["rejected"][1]["content"]
        return ex
    
    dataset = dataset.map(get_response, num_proc=8)
    dataset = dataset.shuffle()
    return dataset



def get_alpaca_eval(split: str):
    dataset = load_dataset('tatsu-lab/alpaca_eval', "alpaca_eval_gpt4_baseline", split="eval")
    def format_example(ex):
        ex['prompt'] = ex['instruction'] 
        ex["completion"] = ex["output"]
        return ex
    
    dataset = dataset.map(format_example)
    return dataset


def get_alpaca_train(split: str):
    dataset = load_dataset("tatsu-lab/alpaca_farm", "alpaca_instructions", split="sft")
    def format_example(ex):
        if ex['input'] != "":
            ex['prompt'] = ex['instruction'] + "\n" + ex["input"]
        else:
            ex['prompt'] = ex["instruction"]
        
        ex["completion"] = ex["output"]
        return ex
    
    dataset = dataset.map(format_example)
    return dataset

def flatten_data(data):
    results = []
    for sample in data:
        prompt = sample["prompt"]
        for completion in sample["completion"]:
            results.append({"prompt": prompt, "completion": completion})
    return results

def get_alpaca_qwen1_5b_pref(split: str):
    all_data = []
    with open("samples/alpaca_qwen7b_Qwen2.5-1.5B-Instruct_train.jsonl", "r") as f:
        for line in f:
            all_data.append(json.loads(line))
    all_data = hf_dataset.from_list(all_data)
    all_data = all_data.shuffle()
    return all_data

def get_alpaca_qwen3b_pref(split: str):
    all_data = []
    with open("samples/alpaca_qwen7b_Qwen2.5-3B-Instruct_train.jsonl", "r") as f:
        for line in f:
            all_data.append(json.loads(line))
    all_data = hf_dataset.from_list(all_data)
    all_data = all_data.shuffle()
    return all_data

def get_gpt2_large_tldr_refine(split: str, flatten: bool = True):
    all_data = []
    with open(f"samples/Llama-3.2-3B_train_tldr_sft_gpt2-large_refine.jsonl", "r") as f:
        for line in f:
            all_data.append(json.loads(line))
        
    if flatten:
        all_data = flatten_data(all_data)
    all_data = hf_dataset.from_list(all_data)
    def preprocess(ex):
        icl_prompt = ex['prompt']
        prompt_idx = icl_prompt.rfind("SUBREDDIT")
        prompt = icl_prompt[prompt_idx:]
        ex['prompt'] = prompt
        idx = ex['completion'].rfind("SUBREDDIT")
        if idx != -1:
            ex['completion'] = ex['completion'][:idx]
        return ex
    all_data = all_data.map(preprocess, num_proc=8, batched=False)
    all_data = all_data.shuffle()
    return all_data
    
def get_llama3b_tldr_refine(split: str, flatten: bool = True):
    all_data = []
    with open(f"samples/Llama-3.2-3B_train_tldr_sft.jsonl", "r") as f:
        for line in f:
            all_data.append(json.loads(line))
        
    if flatten:
        all_data = flatten_data(all_data)
    all_data = hf_dataset.from_list(all_data)
    def preprocess(ex):
        icl_prompt = ex['prompt']
        prompt_idx = icl_prompt.rfind("SUBREDDIT")
        prompt = icl_prompt[prompt_idx:]
        ex['prompt'] = prompt
        idx = ex['completion'].rfind("SUBREDDIT")
        if idx != -1:
            ex['completion'] = ex['completion'][:idx]
        return ex
    all_data = all_data.map(preprocess, num_proc=8, batched=False)
    all_data = all_data.shuffle()
    return all_data

def get_gpt2_medium_tldr_refine(split: str, flatten: bool = True):
    all_data = []
    with open(f"samples/Llama-3.2-3B_train_tldr_sft_gpt2-medium_refine.jsonl", "r") as f:
        for line in f:
            all_data.append(json.loads(line))
        
    if flatten:
        all_data = flatten_data(all_data)
    print(len(all_data))
    all_data = hf_dataset.from_list(all_data)
    def preprocess(ex):
        icl_prompt = ex['prompt']
        prompt_idx = icl_prompt.rfind("SUBREDDIT")
        prompt = icl_prompt[prompt_idx:]
        ex['prompt'] = prompt
        idx = ex['completion'].rfind("SUBREDDIT")
        if idx != -1:
            ex['completion'] = ex['completion'][:idx]
        return ex
    all_data = all_data.map(preprocess, num_proc=8, batched=False)
    all_data = all_data.shuffle()
    return all_data

def get_gpt2_dpo_tldr(split: str, flatten: bool = True):
    all_data = []
    with open(f"samples/gpt2_dpo_beta_0.1_tldr_pref_train_tldr_sft.jsonl", "r") as f:
        for line in f:
            all_data.append(json.loads(line))
        
    if flatten:
        all_data = flatten_data(all_data)
    all_data = hf_dataset.from_list(all_data)
    all_data = all_data.shuffle()
    return all_data

def get_gpt2_tldr_refine(split: str, flatten: bool = True):
    all_data = []
    i = 0 
    with open(f"samples/Llama-3.2-3B_train_tldr_sft_gpt2_refine.jsonl", "r") as f:
        for line in f:
            sample = json.loads(line)
            all_data.append(sample)
        
    if flatten:
        all_data = flatten_data(all_data)
    all_data = hf_dataset.from_list(all_data)
    def preprocess(ex):
        icl_prompt = ex['prompt']
        prompt_idx = icl_prompt.rfind("SUBREDDIT")
        prompt = icl_prompt[prompt_idx:]
        ex['prompt'] = prompt
        idx = ex['completion'].rfind("SUBREDDIT")
        if idx != -1:
            ex['completion'] = ex['completion'][:idx]
        return ex
    all_data = all_data.map(preprocess, num_proc=8, batched=False)
    all_data = all_data.shuffle()
    return all_data

def get_alpaca_qwen1_5b_refine(split: str, flatten: bool = False):
    all_data = []
    with open(f"samples/refine/Qwen2.5-7B-Instruct_train_alpaca_qwen1_5b_refine.jsonl", "r") as f:
        for line in f:
            all_data.append(json.loads(line))
        
    if flatten:
        all_data = flatten_data(all_data)
    print(len(all_data))
    all_data = hf_dataset.from_list(all_data)
    all_data = all_data.shuffle()
    return all_data


def get_ultrachat_llama3_8b(split: str, flatten: bool = False):
    all_data = []
    with open(f"samples/refine/Llama-3.1-8B-Instruct_train_ultrachat_w2s_refine.jsonl", "r") as f:
        for line in f:
            all_data.append(json.loads(line))
        
    if flatten:
        all_data = flatten_data(all_data)
    print(len(all_data))
    all_data = hf_dataset.from_list(all_data)
    all_data = all_data.shuffle()
    return all_data


def get_ultrachat_qwen7b_pref(split: str):
    all_data = []
    with open(f"samples/label/ultrachat_qwen7b_Qwen2.5-1.5B-Instruct_train.jsonl", "r") as f:
        for line in f:
            all_data.append(json.loads(line))
    print(len(all_data))
    all_data = hf_dataset.from_list(all_data)
    all_data = all_data.shuffle()
    return all_data

def get_ultrachat_qwen1_5b(split: str, flatten: bool = True):
    all_data = []
    with open(f"samples/refine/Qwen2.5-1.5B-Instruct_train_ultrachat_w2s_refine.jsonl", "r") as f:
        for line in f:
            all_data.append(json.loads(line))
        
    if flatten:
        all_data = flatten_data(all_data)
    print(len(all_data))
    all_data = hf_dataset.from_list(all_data)
    all_data = all_data.shuffle()
    return all_data

def get_ultrachat_qwen7b(split: str, flatten: bool = True):
    all_data = []
    with open(f"samples/refine/Qwen2.5-7B-Instruct_train_ultrachat_w2s_refine.jsonl", "r") as f:
        for line in f:
            all_data.append(json.loads(line))
        
    if flatten:
        all_data = flatten_data(all_data)
    print(len(all_data))
    all_data = hf_dataset.from_list(all_data)
    all_data = all_data.shuffle()
    return all_data

def get_alpaca_qwen1_5b(split: str, flatten: bool = True):
    all_data = []
    with open(f"samples/Qwen2.5-1.5B-Instruct_{split}_alpaca_train.jsonl", "r") as f:
        for line in f:
            all_data.append(json.loads(line))
        
    if flatten:
        all_data = flatten_data(all_data)
    all_data = hf_dataset.from_list(all_data)
    all_data = all_data.shuffle()
    return all_data

def get_alpaca_qwen3b(split: str, flatten: bool = True):
    all_data = []
    with open(f"samples/Qwen2.5-3B-Instruct_{split}_alpaca_train.jsonl", "r") as f:
        for line in f:
            all_data.append(json.loads(line))
        
    if flatten:
        all_data = flatten_data(all_data)
    all_data = hf_dataset.from_list(all_data)
    all_data = all_data.shuffle()
    return all_data

def get_alpaca_qwen7b(split: str, flatten: bool = True):
    all_data = []
    with open(f"samples/Qwen2.5-7B-Instruct_{split}_alpaca_train.jsonl", "r") as f:
        for line in f:
            all_data.append(json.loads(line))
        
    if flatten:
        all_data = flatten_data(all_data)
    all_data = hf_dataset.from_list(all_data)
    all_data = all_data.shuffle()
    return all_data


def get_sampled_pair(sample_path: str = "samples/online_ipo_samples.parquet", split: str = "train"):
    if split == "test":
        return get_tldr_llama3_8b(split)
    else:
        dataset = load_dataset("parquet", data_files="samples/online_ipo_samples.parquet", split='train')
        return dataset

def get_tldr_pref_shift(split: str = "train"):
    if split == "test":
        return get_tldr_llama3_8b(split)
    else:
        all_data = []
        with open("samples/sampled_pair_Llama-3.1-8B-rm_train.jsonl", 'r') as f:
            for line in f:
                all_data.append(json.loads(line))
        return all_data

if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch.nn.functional as F
    ds = get_full_sft('train')
    test_ds = get_full_sft('test')
    print(ds)
    # dataset = load_dataset('trl-lib/tldr', split='train')
    tokenizer = autotokenizer.from_pretrained('qwen/qwen2.5-7b')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = TorchDataset(ds, tokenizer)
    
    # packed_dataloader = DataLoader(dataset, batch_size=4, collate_fn=dataset.packing_collate_fn, shuffle=False)
    padded_dataloader = DataLoader(dataset, batch_size=4, collate_fn=dataset.padding_collate_fn, shuffle=True)

    for padded_batch in padded_dataloader:
        print(padded_batch)
        break