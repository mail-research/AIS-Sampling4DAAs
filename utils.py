import torch
import numpy as np
import gc
import os
from typing import Optional
import json
from typing import Dict
from typing import Dict, Union, Type, List, TextIO
from transformers import StoppingCriteria


class StreamingJSONWriter:
    """Writes JSON arrays to a file in a streaming fashion."""
    def __init__(self, file: TextIO):
        self.file = file
        self.is_first = True
        self.file.write('[\n')
    
    def write_item(self, item: Dict):
        """Write a single item to the JSON array."""
        if not self.is_first:
            self.file.write(',\n')
        json.dump(item, self.file, indent=2)
        self.is_first = False
        # Flush after each write to ensure immediate disk writing
        self.file.flush()
    
    def close(self):
        """Close the JSON array and the file."""
        self.file.write('\n]')
        self.file.flush()


def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}


def delete_dict(d: Dict):
    """Delete all items inside the dict."""
    for k in list(d.keys()):
        del d[k]


def delete_dicts(*dicts: Dict):
    """Delete all items inside the given dictionaries."""
    for d in dicts:
        for k in list(d.keys()):
            del d[k]
        
def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


@torch.no_grad()
def get_batch_samples(batch, policy, tokenizer, return_tensors=False, **generation_kwargs):

    policy_output = policy.generate(
        batch['prompt_input_ids'],
        attention_mask=batch['prompt_attention_mask'],
        **generation_kwargs
    )
    if return_tensors:
        return policy_output
    else:
        policy_output_decoded = tokenizer.batch_decode(policy_output, skip_special_tokens=True)
        return policy_output_decoded

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)

def save(policy, example_counter, accelerator, tokenizer, config, metrics: Optional[Dict] = {}):
    output_dir = os.path.join(config.output_dir, f'step-{example_counter}')
    
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        accelerator.print(f"Saving tokenizer...")
        tokenizer.save_pretrained(output_dir)

        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            metrics['counter'] = example_counter
            json.dump(metrics, f)
    
    accelerator.wait_for_everyone()
    accelerator.print(f"Saving model...")
    
    state_dict = accelerator.get_state_dict(policy)
    unwrapped_model = accelerator.unwrap_model(policy)
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
        safe_serialization=False
    )
    accelerator.wait_for_everyone()
    

def free_memory():
    torch.cuda.empty_cache()
    gc.collect()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer

def concatneted_inputs(padded_batch, tokenizer):
    max_length = max(padded_batch['chosen_combined_input_ids'].shape[1], padded_batch['rejected_combined_input_ids'].shape[1])
    concatenated_batch = {}

    for k in padded_batch:
        if k.startswith('chosen') and isinstance(padded_batch[k], torch.Tensor):
            if 'labels' in k:
                pad_value = -100 
            elif 'input_ids' in k:
                pad_value = tokenizer.pad_token_id
            elif 'attention_mask' in k:
                pad_value = 0

            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(padded_batch[k], max_length, pad_value=pad_value)

    for k in padded_batch:
        if k.startswith('rejected') and isinstance(padded_batch[k], torch.Tensor):
            if 'labels' in k:
                pad_value = -100 
            elif 'input_ids' in k:
                pad_value = tokenizer.pad_token_id
            elif 'attention_mask' in k:
                pad_value = 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(padded_batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch