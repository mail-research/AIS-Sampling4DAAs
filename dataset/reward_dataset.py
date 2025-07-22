import sys
sys.path.append('/cluster/tufts/laolab/kdoan02/RLHF-training')
from dataset.data_utils import *
from utils import pad_to_length

class RewardDataset(TorchDataset):
    def __getitem__(self, index):
        element = self.dataset[index]
        batch_element = {}
        for prefix in ['chosen', 'rejected']:
            batch_element.update(self.tokenize_batch_element(
                element['prompt'],
                element[prefix],
                truncation_mode='keep_end' if '\n\nHuman:' in element['prompt'] else 'keep_start',
                prefix=prefix
            ))
            
        if "chosen_logps" in element.keys() or "rejected_logps" in element.keys():
            for prefix in ['chosen', 'rejected']:
                batch_element[f'{prefix}_logps'] = element[f'{prefix}_logps']
        if "ref_chosen_logps" in element.keys() or "ref_rejected_logps" in element.keys():
            for prefix in ['chosen', 'rejected']:
                batch_element[f'ref_{prefix}_logps'] = element[f'ref_{prefix}_logps']
            
        return batch_element
    

    def packing_collate_fn(self, batch):
        packed_batch = {}
        packed_batch['chosen_seq_len'] = []
        packed_batch['rejected_seq_len'] = []
        packed_batch['chosen_combined_position_ids'] = []
        packed_batch['rejected_combined_position_ids'] = []
        
        index = 1
        for example in batch:
            for k, v in example.items():
                if (k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels')):
                    if k.endswith('_attention_mask') and k.startswith('prompt') == False:
                        if k not in packed_batch:
                            packed_batch[k] = []
                        if k .startswith('chosen'):
                            value = torch.full_like(torch.tensor(example['chosen_combined_input_ids'][:max_len]), index)
                            packed_batch[k].append(value.flatten())
                        else:
                            value = torch.full_like(torch.tensor(example['chosen_combined_input_ids'][:max_len]), index + len(batch))
                            packed_batch[k].append(value.flatten())
                    else:
                        if k not in packed_batch:
                            packed_batch[k] = []

                        max_len = self.max_prompt_length if 'prompt' in k else self.max_length
                        v = v[:max_len]
                        packed_batch[k].append(torch.tensor(v).flatten())

                    if k.endswith('combined_input_ids') and k.startswith('prompt') == False:
                        if k.startswith('chosen'):
                            packed_batch['chosen_seq_len'].append(len(v))
                        elif k.startswith('rejected'):
                            packed_batch['rejected_seq_len'].append(len(v))
                    
            index += 1
            
            for prefix in ['chosen', 'rejected']:
                packed_batch[f'{prefix}_combined_position_ids'].append(torch.tensor(list(range(len(example[f'{prefix}_combined_input_ids'][:max_len])))))
        
        for postfix in ['input_ids', 'attention_mask', 'labels', 'position_ids']:
            if postfix == 'labels':
                packed_batch[f'concatenated_{postfix}'] = packed_batch[f'chosen_{postfix}'] + packed_batch[f'rejected_{postfix}']
            else:
                packed_batch[f'concatenated_{postfix}'] = packed_batch[f'chosen_combined_{postfix}'] + packed_batch[f'rejected_combined_{postfix}']
        
        packed_batch['seq_len'] = packed_batch['chosen_seq_len'] + packed_batch['rejected_seq_len']

        for k, v in packed_batch.items():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels') or k.endswith('_position_ids') or k.endswith('seq_len'):
                if k.endswith('seq_len'):
                    packed_batch[k] = torch.tensor(packed_batch[k])
                else:
                    packed_batch[k] = torch.cat(packed_batch[k], dim=0).unsqueeze(0)
        return packed_batch
    
if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from data_utils import get_full_pref

    # dataset = load_dataset('trl-lib/`tldr-preference', split='train')
    dataset = get_full_pref('train')
    tokenizer = AutoTokenizer.from_pretrained('qwen/qwen2.5-7b')
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    def concatneted_inputs(padded_batch):
        max_length = max(padded_batch['chosen_combined_input_ids'].shape[1], padded_batch['rejected_combined_input_ids'].shape[1])
        concatenated_batch = {}

        for k in padded_batch:
            if k.startswith('chosen') and isinstance(padded_batch[k], torch.Tensor):
                pad_value = -100 if 'labels' in k else 0
                concatenated_key = k.replace('chosen', 'concatenated')
                concatenated_batch[concatenated_key] = pad_to_length(padded_batch[k], max_length, pad_value=pad_value)

        for k in padded_batch:
            if k.startswith('rejected') and isinstance(padded_batch[k], torch.Tensor):
                pad_value = -100 if 'labels' in k else 0
                concatenated_key = k.replace('rejected', 'concatenated')
                concatenated_batch[concatenated_key] = torch.cat((
                    concatenated_batch[concatenated_key],
                    pad_to_length(padded_batch[k], max_length, pad_value=pad_value),
                ), dim=0)
        return concatenated_batch

    dataset = RewardDataset(dataset, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=dataset.padding_collate_fn)
    all_seqs = []
    index = 0
    for batch in dataloader:
        check = concatneted_inputs(batch)
        breakpoint( )