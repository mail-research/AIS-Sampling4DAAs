from vllm import LLM, SamplingParams
import jsonlines
from accelerate import Accelerator
import sys
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import torch
sys.path.append('/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/benchmark/RLHF-training')
from dataset.reward_dataset import *
from config import EvaluateConfig
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


def to_batch(x, batch_size):
    for i in range(0, len(x), batch_size):
        yield x[i : i + batch_size]

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
    config = tyro.cli(EvaluateConfig)

    world_size = torch.cuda.device_count()
    policy = LLM(config.model_path, tensor_parallel_size=world_size, enable_prefix_caching=True, max_model_len=4096, enforce_eager=False, gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(n=4, temperature=0.8, max_tokens=1024)
    tokenizer = policy.get_tokenizer()

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    sft_dataset = globals()[f'get_{config.dataset_name}'](config.split)
    # sft_dataset = TorchDataset(sft_dataset, tokenizer)
    path_name = config.model_path.split("/")[-1]
    print(path_name)
    # dataloader = DataLoader(sft_dataset, batch_size=config.batch_size, collate_fn=eval_dataset.padding_collate_fn, pin_memory=True, num_workers=8, shuffle=False)
    total_items = 0
    i = 0

    with jsonlines.open(f"samples/refine/{path_name}_{config.split}_{config.dataset_name}_refine.jsonl", "w") as writer:
        for batch in tqdm(to_batch(sft_dataset, config.batch_size), total=len(sft_dataset) // config.batch_size):
            prompts = batch['prompt']
            conv = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]
            outputs = policy.generate(conv, sampling_params=sampling_params)
            outputs = [[output.outputs[idx].text.rstrip() for idx in range(len(output.outputs))] for output in outputs]
            if i == 0:
                print(outputs[0])
                print("---" * 80)
                print(outputs[5])
                i += 1

            for prompt, completion in safe_zip(conv, outputs):
                item = {
                    "prompt": prompt,
                    "completion": completion
                }
                writer.write(item)
        
        writer.close()
    print(f"Total items: {total_items}")