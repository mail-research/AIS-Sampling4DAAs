import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
import torch
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--repo_id', type=str)
    parser.add_argument('--generative', action='store_false')
    args = parser.parse_args()

    policy_kwargs = {
        "torch_dtype": torch.bfloat16,
    }
    model_class = AutoModelForCausalLM if args.generative else AutoModelForSequenceClassification
    print(model_class)
    model = model_class.from_pretrained(args.model_path, **policy_kwargs)
    print(model.model.embed_tokens.weight.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model.resize_token_embeddings(len(tokenizer))

    model.push_to_hub(args.repo_id, private=True)
    tokenizer.push_to_hub(args.repo_id, private=True)

