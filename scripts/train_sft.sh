#!/bin/bash
#SBATCH -c 16 # request two cores 
#SBATCH -p kisski-h100
#SBATCH -o log/rlhf-sft.out
#SBATCH -e log/error-rlhf-sft.out
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=sft_train
#SBATCH --ntasks-per-node=1
#SBATCH -G H100:4

# source ~/.shadow1
# conda activate align
nvidia-smi
GRAD_ACCUMULATION_STEPS=8
BATCH_SIZE=16
EVAL_BATCH_SIZE=8
MODEL_PATHS=("meta-llama/Llama-3.2-1B")
DATASET_NAME=ultrachat_sft

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
OUTPUT_DIR=models/sft/${MODEL_PATH}_sft_ultrachat

accelerate launch --config_file accelerate_configs/deepspeed_zero3_4gpu.yaml \
    --main_process_port=3500\
    cli/train_sft.py\
    --model_path=${MODEL_PATH}\
    --max_length=2048\
    --max_prompt_length=1024\
    --batch_size=$BATCH_SIZE \
    --eval_every=200000\
    --exp_name=${MODEL_PATH}-ultrachat_sft\
    --project_name=weak-to-strong-chat\
    --eval_batch_size=$EVAL_BATCH_SIZE \
    --num_train_epochs=1 \
    --dataset_name=$DATASET_NAME \
    --use-liger \
    --gradient_accumulation_steps=$GRAD_ACCUMULATION_STEPS \
    --no-debug \
    --no-use-packing\
    --warm-up-steps=150 \
    --lr=1e-5\
    --average_log_prob\
    --output_dir=$OUTPUT_DIR 
done