#!/bin/bash
#SBATCH -c 2 # request two cores 
#SBATCH -p preempt
#SBATCH -o log/experiment-rlhf-reward.out
#SBATCH -e log/error_experiment-rlhf-reward.out
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=bash
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:2

nvidia-smi

# MODEL_PATH=Llama-3.2-3B_tldr_sft
MODEL_PATHS=("gpt2" "gpt2-medium" "gpt2-large")
GRAD_ACCUMULATION_STEPS=1
DATASET_NAME=hh_pref
BATCH_SIZE=32
EVAL_BATCH_SIZE=4

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
OUTPUT_DIR=models/reward/${MODEL_PATH}_reward_hh
SFT_MODEL_PATH=models/sft/${MODEL_PATH}_hh_sft/step-16000
accelerate launch --config_file accelerate_configs/deepspeed_zero3_4gpu.yaml \
    --main_process_port=6500\
    cli/train_reward.py\
    --model_path=$SFT_MODEL_PATH \
    --batch_size=$BATCH_SIZE \
    --eval_every=51200 \
    --exp_name=${MODEL_PATH}_reward_hh \
    --eval_batch_size=$EVAL_BATCH_SIZE \
    --dataset_name=$DATASET_NAME \
    --num_train_epochs=1\
    --no-use-packing \
    --gradient_accumulation_steps=$GRAD_ACCUMULATION_STEPS \
    --no-debug \
    --warm-up-steps=10 \
    --lr=1e-6\
    --output_dir=$OUTPUT_DIR
done