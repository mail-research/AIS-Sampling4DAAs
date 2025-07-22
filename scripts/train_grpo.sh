#!/bin/bash
#SBATCH -c 16 # request two cores 
#SBATCH -p kisski-h100
#SBATCH -o log/exp-grpo-Qwen1.5B.out
#SBATCH -e log/error_exp-grpo-Qwen1.5B.out
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=grpo-Qwen1.5B
#SBATCH --ntasks-per-node=1
#SBATCH -G H100:4
nvidia-smi
MODEL_SIZES=('3B')

GRAD_ACCUMULATION_STEPS=4
DATASET_NAME=tldr_sft
BATCH_SIZE=16
EVAL_BATCH_SIZE=4
SEEDS=('42')
BETAS=("0.01")

SFT_MODEL_PATH=DatPySci/Llama-3.2-3B-sft-mixture
REWARD_MODEL_PATH=models/reward/Llama-3.2-3B_tldr_sft_reward/step-92858

for MODEL_SIZE in "${MODEL_SIZES[@]}"; do
for SEED in "${SEEDS[@]}"; do
for BETA in "${BETAS[@]}"; do
OUTPUT_DIR=models/grpo/Llama-3.2-${MODEL_SIZE}-grpo_beta_${BETA}_hh_seed_${SEED}

accelerate launch --config_file accelerate_configs/deepspeed_zero2.yaml \
    --main_process_port=3500\
    cli/train_grpo.py\
    --model_path=$SFT_MODEL_PATH \
    --reward_model_path=$REWARD_MODEL_PATH\
    --batch_size=$BATCH_SIZE \
    --seed=$SEED \
    --eval_every=12480 \
    --no-debug\
    --beta=${BETA} \
    --exp_name=Llama-3.2-${MODEL_SIZE}_grpo_beta_${BETA}_tldr_sft_seed_${SEED} \
    --eval_batch_size=$EVAL_BATCH_SIZE \
    --num_iters=1000\
    --dataset_name=$DATASET_NAME \
    --no-use-packing \
    --num_train_epochs=1\
    --gradient_accumulation_steps=$GRAD_ACCUMULATION_STEPS \
    --warm-up-steps=100 \
    --lr=1e-6\
    --output_dir=$OUTPUT_DIR 
done
done
done