#!/bin/bash
#SBATCH -c 8 # request two cores 
#SBATCH -p kisski
#SBATCH -o log/experiment-rlhf-online-dpo2.out
#SBATCH -e log/error_experiment-rlhf-online-dpo2.out
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=online-dpo3
#SBATCH --ntasks-per-node=1
#SBATCH -G A100:4

nvidia-smi
source ~/.shadow1
conda activate align
# MODEL_PATHS=('Llama-3.2-1B')
GRAD_ACCUMULATION_STEPS=4
BATCH_SIZE=16
EVAL_BATCH_SIZE=8
SEEDS=('1')
# BETAS=('0.05' '0.1' '0.01')
BETAS=("0.1")
# DATASET_NAME=${MODEL_SIZES}_llama3_3b_pref
# DATASET_NAME=ultrachat_qwen7b_pref
DATASET_NAME=tldr_sft
SFT_MODEL_PATH=DatPySci/Llama-3.2-3B-sft-mixture
RM_MODEL_PATH=models/reward/Llama-3.2-3B_tldr_sft_reward/step-92858
MODEL_PATH="Llama-3.2-3B-sft"

# for MODEL_PATH in "${MODEL_PATHS[@]}"; do
for SEED in "${SEEDS[@]}"; do
for BETA in "${BETAS[@]}"; do
OUTPUT_DIR=models/dpo/${MODEL_PATH}_online_dpo_beta_${BETA}_tldr_sft

accelerate launch --config_file accelerate_configs/deepspeed_zero3_4gpu.yaml \
    --main_process_port=4500\
    cli/train_online_dpo.py\
    --model_path=$SFT_MODEL_PATH \
    --reward_model_path=$RM_MODEL_PATH \
    --batch_size=$BATCH_SIZE \
    --seed=$SEED \
    --eval_every=192_00 \
    --max_prompt_length=512\
    --max_length=640 \
    --beta=${BETA} \
    --project_name=IS-DPo\
    --exp_name=Llama-3B-online-dpo-beta-${BETA}-tldr_sft\
    --eval_batch_size=$EVAL_BATCH_SIZE \
    --dataset_name=$DATASET_NAME \
    --no-use-packing \
    --use-liger \
    --num_train_epochs=2\
    --gradient_accumulation_steps=$GRAD_ACCUMULATION_STEPS \
    --no-debug \
    --warm-up-steps=150 \
    --lr=1e-6\
    --output_dir=$OUTPUT_DIR 
done
done
# done