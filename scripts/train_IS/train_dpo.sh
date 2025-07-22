#!/bin/bash
#SBATCH -c 8 # request two cores 
#SBATCH -p kisski-h100
#SBATCH -o log/experiment-rlhf-dpo-shift.out
#SBATCH -e log/error_experiment-rlhf-dpo-shift.out
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=rlhf
#SBATCH --ntasks-per-node=1
#SBATCH -G H100:4

nvidia-smi
source ~/.shadow1
conda activate align
# MODEL_PATHS=('Llama-3.2-1B')
GRAD_ACCUMULATION_STEPS=4
BATCH_SIZE=32
EVAL_BATCH_SIZE=8
SEEDS=('1')
BETAS=('0.05')
# DATASET_NAME=${MODEL_SIZES}_llama3_3b_pref
DATASET_NAME=hh
SFT_MODEL_PATH=DatPySci/Llama-3.2-3B-sft-mixture

# for MODEL_PATH in "${MODEL_PATHS[@]}"; do
for SEED in "${SEEDS[@]}"; do
for BETA in "${BETAS[@]}"; do
OUTPUT_DIR=models/dpo/${MODEL_PATH}_dpo_beta_${BETA}_hh

accelerate launch --config_file accelerate_configs/deepspeed_zero3_4gpu.yaml \
    --main_process_port=4500\
    cli/train_dpo.py\
    --model_path=$SFT_MODEL_PATH \
    --batch_size=$BATCH_SIZE \
    --seed=$SEED \
    --eval_every=1920000 \
    --max_prompt_length=512\
    --max_length=640 \
    --beta=${BETA} \
    --project_name=weak-to-strong-chat\
    --exp_name=${MODEL_PATH}-dpo-beta-${BETA}-ultrachat_pref_w2s\
    --eval_batch_size=$EVAL_BATCH_SIZE \
    --dataset_name=$DATASET_NAME \
    --no-use-packing \
    --use-liger \
    --num_train_epochs=1\
    --gradient_accumulation_steps=$GRAD_ACCUMULATION_STEPS \
    --no-debug \
    --warm-up-steps=150 \
    --lr=1e-6\
    --output_dir=$OUTPUT_DIR 
done
done
# done