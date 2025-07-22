#!/bin/bash
#SBATCH -c 2 # request two cores 
#SBATCH -p preempt
#SBATCH -o log/experiment-rlhf-is-dpo.out
#SBATCH -e log/error_experiment-rlhf-is-dpo.out
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=is-dpo-bash
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:2

nvidia-smi
MODEL_SIZE=3B

SFT_MODEL_PATH=DatPySci/Llama-3.2-3B-sft-mixture
GRAD_ACCUMULATION_STEPS=8
DATASET_NAME=hh
BATCH_SIZE=16
EVAL_BATCH_SIZE=4
# SEEDS=('1' '2' '3')
SEEDS=('1')
# BETAS=('0.01' '0.05' '0.1')
BETAS=('0.05')

for SEED in "${SEEDS[@]}"; do
for BETA in "${BETAS[@]}"; do
OUTPUT_DIR=models/is_dpo/Llama-3.2-${MODEL_SIZE}-is_dpo_beta_${BETA}_eps_1.0_tldr_pref_shift_seed_${SEED}

accelerate launch --config_file accelerate_configs/deepspeed_zero3_4gpu.yaml \
    --main_process_port=5500\
    cli/train_is_dpo.py\
    --model_path=$SFT_MODEL_PATH \
    --batch_size=$BATCH_SIZE \
    --seed=$SEED \
    --eps=1.0 \
    --eval_every=19200 \
    --beta=${BETA} \
    --exp_name=Llama-3.2-${MODEL_SIZE}_is_dpo_beta_${BETA}_eps_1.0_hh_pref_seed_${SEED} \
    --eval_batch_size=$EVAL_BATCH_SIZE \
    --dataset_name=$DATASET_NAME \
    --no-use-packing \
    --num_train_epochs=1\
    --gradient_accumulation_steps=$GRAD_ACCUMULATION_STEPS \
    --no-debug \
    --warm-up-steps=50 \
    --lr=1e-6\
    --output_dir=$OUTPUT_DIR 
done
done