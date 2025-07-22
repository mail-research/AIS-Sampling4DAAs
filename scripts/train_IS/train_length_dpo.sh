#!/bin/bash
#SBATCH -c 2 # request two cores 
#SBATCH -p preempt
#SBATCH -o log/experiment-rlhf-dpo.out
#SBATCH -e log/error_experiment-rlhf-dpo.out
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=dpo-bash
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:2

nvidia-smi
export CUDA_VISIBLE_DEVICES=2,3
MODEL_SIZES=('3B')
GRAD_ACCUMULATION_STEPS=4
DATASET_NAME=hh
BATCH_SIZE=16
EVAL_BATCH_SIZE=4
SEEDS=('3')
BETAS=('0.01')

for MODEL_SIZE in "${MODEL_SIZES[@]}"; do
for SEED in "${SEEDS[@]}"; do
for BETA in "${BETAS[@]}"; do
OUTPUT_DIR=models/length_dpo/Llama-3.2-${MODEL_SIZE}-length_dpo_beta_${BETA}_hh_seed_${SEED}
SFT_MODEL_PATH=DatPySci/Llama-3.2-3B-sft-mixture

accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
    --main_process_port=5500\
    cli/train_length_dpo.py\
    --model_path=$SFT_MODEL_PATH \
    --batch_size=$BATCH_SIZE \
    --seed=$SEED \
    --eval_every=12480 \
    --beta=${BETA} \
    --exp_name=Llama-3.2-${MODEL_SIZE}_length_dpo_beta_${BETA}_hh_seed_${SEED} \
    --eval_batch_size=$EVAL_BATCH_SIZE \
    --dataset_name=$DATASET_NAME \
    --no-use-packing \
    --num_train_epochs=1\
    --gradient_accumulation_steps=$GRAD_ACCUMULATION_STEPS \
    --no-debug \
    --warm-up-steps=150 \
    --lr=1e-6\
    --output_dir=$OUTPUT_DIR 
done
done
done