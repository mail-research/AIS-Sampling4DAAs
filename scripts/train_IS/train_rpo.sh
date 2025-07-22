#!/bin/bash
#SBATCH -c 2 # request two cores 
#SBATCH -p preempt
#SBATCH -o log/experiment-rlhf-dpo-1B.out
#SBATCH -e log/error_experiment-rlhf-dpo-1B.out
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=dpo-bash
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:2

nvidia-smi
MODEL_SIZES=('3B')
GRAD_ACCUMULATION_STEPS=2
DATASET_NAME=hh
BATCH_SIZE=32
EVAL_BATCH_SIZE=4
SEEDS=('1' '2' '3')
BETAS=('0.01' '0.05' '0.1')

for MODEL_SIZE in "${MODEL_SIZES[@]}"; do
for SEED in "${SEEDS[@]}"; do
for BETA in "${BETAS[@]}"; do
OUTPUT_DIR=models/rpo/Llama-3.2-${MODEL_SIZE}-rpo_beta_${BETA}_hh_seed_${SEED}
SFT_MODEL_PATH=DatPySci/Llama-3.2-3B-sft-mixture

accelerate launch --config_file accelerate_configs/deepspeed_zero3_4gpu.yaml \
    --main_process_port=4500\
    cli/train_rpo.py\
    --model_path=$SFT_MODEL_PATH \
    --batch_size=$BATCH_SIZE \
    --seed=$SEED \
    --eval_every=12480 \
    --beta=${BETA} \
    --exp_name=Llama-3.2-${MODEL_SIZE}_rpo_beta_${BETA}_hh_seed_${SEED} \
    --eval_batch_size=$EVAL_BATCH_SIZE \
    --dataset_name=$DATASET_NAME \
    --no-use-packing \
    --num_train_epochs=1\
    --gradient_accumulation_steps=$GRAD_ACCUMULATION_STEPS \
    --no-debug \
    --warm-up-steps=100 \
    --lr=1e-6\
    --output_dir=$OUTPUT_DIR 
done
done
done