#!/bin/bash
#SBATCH -c 8 # request two cores 
#SBATCH -p kisski-h100
#SBATCH -o log/experiment-weso.out
#SBATCH -e log/error_experiment-weso.out
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=weso-logprobs
#SBATCH --ntasks-per-node=1
#SBATCH -G H100:4

nvidia-smi
source ~/.shadow1
conda activate align
# export CUDA_VISIBLE_DEVICES=0
# MODEL_SIZES=('gpt2_medium' 'gpt2_large')

GRAD_ACCUMULATION_STEPS=8
# DATASET_NAME=alpaca_qwen1_5b_refine
# DATASET_NAME=alpaca_qwen7b
BATCH_SIZE=16
EVAL_BATCH_SIZE=16
SEEDS=('1')
GAMMAS=("1.0")

SFT_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
DATASET_NAME=ultrachat_qwen7b
# ALIGNED_MODEL_PATH=models/dpo/${MODEL_PATH}_dpo_beta_0.1_tldr_pref/step-92858
# BASE_MODEL_PATH=models/sft/${MODEL_PATH}_sft_tldr/step-108722
# DATASET_NAME=${MODEL_PATH}_tldr_refine

# for MODEL_PATH in "${MODEL_SIZES[@]}"; do
for SEED in "${SEEDS[@]}"; do
for GAMMA in "${GAMMAS[@]}"; do
ALIGNED_MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
BASE_MODEL_PATH=Qwen/Qwen2.5-1.5B
OUTPUT_DIR=models/weso/${MODEL_PATH}_weso_ultrachat_refine_gamma_${GAMMA}
LOCAL_RUN_DIR=data/weso/Qwen2.5-7B_refine_Qwen2.5-1.5B_gamma_${GAMMA}

accelerate launch --config_file accelerate_configs/deepspeed_zero3_4gpu_avg.yaml \
    --main_process_port=2550\
    cli/train_weso.py\
    --load_reference_logprobs=${LOCAL_RUN_DIR}/cached_reference_logprobs.pkl\
    --model_path=$SFT_MODEL_PATH \
    --gamma=$GAMMA\
    --local_run_dir=$LOCAL_RUN_DIR\
    --base_weak_model_path=$BASE_MODEL_PATH \
    --aligned_weak_model_path=$ALIGNED_MODEL_PATH \
    --batch_size=$BATCH_SIZE \
    --seed=$SEED \
    --debug\
    --eval_every=192_0000 \
    --use-liger \
    --project_name=weak-to-strong-WESO\
    --exp_name=Qwen2.5-7B-w2s-Qwen2.5-1.5B_weso_gamma_${GAMMA}_ultrachat_seed_${SEED} \
    --eval_batch_size=$EVAL_BATCH_SIZE \
    --dataset_name=$DATASET_NAME \
    --no-use-packing \
    --num_train_epochs=1\
    --gradient_accumulation_steps=$GRAD_ACCUMULATION_STEPS \
    --warm-up-steps=100 \
    --lr=1e-5\
    --output_dir=$OUTPUT_DIR 
done
done
# done