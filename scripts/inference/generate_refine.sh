#!/bin/bash
#SBATCH -c 16 # request two cores 
#SBATCH -p kisski-h100
#SBATCH -o log/exp-generate-Qwen1.5B.out
#SBATCH -e log/error_exp-generate-Qwen1.5B.out
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=generate-Qwen1.5B
#SBATCH --ntasks-per-node=1
#SBATCH -G H100:1
export CUDA_VISIBLE_DEVICES=0

source ~/.shadow1
conda activate rl_train

# MODEL_PATHS=("models/weso/Qwen2.5-7B_weso_ultrachat_refine_gamma_1.0/step-196284")
MODEL_PATHS=("models/dpo/Qwen2.5-7B_dpo_beta_0.05_ultrachat_pref/step-196284")

SPLITS=("train")
DATASET_NAME=ultrachat_w2s_iter1

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
for SPLIT in "${SPLITS[@]}"; do
python inference/generate_refine.py --model_path=$MODEL_PATH\
                             --dataset_name=ultrachat_w2s_iter1\
                             --batch_size=4096\
                             --split=${SPLIT}
done
done