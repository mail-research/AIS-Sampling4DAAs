#!/bin/bash
#SBATCH -c 16 # request two cores 
#SBATCH -p kisski-h100
#SBATCH -o log/pref-label.out
#SBATCH -e log/error-pref-label.out
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=weak-pref-label
#SBATCH --ntasks-per-node=1
#SBATCH -G H100:1

source ~/.shadow1
conda activate align

export CUDA_VISIBLE_DEVICES=0

ALIGNED_MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
BASE_MODEL_PATH=Qwen/Qwen2.5-1.5B

accelerate launch inference/label.py --dataset_name=ultrachat_qwen7b\
                --judge_path=$ALIGNED_MODEL_PATH \
                --implicit\
                --batch_size=16\
                --model_path=${ALIGNED_MODEL_PATH}\
                --ref_model_path=${BASE_MODEL_PATH}\
                --split=train