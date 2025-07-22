#!/bin/bash
#SBATCH -c 2 # request two cores 
#SBATCH -p kisski-h100
#SBATCH -o log/eval/experiment-rlhf-eval-tldr.out
#SBATCH -e log/eval/error_experiment-rlhf-eval-tldr.out
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=eval-bash
#SBATCH --ntasks-per-node=1

export CUDA_VISIBLE_DEVICES=1

# MODEL_SIZES=('gpt2' "gpt2-medium" "gpt2-large")
# STEPS=('12800' '25600' '38400' '51200' '64000' '76800'
# STEPS=("76800" '89600' '102400' '115200' '128000')
# STEPS=("38400" "57600" "76800"
STEPS=("96000" "115200" "134400" "153600")
# STEPS=("19200")
# STEPS=("16000")
# # STEPS=("12480" "24960" "37440" "49920" "62400" "74880" "87360" "99840" "111352")
# STEPS=("6720" "13440" "20160" "26880" "32128" "33600" "40320" "47040" "53760" "60480" "64000")
SEEDS=("1")
# BETAS=("0.01" "0.05" "0.1")
BETAS=("0.1")
DATASET_NAME=tldr_pref
# MODEL_SIZE=3B
LOSS_NAME=dpo
MODEL_PATH=Llama-3.2-3B-sft
# --model_path=models/${LOSS_NAME}/LlaDatPySci/Llama-3.2-3B-sft-mixture \ma-3.2-${MODEL_SIZE}-${LOSS_NAME}_beta_${BETA}_tldr_pref_shift_seed_${SEED}/step-${STEP}\
# DatPySci/Llama-3.2-3B-sft-mixture \
# for MODEL_SIZE in "${MODEL_SIZES[@]}"; do
for SEED in "${SEEDS[@]}"; do
for BETA in "${BETAS[@]}"; do
for STEP in "${STEPS[@]}"; do
# accelerate launch fast_eval.py --model_path=models/${LOSS_NAME}/${MODEL_PATH}_online_dpo_beta_${BETA}_tldr_sft/step-${STEP}\
accelerate launch fast_eval.py --model_path=models/dpo/Llama-3.2-3B-sft_online_dpo_beta_0.1_tldr_sft/step-${STEP}\
                    --ref_model_path=DatPySci/Llama-3.2-3B-sft-mixture \
                    --dataset_name=${DATASET_NAME} \
                    --batch_size=16\
                    --key_name=Llama-3.2-online-dpo_beta_${BETA}_seed_${SEED}_step-${STEP}\
                    --judge_path=DatPySci/Llama-3.1-8B-rm-mixture\
                    --num_samples=512\
                    --temperature=0.7\
                    --top_p=0.95
done
done
done
# done