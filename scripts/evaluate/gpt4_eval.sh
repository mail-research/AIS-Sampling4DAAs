nvidia-smi
export CUDA_VISIBLE_DEVICES=1

MODEL_SIZES=('3B')
# STEPS=('12800' '25600' '38400' '51200' '64000' '76800' '89600' '102400' '115200' '127442')
# STEPS=("12480" "24960" "37440" "49920" "62400" "74880" "87360" "99840"
STEPS=("32000")
SEEDS=('1')
BETAS=("0.1")
DATASET_NAME=tldr_pref
MODEL_SIZE=3B

# WEAK_MODELS=('gpt2' 'gpt2_medium' 'gpt2_large' 'llama3b')
WEAK_MODELS=('gpt2-large')

for WEAK_MODEL in "${WEAK_MODELS[@]}"; do
for SEED in "${SEEDS[@]}"; do
for BETA in "${BETAS[@]}"; do
for STEP in "${STEPS[@]}"; do
python RLHF-training/llm_as_judge.py --model_path=Ruler/Align/models/chipo_2/llama3_3b_sft_chipo_beta_0.01_hh_ep1/step-111352 \
                                    --dataset_name=hh \
                                    --ref_model_path=DatPySci/Llama-3.2-3B-sft-mixture \
                                    --num_samples=256 \
                                    --batch_size=16 \
                                    --judge_path=DatPySci/Llama-3.1-8B-rm-mixture \
                                    --key_name=Llama-3.2-3B_chipo_hh_beta_0.05\
                                    --temperature=0.7 \
                                    --top_p=0.95
# accelerate launch llm_as_judge.py --model_path=models/weso/gpt2_large_weso_tldr_w2s_refine_gamma_0.5/step-32000\
#                     --ref_model_path=meta-llama/Llama-3.2-3B\
#                     --dataset_name=${DATASET_NAME} \
#                     --key_name=Llama-3.2-3B_weso_weak_${WEAK_MODEL}_refine_${DATASET_NAME}_step-${STEP}\
#                     --num_samples=256\
done
done
done
done