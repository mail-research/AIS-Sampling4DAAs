export CUDA_VISIBLE_DEVICES=0

# MODEL_SIZES=('gpt2' "gpt2-medium" "gpt2-large")
STEPS=('12800' '25600' '38400' '51200' '64000' '76800')
STEPS=("80000")
# # STEPS=("12480" "24960" "37440" "49920" "62400" "74880" "87360" "99840" "111352")
# STEPS=("6720" "13440" "20160" "26880" "32128" "33600" "40320" "47040" "53760" "60480" "64000")
# GAMMAS=("0.1" "0.5" "1.0")
GAMMAS=("2.0")
# BETAS=("0.01" "0.1")
LOSS_NAME=sft
WEAK_MODEL_PATH=qwen1_5b
# accelerate launch alpaca_judge.py --model_path=models/weso/Qwen2.5-7B_weso_alpaca_w2s_refine_gamma_0.05/step-${STEP}\
# for BETA in "${BETAS[@]}"; do
for GAMMA in "${GAMMAS[@]}"; do
for STEP in "${STEPS[@]}"; do
accelerate launch alpaca_judge.py --model_path=models/sft/qwen7b_sft_alpaca_w2s/step-${STEP}\
                    --ref_model_path=models/sft/${WEAK_MODEL_PATH}_${LOSS_NAME}_alpaca_w2s/step-16000\
                    --batch_size=16\
                    --key_name=${WEAK_MODEL_PATH}-w2s-Qwen2.5-7B-refine_${LOSS_NAME}_step-${STEP}\
                    --judge_path=DatPySci/Llama-3.1-8B-rm-mixture\
                    --temperature=0.7\
                    --top_p=0.95
done
done
# for STEP in "${STEPS[@]}"; do
# accelerate launch fast_eval.py --model_path=models/${LOSS_NAME}/${MODEL_PATH}_${LOSS_NAME}_hh_w2s_pref/step-${STEP}\
#                     --ref_model_path=models/sft/${MODEL_PATH}_sft_hh_w2s/step-16000\
#                     --dataset_name=${DATASET_NAME} \
#                     --batch_size=8\
#                     --key_name=Llama-3.2-w2s-Llama-3.2-1B-${LOSS_NAME}_gamma_1.0_hh_seed_${SEED}_step-${STEP}\
#                     --judge_path=DatPySci/Llama-3.1-8B-rm-mixture\
#                     --num_samples=1024\
#                     --temperature=0.7\
#                     --top_p=0.95
# done
# done