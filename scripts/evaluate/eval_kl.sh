export CUDA_VISIBLE_DEVICES=3

# MODEL_SIZES=('gpt2' "gpt2-medium" "gpt2-large")
# STEPS=('12800' '25600' '38400' '51200' '64000' '76800' "80000")
STEPS=("1600" "3200" "4800" "6400" "8000" "9600" "11200" "11200" "12800")
# # STEPS=("12480" "24960" "37440" "49920" "62400" "74880" "87360" "99840" "111352")
# STEPS=("6720" "13440" "20160" "26880" "32128" "33600" "40320" "47040" "53760" "60480" "64000")
# GAMMAS=("0.1" "0.5" "1.0")
# GAMMAS=("0.1" "1.0" "2.0")
# GAMMAS=("1.0")
BETAS=('0.01' "0.05" "0.1")
LOSS_NAME=dpo
WEAK_MODEL_PATH=qwen1_5b

# models/sft/gpt2-large_sft_tldr_gpt2_dpo/step-${STEP}
# for GAMMA in "${GAMMAS[@]}"; do
for BETA in "${BETAS[@]}"; do
for STEP in "${STEPS[@]}"; do
accelerate launch get_kl.py --model_path=openai-community/gpt2-large\
                            --ref_model_path=models/sft/gpt2_sft_tldr/step-116722\
                            --dataset_name=tldr_pref \
                            --seed=42\
                            --num_samples=512\
                            --key_name=gpt2-large_sft_gpt2_dpo_kl_step-${STEP}

# accelerate launch calc_kl.py --model_path=models/${LOSS_NAME}/Qwen2.5-7B_${LOSS_NAME}_alpaca_w2s_refine_gamma_${GAMMA}/step-${STEP}\
#                     --ref_model_path=models/sft/${WEAK_MODEL_PATH}_${LOSS_NAME}_alpaca_w2s/step-16000\
#                     --batch_size=16\
#                     --key_name=${WEAK_MODEL_PATH}-w2s-Qwen2.5-7B-refine_${LOSS_NAME}_gamma_${GAMMA}_step-${STEP}\
#                     --judge_path=DatPySci/Llama-3.1-8B-rm-mixture\
#                     --temperature=0.7\
#                     --top_p=0.95
# accelerate launch eval_gsm8k.py --model_path=models/${LOSS_NAME}/${WEAK_MODEL_PATH}_${LOSS_NAME}_beta_${BETA}_alpaca_w2s/step-${STEP}\
#                     --ref_model_path=Qwen/Qwen2.5-7B-Instruct\
#                     --batch_size=16\
#                     --key_name=${WEAK_MODEL_PATH}-w2s-Qwen2.5-7B-${LOSS_NAME}_beta_${BETA}_step-${STEP}\
#                     --judge_path=DatPySci/Llama-3.1-8B-rm-mixture\
#                     --temperature=0.7\
#                     --top_p=0.95
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