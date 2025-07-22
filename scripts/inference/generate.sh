export CUDA_VISIBLE_DEVICES=1

# MODEL_PATHS=("models/dpo/gpt2_dpo_beta_0.1_tldr_pref/step-92858")
MODEL_PATHS=("Qwen/Qwen2.5-1.5B-Instruct")
# MODEL_PATHS=("models/dpo/Llama-3.2-1B_dpo_beta_0.1_ultrachat_pref/step-61135")
SPLITS=("train")

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
for SPLIT in "${SPLITS[@]}"; do
python inference/generate_refine.py --model_path=$MODEL_PATH\
                             --dataset_name=ultrachat_w2s\
                             --batch_size=4096\
                             --split=${SPLIT}
done
done
