export CUDA_VISIBLE_DEVICES=2,3

WESO_PATH=models/weso/Qwen2.5-7B_weso_alpaca_w2s_refine_gamma_1.0/step-80000
DPO_PATH=models/dpo/qwen1_5b_dpo_beta_0.05_alpaca_w2s/step-80000
 lm_eval --model hf \
        --model_args pretrained=${DPO_PATH},dtype=bfloat16,parallelize=true\
        --tasks mmlu \
        --num_fewshot 5\
        --batch_size 16 \
        --output_path lm_eval_results\
        --apply_chat_template

lm_eval --model hf \
        --model_args pretrained=${DPO_PATH},dtype=bfloat16,parallelize=true\
        --tasks winogrande \
        --num_fewshot 5\
        --batch_size 16 \
        --output_path lm_eval_results\
        --apply_chat_template

lm_eval --model hf \
        --model_args pretrained=${DPO_PATH},dtype=bfloat16,parallelize=true\
        --tasks  gsm8k\
        --num_fewshot 0\
        --batch_size 16 \
        --output_path lm_eval_results\
        --apply_chat_template

lm_eval --model hf \
        --model_args pretrained=${DPO_PATH},dtype=bfloat16,parallelize=true\
        --tasks arc_challenge \
        --num_fewshot 25\
        --batch_size 16 \
        --output_path lm_eval_results\
        --apply_chat_template

lm_eval --model hf \
        --model_args pretrained=${DPO_PATH},dtype=bfloat16,parallelize=true\
        --tasks hellaswag \
        --num_fewshot 10\
        --batch_size 16 \
        --output_path lm_eval_results\
        --apply_chat_template

