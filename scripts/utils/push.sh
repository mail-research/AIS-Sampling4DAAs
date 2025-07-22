#!/bin/bash
#SBATCH -c 2 # request two cores 
#SBATCH -p preempt
#SBATCH -o log/experiment-rlhf-push.out
#SBATCH -e log/error_experiment-rlhf-push.out
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=bash
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:h100:1

python push.py --model_path=models/dpo/Llama-3.2-3B-dpo_beta_0.05_hh_seed_1/step-127442 --repo_id=Llama-3.2-3B-dpo-hh-beta-0.05