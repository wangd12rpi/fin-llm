#!/bin/bash
#SBATCH -J LLaMA2-13B-hf-lr       # Task name
#SBATCH -o LLaMA2-13B-lr-fin_oneGPU.out     # output file ***.out
#SBATCH -N 1                        # node count
#SBATCH --cpus-per-task=64          # CPU count per node
#SBATCH --gres=gpu:1                # GPU count per node

deepspeed train_lora.py
