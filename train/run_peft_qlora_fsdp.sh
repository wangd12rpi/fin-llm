#!/bin/bash


# Task selection (headline, ner, senti, xbrl)
task="senti"  
quant_bits=8  # Can be 4 or 8
lora_r=8
model_name_short="llama_3.1_8b"  # Can be "llama_3.1_8b" or "llama_3.1_70b"

# Map tasks to dataset paths
declare -A dataset_map=(
  ["headline"]="../data/train/fingpt_headline_train.jsonl"
  ["ner"]="../data/train/fingpt_ner_cls_train.jsonl"
  ["senti"]="../data/train/fingpt_sentiment_train.jsonl"
  ["xbrl"]="../data/train/xbrl_train.jsonl"
)



# Map short model names to full Hugging Face model names
declare -A model_map=(
  ["llama_3.1_8b"]="meta-llama/Llama-3.1-8B-Instruct"
  ["llama_3.1_70b"]="meta-llama/Llama-3.1-70B-Instruct"
)

# Construct output directory
output_dir="ft_models/${task}/${model_name_short}_${quant_bits}bits_r${lora_r}"

# Use 8-bit quantization if quant_bits is 8, otherwise use 4-bit
use_8bit_quantization=$([ "$quant_bits" == "8" ]; echo true; echo false)


# Start the training job in a detached tmux session
tmux new-session -d -s "training_job_${task}" '
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  eval "$(conda shell.bash hook)"
  conda activate finenv
  accelerate launch --config_file "configs/fsdp_config_qlora.yaml"  train.py \
  --seed 100 \
  --model_name_or_path '"${model_map[$model_name_short]}"' \
  --dataset_name '"${dataset_map[$task]}"' \
  --chat_template_format "none" \
  --add_special_tokens False \
  --append_concat_token False \
  --splits "train,test" \
  --max_seq_len 2048 \
  --num_train_epochs 4 \
  --logging_steps 0.05 \
  --log_level "info" \
  --logging_strategy "steps" \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --hub_private_repo True \
  --hub_strategy "every_save" \
  --bf16 True \
  --packing False \
  --learning_rate 1e-4 \
  --lr_scheduler_type "cosine" \
  --weight_decay 1e-4 \
  --warmup_ratio 0.0 \
  --max_grad_norm 1.0 \
  --output_dir "'"$output_dir"'" \
  \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing True \
  --use_reentrant True \
  --dataset_text_field "content" \
  --use_flash_attn True \
  --use_peft_lora True \
  --lora_r '"$lora_r"' \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --lora_target_modules "all-linear" \
  --use_8bit_quantization '"$use_8bit_quantization"' \
  --use_nested_quant True \
  --bnb_4bit_compute_dtype "bfloat16" \
  --bnb_4bit_quant_storage_dtype "bfloat16"

  read -p "Press Enter to exit..."
'


tmux attach -t "training_job_${task}"

