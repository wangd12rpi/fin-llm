
# Task selection (headline, ner, senti, xbrl)
task="xbrl"  
quant_bits=8  
lora_r=8
model_name_short="llama_3.1_8b"  # Can be "llama_3.1_8b" or "llama_3.1_70b"

# Map tasks to dataset paths
declare -A dataset_map=(
  ["headline"]="./data/train/fingpt_headline_train.jsonl"
  ["ner"]="./data/train/fingpt_ner_cls_train.jsonl"
  ["senti"]="./data/train/fingpt_sentiment_train.jsonl"
  ["xbrl_tags"]="./xbrl/xbrl_xbrl_tags_train.jsonl"
  ["xbrl_value"]="./xbrl/xbrl_value_train.jsonl"
  ["xbrl"]="./xbrl/xbrl_train.jsonl"
  ["xbrl_formula"]="./xbrl/xbrl_formula_formatted_with_tags_train.jsonl"
)



# Map short model names to full Hugging Face model names
declare -A model_map=(
  ["llama_3.1_8b"]="meta-llama/Llama-3.1-8B-Instruct"
  ["llama_3.1_70b"]="meta-llama/Llama-3.1-70B-Instruct"
)


# Start the training job in a detached tmux session
tmux new-session -d -s "training_job_${task}" '
  export CUDA_VISIBLE_DEVICES=0,1,4,7
    export NCCL_IGNORE_DISABLED_P2P=1
    export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
    export TOKENIZERS_PARALLELISM=0
  eval "$(conda shell.bash hook)"
  conda activate finenv
  
   deepspeed train_lora.py \
    --base_model '"${model_map[$model_name_short]}"' \
    --dataset '"${dataset_map[$task]}"' \
    --max_length 128000 \
    --batch_size 2 \
    --grad_accu 2 \
    --learning_rate 5e-5 \
    --num_epochs 1 \
    --log_interval 10 \
    --warmup_ratio 0.03 \
    --scheduler linear \
    --evaluation_strategy steps \
    --ds_config config_.json \
    --eval_steps 0.05 \
    --quant_bits '"$quant_bits"' \
    --r '"$quant_bits"' \
    --max_steps -1 

  read -p "Press Enter to exit..."
'


tmux attach -t "training_job_${task}"
