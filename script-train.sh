export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_IGNORE_DISABLED_P2P=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0



deepspeed train_lora.py \
--run_name test1 \
--base_model meta-llama/Llama-3.1-8B-Instruct \
--dataset sentiment-train \
--max_length 512 \
--batch_size 8 \
--learning_rate 1e-4 \
--num_epochs 1 \
--log_interval 10 \
--warmup_ratio 0.03 \
--scheduler linear \
--evaluation_strategy steps \
--ds_config config_.json
