# export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# export TOKENIZERS_PARALLELISM=0
export CUDA_VISIBLE_DEVICES=0,1,2,3



#---- ner ----

python benchmarks.py \
--dataset nwgi,fiqa,fpb,tfns \
--base_model meta-llama/Llama-3.1-8B-Instruct \
# --peft_model ../finetuned_models/meta-llama-Llama-3.1-8B-Instruct/data-train-fingpt_sentiment_train.jsonl-20241023T0144-meta-llama-Llama-3.1-8B-Instruct-8bits-r8 \
--batch_size 20 \
--max_length 512 \

