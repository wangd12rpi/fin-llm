# export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# export TOKENIZERS_PARALLELISM=0
export CUDA_VISIBLE_DEVICES=0,1,2,3



#---- ner ----

python benchmarks.py \
--dataset fpb \
--base_model mistralai/Mistral-Small-Instruct-2409 \
#--peft_model ../finetuned_models/mistralai-Mistral-Small-Instruct-2409/20241022T0521-mistralai-Mistral-Small-Instruct-2409-FinGPT-fingpt-ner-cls-8bits-r8 \
--batch_size 8 \
--max_length 512 \

