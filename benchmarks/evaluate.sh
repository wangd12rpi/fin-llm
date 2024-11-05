# export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# export TOKENIZERS_PARALLELISM=0
export CUDA_VISIBLE_DEVICES=0,1,2,3



#---- ner ----

python benchmarks.py \
--dataset nwgi,fiqa,fpb,tfns \
--base_model meta-llama/Llama-3.1-8B-Instruct \
--peft_model ../train/ft_models/senti/llama_3.1_8b_8bits_r8 \
--batch_size 20 \
--max_length 512 \

