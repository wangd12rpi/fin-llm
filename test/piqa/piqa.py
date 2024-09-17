# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import warnings
warnings.filterwarnings("ignore")

import sys

sys.path.append("/colab_space/yanglet/guoxuan/Task3/ChatGLM/inferencing")
# sys.path.append("/colab_space/yanglet/guoxuan/Task3/ChatGLM_tl")

# %%
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score, auc
from tqdm import tqdm
from peft import PeftModel
import pandas as pd
import json, torch

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast,BitsAndBytesConfig
from datasets import load_dataset

from cover_alpaca2jsonl import format_example

# sys.path.append("/xfs/home/tensor_zy/guoxuan/Task3/ChatGLM/inferencing")
sys.path.append("/colab_space/yanglet/guoxuan/Task3/llama2_tl")
# %%
# %%
# Load Models
base_model = "daryl149/Llama-2-13b-chat-hf"
# base_model = '/colab_space/yanglet/models--daryl149--Llama-2-7b-chat-hf/snapshots/bbc9b373dacff93e600e4426f2b3d3dd264e90ed'
# peft_model = "oliverwang15/FinGPT_ChatGLM2_Sentiment_Instruction_LoRA_FT"
# peft_model = "../../finetuned_model"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, torch_dtype=torch.float16, device_map = "auto")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

model = LlamaForCausalLM.from_pretrained(
    base_model, 
    trust_remote_code=True,
    # cache_dir='/colab_space/yanglet/guoxuan/model_cache', 
    # load_in_8bit=True, 
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map = "auto",
    quantization_config = quantization_config
    )

# model = prepare_model_for_int8_training(model)

# peft_path = "../../finetuned_model_fin_CP/"
# peft_path = "../../finetuned_model_13B_rl_fin_oneGPU/"
# peft_path = "../../finetuned_model_13B_rl_fin/"
peft_path = "../../LLama2-13B-4bit-gen-r8"
model = PeftModel.from_pretrained(model, peft_path)
# model = torch.compile(model)
model = model.eval()

# %% [markdown]
# ### Load Dataset

# %%
from datasets import load_dataset
import datasets

dataset = load_dataset("piqa")
dataset = dataset['validation']
dataset

# %%
dataset = dataset.to_pandas()
dataset['instruction'] = 'Please choose the best answer from the two options.'
dataset['input'] = dataset.apply(lambda x:x['goal'] + " {" + x['sol1']+"/"+x["sol2"]+"}.",axis = 1)
dataset['output'] = dataset.apply(lambda x:x['sol1'] if x['label'] == 0 else x['sol2'], axis = 1)
dataset.head(2)

# %%
dataset[["context","target"]] = dataset.apply(format_example, axis = 1, result_type="expand")
dataset.head(2)

# %%
context = dataset['context'].tolist()
len(context)

# %%
batch_size = 8

total_steps = dataset.shape[0]//batch_size
total_steps

# %%
res_list = []
res_sentences_list = []
out_text_list = []

for i in tqdm(range(total_steps+1)):
    tmp_context = context[i* batch_size:(i+1)* batch_size]
    tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=512)
    tokens.pop('token_type_ids')
    for k in tokens.keys():
        tokens[k] = tokens[k].cuda()
    
    res = model.generate(**tokens, max_length=512)
    res_sentences = [tokenizer.decode(i) for i in res]
    out_text = [o.split("Answer: ")[1] for o in res_sentences]
    res_list += res
    res_sentences_list += res_sentences
    out_text_list += out_text
    torch.cuda.empty_cache()

# %%
res_list = [i.cpu() for i in res_list]

# %%
dataset["res"] = res_list
dataset["res_sentences"] = res_sentences_list
dataset["out_text"] = out_text_list

# %%
dataset['sol1_token'] = dataset['sol1'].apply(lambda x: tokenizer.encode(x))
dataset['sol2_token'] = dataset['sol2'].apply(lambda x: tokenizer.encode(x))
dataset['out_token'] = dataset['out_text'].apply(lambda x: tokenizer.encode(x))

# %%
import difflib

# %%
dataset['sol1_score'] = dataset.apply(lambda x: difflib.SequenceMatcher(None, x['sol1_token'], x['out_token']).ratio(), axis = 1)
dataset['sol2_score'] = dataset.apply(lambda x: difflib.SequenceMatcher(None, x['sol2_token'], x['out_token']).ratio(), axis = 1)
dataset['new_out'] = dataset.apply(lambda x:0 if x['sol1_score'] > x['sol2_score'] else 1, axis = 1)
dataset[['sol1_score', 'sol2_score', 'new_out']]

# %%
dataset['target']

# %%
dataset["new_target"] = dataset["label"]
dataset["new_target"].hist()


# %% [markdown]
# ### 8-bit

# %%
acc = accuracy_score(dataset["new_target"], dataset["new_out"])
acc

# %%
f1_macro = f1_score(dataset["new_target"], dataset["new_out"], average = "macro")


# %%
f1_micro = f1_score(dataset["new_target"], dataset["new_out"], average = "micro")


# %%
f1_weighted = f1_score(dataset["new_target"], dataset["new_out"], average = "weighted")

print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted (BloombergGPT): {f1_weighted}. ")

# TT r64_sc2 Acc: 0.7317736670293797. F1 macro: 0.7316401322883654. F1 micro: 0.7317736670293796. F1 weighted (BloombergGPT): 0.7316987572966157. 
# %% [markdown]
# ### Full

# %%


# %%


# %%
dataset.to_csv("fiqa_ori.csv")


