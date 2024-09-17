# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import warnings
warnings.filterwarnings("ignore")

import sys

sys.path.append("/colab_space/yanglet/guoxuan/Task3/ChatGLM/inferencing")
# sys.path.append("/colab_space/yanglet/guoxuan/Task3/ChatGLM_tl")

# %%
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score, auc
from tqdm import tqdm
import pandas as pd
import json, torch

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast
from datasets import load_dataset

from cover_alpaca2jsonl import format_example

# sys.path.append("/xfs/home/tensor_zy/guoxuan/Task3/ChatGLM/inferencing")
sys.path.append("/colab_space/yanglet/guoxuan/Task3/llama2_tl")
# %%
# %%
# Load Models
base_model = "daryl149/Llama-2-7b-chat-hf"
# peft_model = "oliverwang15/FinGPT_ChatGLM2_Sentiment_Instruction_LoRA_FT"
# peft_model = "../../finetuned_model"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map = "auto")
# model = PeftModel.from_pretrained(model, peft_model)
from peft_tl import PeftModel
peft_path = "../../finetuned_model_CP/"
model = PeftModel.from_pretrained(model, peft_path)
# model = torch.compile(model)
model = model.eval()

# %% [markdown]
# ### Load Dataset

# %%
from datasets import load_dataset
import datasets

dataset = load_dataset("boolq")
dataset = dataset['validation']

# %%
dataset = dataset.to_pandas()
dataset['answer'] = dataset['answer'].astype('str')
dataset.head(2)

# %%
dataset.columns = ['instruction', 'output', 'input']

# %%
dataset[["context","target"]] = dataset.apply(format_example, axis = 1, result_type="expand")
dataset.head(2)

# %%
context = dataset['context'].tolist()
len(context)

# %%
batch_size = 2

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
out_text_list[:10]

# %%
def change_target(x):
    if "True" in x or 'true' in x:
        return "True"
    elif 'False' in x or 'false' in x:
        return "False"
    else:
        return 'missing'

# %%
# dataset["new_target"] = dataset["target"].apply(change_target)
dataset["new_target"] = dataset["target"]
dataset["new_target"].hist()

# %%
dataset["new_out"] = dataset["out_text"].apply(change_target)
dataset["new_out"].hist()

# %% [markdown]
# ### 8-bit

# %%
acc = accuracy_score(dataset["new_target"], dataset["new_out"])
acc

# %%
f1 = f1_score(dataset["new_target"], dataset["new_out"], average = "macro")
f1

# %%
f1 = f1_score(dataset["new_target"], dataset["new_out"], average = "micro")
f1

# %%
f1 = f1_score(dataset["new_target"], dataset["new_out"], average = "weighted")
f1

# %% [markdown]
# ### Full

# %%
acc = accuracy_score(dataset["new_target"], dataset["new_out"])
acc

# %%
f1 = f1_score(dataset["new_target"], dataset["new_out"], average = "macro")
f1

# %%
f1 = f1_score(dataset["new_target"], dataset["new_out"], average = "micro")
f1

# %%
f1 = f1_score(dataset["new_target"], dataset["new_out"], average = "weighted")
f1

# %%


# %%


# %%
dataset.to_csv("fiqa_ori.csv")


