# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import warnings
warnings.filterwarnings("ignore")

# %%
import torch
torch.cuda.is_available()
import sys
# sys.path.append("/xfs/home/tensor_zy/guoxuan/Task3/ChatGLM/inferencing")
sys.path.append("/colab_space/yanglet/guoxuan/Task3/llama2_tl")

# %%
from sklearn.metrics import accuracy_score,f1_score
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
import pandas as pd
import datasets
import torch


# %%
# Load Models
base_model = "daryl149/Llama-2-13b-chat-hf"
# base_model = '/colab_space/yanglet/models--daryl149--Llama-2-7b-chat-hf/snapshots/bbc9b373dacff93e600e4426f2b3d3dd264e90ed'
# peft_model = "oliverwang15/FinGPT_ChatGLM2_Sentiment_Instruction_LoRA_FT"
# peft_model = "../../finetuned_model"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, torch_dtype=torch.float16, device_map = "auto")
# model = PeftModel.from_pretrained(model, peft_model)

model = LlamaForCausalLM.from_pretrained(
    base_model,
    cache_dir='/colab_space/yanglet/guoxuan/model_cache', 
    # load_in_8bit=True,
    trust_remote_code=True, 
    torch_dtype=torch.float16,
    # device='cuda',
    # device_map = f'cuda:{local_rank}',
    device_map = "auto"
)

# peft_path = "../../finetuned_model_fin_CP/"
# peft_path = "../../finetuned_model_fin_TT_r64_sc4/"
# peft_path = "../../finetuned_model_13B_rl_fin_oneGPU/"
# peft_path = "../../finetuned_model_13B_rl_fin/"
# model = PeftModel.from_pretrained(model, peft_path)
# model = torch.compile(model)
# model = model.eval()

peft_path = "../../LLama2-13B-16bit-fin-r4_new"
model = PeftModel.from_pretrained(model, peft_path)
model = model.eval()


# %% [markdown]
# ### Load Datasets

# %%
dic = {
    0:"negative",
    1:'positive',
    2:'neutral',
}

def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}

def change_target(x):
    if 'positive' in x or 'Positive' in x:
        return 'positive'
    elif 'negative' in x or 'Negative' in x:
        return 'negative'
    else:
        return 'neutral'

def test_tfns(model, tokenizer, batch_size = 8, prompt_fun = None ):
    dataset = load_dataset('zeroshot/twitter-financial-news-sentiment')
    dataset = dataset['validation']
    dataset = dataset.to_pandas()
    dataset['label'] = dataset['label'].apply(lambda x:dic[x])
    
    if prompt_fun is None:
        dataset["instruction"] = 'What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.'
    else:
        dataset["instruction"] = dataset.apply(prompt_fun, axis = 1)

    dataset.columns = ['input', 'output', 'instruction']
    dataset[["context","target"]] = dataset.apply(format_example, axis = 1, result_type="expand")

    # print example
    print(f"\n\nPrompt example:\n{dataset['context'][0]}\n\n")

    context = dataset['context'].tolist()
    
    total_steps = dataset.shape[0]//batch_size + 1
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")


    out_text_list = []
    for i in tqdm(range(total_steps)):
        tmp_context = context[i* batch_size:(i+1)* batch_size]
        tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=512)
        # tokens.pop('token_type_ids')
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()
        res = model.generate(**tokens, max_length=512)
        res_sentences = [tokenizer.decode(i) for i in res]
        out_text = [o.split("Answer: ")[1] for o in res_sentences]
        out_text_list += out_text
        torch.cuda.empty_cache()

    dataset["out_text"] = out_text_list
    dataset["new_target"] = dataset["target"].apply(change_target)
    dataset["new_out"] = dataset["out_text"].apply(change_target)

    acc = accuracy_score(dataset["new_target"], dataset["new_out"])
    f1_macro = f1_score(dataset["new_target"], dataset["new_out"], average = "macro")
    f1_micro = f1_score(dataset["new_target"], dataset["new_out"], average = "micro")
    f1_weighted = f1_score(dataset["new_target"], dataset["new_out"], average = "weighted")

    print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted (BloombergGPT): {f1_weighted}. ")

    return dataset

# %% [markdown]
# Acc: 0.6641541038525963. F1 macro: 0.6596022406545043. F1 micro: 0.6641541038525963. F1 weighted (BloombergGPT): 0.6696344293081012. 
# Acc: 0.804857621440536. F1 macro: 0.7753384624395009. F1 micro: 0.804857621440536. F1 weighted (BloombergGPT): 0.8096782090540406.

#16bit-r8
# Acc: 0.6557788944723618. F1 macro: 0.26403641881638845. F1 micro: 0.6557788944723618. F1 weighted (BloombergGPT): 0.5194485324955582.

# TT r64 sc1 Acc: 0.8726968174204355. F1 macro: 0.8419524123472614. F1 micro: 0.8726968174204355. F1 weighted (BloombergGPT): 0.8727499660549283. 
# TT r64 sc2 Acc: 0.871859296482412. F1 macro: 0.8415316896728641. F1 micro: 0.871859296482412. F1 weighted (BloombergGPT): 0.8727817127912222.
# %%
dataset = test_tfns(model, tokenizer)

# %%
dataset["new_target"].hist()

# %%
dataset["new_out"].hist()


