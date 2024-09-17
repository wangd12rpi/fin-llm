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
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast, BitsAndBytesConfig
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

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

model = LlamaForCausalLM.from_pretrained(
    base_model, 
    trust_remote_code=True,
    cache_dir='/colab_space/yanglet/guoxuan/model_cache', 
    # load_in_8bit=True, 
    #load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map = "auto",
    # quantization_config = quantization_config
    )

# model = prepare_model_for_int8_training(model)

# peft_path = "../../finetuned_model_fin_CP/"
# peft_path = "../../finetuned_model_13B_rl_fin_oneGPU/"
# peft_path = "../../finetuned_model_13B_rl_fin/"
peft_path = "../../LLama2-13B-16bit-fin-r4_new"
model = PeftModel.from_pretrained(model, peft_path)
# model = torch.compile(model)
model = model.eval()

# %%
def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}

def add_instructions(x):
    if x.format == "post":
        return "What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}."
    else:
        return "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."

def make_label(x):
    if x < - 0.1: return "negative"
    elif x >=-0.1 and x < 0.1: return "neutral"
    elif x >= 0.1: return "positive"

def change_target(x):
    if 'positive' in x or 'Positive' in x:
        return 'positive'
    elif 'negative' in x or 'Negative' in x:
        return 'negative'
    else:
        return 'neutral'

def test_fiqa(model, tokenizer, batch_size = 16, prompt_fun = None ):
    dataset = load_dataset('pauri32/fiqa-2018')
    dataset = datasets.concatenate_datasets([dataset["train"], dataset["validation"] ,dataset["test"] ])
    dataset = dataset.train_test_split(0.226, seed = 42)['test']
    dataset = dataset.to_pandas()
    dataset["output"] = dataset.sentiment_score.apply(make_label)
    if prompt_fun is None:
        dataset["instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    else:
        dataset["instruction"] = dataset.apply(prompt_fun, axis = 1)

    dataset = dataset[['sentence', 'output',"instruction"]]
    dataset.columns = ["input", "output","instruction"]
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
# Acc: 0.8218181818181818. F1 macro: 0.6096758312504734. F1 micro: 0.8218181818181818. F1 weighted (BloombergGPT): 0.7997399764516329
# Acc: 0.7963636363636364. F1 macro: 0.6450434606663284. F1 micro: 0.7963636363636364. F1 weighted (BloombergGPT): 0.8011154631209293.

# TT R64 sc1 Acc: 0.8363636363636363. F1 macro: 0.7454562056157901. F1 micro: 0.8363636363636363. F1 weighted (BloombergGPT): 0.8519406872585571. 
# TT R64 sc2 Acc: 0.8654545454545455. F1 macro: 0.774264015931414. F1 micro: 0.8654545454545455. F1 weighted (BloombergGPT): 0.872907177421087. 

# TT R64 sc8 Acc: 0.850909090909091. F1 macro: 0.7750470345407053. F1 micro: 0.850909090909091. F1 weighted (BloombergGPT): 0.8660662850766417.

# LLama2-7B-16bit-fin-r4： Acc: 0.6290909090909091. F1 macro: 0.519479943201988. F1 micro: 0.6290909090909091. F1 weighted (BloombergGPT): 0.6857309516894181. 
# LLama2-7B-16bit-fin-r16 Acc: 0.6290909090909091. F1 macro: 0.519479943201988. F1 micro: 0.6290909090909091. F1 weighted (BloombergGPT): 0.6857309516894181. 
# LLama2-7B-16bit-fin-r8 ACC: 0.8218181818181818. F1 macro: 0.7182252388134741. F1 micro: 0.8218181818181818. F1 weighted (BloombergGPT): 0.8454320581379404. 
# %%
with torch.no_grad():
    instructions = test_fiqa(model, tokenizer, prompt_fun = add_instructions)

# %%


# %%


# %%



