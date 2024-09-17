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
    # load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map = "auto",
    # quantization_config = quantization_config
    )

# model = prepare_model_for_int8_training(model)

# peft_path = "../../finetuned_model_fin_CP/"
# peft_path = "../../finetuned_model_13B_rl_fin_oneGPU/"
# peft_path = "../../finetuned_model_13B_rl_fin/"
# peft_path = "../../LLama2-13B-16bit-fin-r4_new"
peft_path = "../../yz_test"
model = PeftModel.from_pretrained(model, peft_path)
# model = torch.compile(model)
model = model.eval()

# %%
dic = {
        0:"negative",
        1:'neutral',
        2:'positive',
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

def test_fpb(model, tokenizer, batch_size = 8, prompt_fun = None ):
    instructions = load_dataset("financial_phrasebank", "sentences_50agree")
    instructions = instructions["train"]
    instructions = instructions.train_test_split(seed = 42)['test']
    instructions = instructions.to_pandas()
    instructions.columns = ["input", "output"]
    instructions["output"] = instructions["output"].apply(lambda x:dic[x])

    if prompt_fun is None:
        instructions["instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    else:
        instructions["instruction"] = instructions.apply(prompt_fun, axis = 1)
    
    instructions[["context","target"]] = instructions.apply(format_example, axis = 1, result_type="expand")

    # print example
    print(f"\n\nPrompt example:\n{instructions['context'][0]}\n\n")


    context = instructions['context'].tolist()
    
    total_steps = instructions.shape[0]//batch_size + 1
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")


    out_text_list = []
    for i in tqdm(range(total_steps)):
        tmp_context = context[i* batch_size:(i+1)* batch_size]
        tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=512)
        # tokens.pop("token_type_ids")
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()
        res = model.generate(**tokens, max_length=512)
        res_sentences = [tokenizer.decode(i) for i in res]
        print(tmp_context)
        print(res_sentences)
        out_text = [o.split("Answer: ")[1] for o in res_sentences]
        out_text_list += out_text
        torch.cuda.empty_cache()

    instructions["out_text"] = out_text_list
    instructions["new_target"] = instructions["target"].apply(change_target)
    instructions["new_out"] = instructions["out_text"].apply(change_target)

    acc = accuracy_score(instructions["new_target"], instructions["new_out"])
    f1_macro = f1_score(instructions["new_target"], instructions["new_out"], average = "macro")
    f1_micro = f1_score(instructions["new_target"], instructions["new_out"], average = "micro")
    f1_weighted = f1_score(instructions["new_target"], instructions["new_out"], average = "weighted")

    print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted (BloombergGPT): {f1_weighted}. ")

    return instructions

# %%
instructions = test_fpb(model, tokenizer)


# %%
#CP : Acc: 0.7533003300330033. F1 macro: 0.7400709662722224. F1 micro: 0.7533003300330033. F1 weighted (BloombergGPT): 0.7521990297725961. 
#TT : Acc: 0.8052805280528053. F1 macro: 0.7965776742596127. F1 micro: 0.8052805280528051. F1 weighted (BloombergGPT): 0.8040653695092289. 

# TT r64 sc1 ACC: Acc: 0.8465346534653465. F1 macro: 0.828770692223959. F1 micro: 0.8465346534653466. F1 weighted (BloombergGPT): 0.8411825435779249. 
# TT r64 sc2 Acc: 0.8415841584158416. F1 macro: 0.8301708293075087. F1 micro: 0.8415841584158416. F1 weighted (BloombergGPT): 0.8391890406015141. 

# TT r64 sc8 4Acc: 0.8424092409240924. F1 macro: 0.8324490083031915. F1 micro: 0.8424092409240924. F1 weighted (BloombergGPT): 0.8391524376380686. 
# LR r260 Acc: 0.8655115511551155. F1 macro: 0.8570817206331226. F1 micro: 0.8655115511551155. F1 weighted (BloombergGPT): 0.8642192464313929.

# CP : Acc: 0.7904290429042904. F1 macro: 0.7782702091174217. F1 micro: 0.7904290429042904. F1 weighted (BloombergGPT): 0.788020742089911.

# LLama2-7B-16bit-fin-r8:
#Acc: 0.6014851485148515. F1 macro: 0.250386398763524. F1 micro: 0.6014851485148515. F1 weighted (BloombergGPT): 0.45181110073913117.