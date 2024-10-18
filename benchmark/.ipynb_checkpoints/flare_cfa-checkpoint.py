from sklearn.metrics import accuracy_score,f1_score
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
import pandas as pd
import datasets
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    LlamaForCausalLM
)

def test_flare(args, model, tokenizer, prompt_fun = None ):
    batch_size = args.batch_size
    
    dataset = load_dataset('TheFinAI/flare-causal20-sc')
    dataset = dataset['test']
    dataset = dataset.to_pandas()
    #dataset = dataset.head(50)

    # print example
    print(f"\n\nPrompt example:\n{dataset['text'][0]}\n\n")

    context = dataset['text'].tolist()
    context = [f"""
    Instruction: In this task, you are provided with sentences extracted 
    from financial news and SEC data. Your goal is to classify each sentence 
    into either 'causal' or 'noise' based on whether or not it indicates a causal 
    relationship between financial events. Please return only the category
    causal or noise. I want my answer to be in lower case. \nInput: {x}\nAnswer:""" for x in context]
    total_steps = dataset.shape[0] // batch_size + 1
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")


    out_text_list = []
    for i in tqdm(range(total_steps)):
        tmp_context = context[i* batch_size: (i+1)* batch_size]
        tokens = tokenizer(tmp_context, return_tensors='pt', padding=True)
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()
        res = model.generate(**tokens, max_new_tokens = 10)
        res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
        print(res_sentences[0])
        out_text = [o.split("Answer:")[1] for o in res_sentences]
        print(out_text)
        out_text_list += out_text
        torch.cuda.empty_cache()

    dataset["out_text"] = out_text_list
    # dataset["new_target"] = dataset["target"].apply(change_target)
    # dataset["new_out"] = dataset["out_text"].apply(change_target)
    

    acc = accuracy_score(dataset["answer"], dataset["out_text"])
    f1_macro = f1_score(dataset["answer"], dataset["out_text"], average = "macro")
    f1_micro = f1_score(dataset["answer"], dataset["out_text"], average = "micro")
    f1_weighted = f1_score(dataset["answer"], dataset["out_text"], average = "weighted")

    print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted (BloombergGPT): {f1_weighted}. ")
    print(f"Acc: {acc}. ")

    return dataset

if __name__ == "__main__":
    test_flare()