import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score,f1_score
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm
import datasets
import torch

from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path

dic = {
        0:"negative",
        1:'neutral',
        2:'positive',
    }

with open(Path(__file__).parent / 'sentiment_templates.txt') as f:
    templates = [l.strip() for l in f.readlines()]
    

def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}

def change_target(x):
    if 'organization' in x.lower():
        return 'organization'
    elif 'person' in x.lower():
        return 'negative'
    elif 'location' in x.lower():
        return 'location'
    return ''
    
def test_ner(args, model, tokenizer, prompt_fun=None):
    batch_size = args.batch_size
    instructions = load_dataset("FinGPT/fingpt-ner-cls")
    # instructions = load_from_disk(Path(__file__).parent.parent / "data/financial_phrasebank-sentences_50agree/")
    instructions = instructions["test"]    
    # print example
    instructions = instructions.to_pandas()

    instructions[["context","target"]] = instructions.apply(format_example, axis = 1, result_type="expand")

    print(f"\n\nPrompt example:\n{instructions['context'][0]}\n\n")
    context = instructions['context'].tolist()
    
    total_steps = instructions.shape[0]//batch_size + 1
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")


    out_text_list = []
    for i in tqdm(range(total_steps)):
        tmp_context = context[i* batch_size: min(len(context), (i+1)* batch_size)]
        if len(tmp_context) == 0:
            continue
        tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=512, return_token_type_ids=False)
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()
        res = model.generate(**tokens, max_new_tokens=20, eos_token_id=tokenizer.eos_token_id)
        res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
        # print(f'{i}: {res_sentences[0]}')
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

    print(f"FPB: Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted (BloombergGPT): {f1_weighted}. ")

    return {"acc": acc, "f1": f1_weighted}

