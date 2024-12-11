import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score,f1_score
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm
import datasets
import torch
import pandas as pd
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path
from src.finetune import utils


def change_target(x):
    x = x.replace('"', '').replace('\n', '').replace(' ', '')
    return x

def evaluate_accuracy(out, target):
  correct_count = 0

  for x, y in zip(out, target):

    if x.startswith(y):  # Check if ground truth is included in LLM output
      correct_count += 1
    else:
      print(x.replace("\n", ""), "ACTUAL:" , y)

  accuracy = correct_count / len(out)
  return accuracy
    
def test_xbrl(args, model, tokenizer, path="../xbrl/xbrl_xbrl_tags_test.jsonl", prompt_fun=None):
    batch_size = 2
    
    instructions = pd.read_json(path_or_buf=path, lines=True)
    # instructions = instructions.head(10)
    # print(f"\n\nPrompt example:\n{instructions['context'][0]}\n\n")
    context = instructions['context'].tolist()
    
    total_steps = instructions.shape[0]//batch_size 
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")


    out_text_list = []
    for i in tqdm(range(total_steps)):
        tmp_context = context[i* batch_size: min(len(context), (i+1)* batch_size)]
        # tmp_context = [utils.add_xml(x, limit=80000) for x in tmp_context]
        tmp_target = instructions['target'].tolist()[i* batch_size: min(len(context), (i+1)* batch_size)]
        
        if len(tmp_context) == 0:
            continue
        tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=512, return_token_type_ids=False)
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()
        res = model.generate(**tokens, max_new_tokens=20, eos_token_id=tokenizer.eos_token_id)
        res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
        out_text = [o.split("\nAnswer:")[1].strip() for o in res_sentences]
        # print(f'llm: {out_text[0]}, actual: {tmp_target[0]}')

        out_text_list += out_text
        torch.cuda.empty_cache()

    
    instructions["target"] = instructions["target"]
    # instructions["new_target"] = instructions["target"].apply(change_target)
    # instructions["new_out"] = instructions["out_text"].apply(change_target)
    target_list = instructions["target"].tolist()
    target_list = [str(x) for x in target_list]

    acc = evaluate_accuracy(out_text_list, target_list)
    # f1_macro = f1_score(instructions["new_target"], instructions["new_out"], average = "macro")
    # f1_micro = f1_score(instructions["new_target"], instructions["new_out"], average = "micro")
    # f1_weighted = f1_score(instructions["new_target"], instructions["new_out"], average = "weighted")

    print(f"XBRL Acc: {acc}. ")

    return {"acc": acc, "f1": 0.0}

