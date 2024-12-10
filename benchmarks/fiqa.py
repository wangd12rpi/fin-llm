from sklearn.metrics import accuracy_score,f1_score
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm
import datasets
import torch

from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path

from batch_inference import perform_batch_inference_with_metrics
from formatPrompt import format_example


with open(Path(__file__).parent / 'sentiment_templates.txt') as f:
    templates = [l.strip() for l in f.readlines()]
    

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
    
def vote_output(x):
    output_dict = {'positive': 0, 'negative': 0, 'neutral': 0} 
    for i in range(len(templates)):
        pred = change_target(x[f'out_text_{i}'].lower())
        output_dict[pred] += 1
    if output_dict['positive'] > output_dict['negative']:
        return 'positive'
    elif output_dict['negative'] > output_dict['positive']:
        return 'negative'
    else:
        return 'neutral'
    

def test_fiqa(args, model, tokenizer, prompt_fun=add_instructions):
    batch_size = args.batch_size
    dataset = load_dataset('pauri32/fiqa-2018')
    #dataset = load_from_disk(Path(__file__).parent.parent / 'data/fiqa-2018/')
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
    dataset[["context","target"]] = dataset.apply(format_example, axis=1, result_type="expand")

    # print example
    print(f"\n\nPrompt example:\n{dataset['context'][0]}\n\n")

    context = dataset['context'].tolist()
    
    # perform batch inference and calculate metrics
    dataset, acc, f1_macro, f1_micro, f1_weighted, batch_times, total_execution_time, gpu_memory_usage = perform_batch_inference_with_metrics(
        context, dataset, batch_size, tokenizer, model, change_target
    )
   
    return dataset


def test_fiqa_mlt(args, model, tokenizer):
    batch_size = args.batch_size
    # dataset = load_dataset('pauri32/fiqa-2018')
    dataset = load_from_disk(Path(__file__).parent.parent / 'data/fiqa-2018/')
    dataset = datasets.concatenate_datasets([dataset["train"], dataset["validation"] ,dataset["test"] ])
    dataset = dataset.train_test_split(0.226, seed=42)['test']
    dataset = dataset.to_pandas()
    dataset["output"] = dataset.sentiment_score.apply(make_label)
    dataset["text_type"] = dataset.apply(lambda x: 'tweet' if x.format == "post" else 'news', axis=1)
    dataset = dataset[['sentence', 'output', "text_type"]]
    dataset.columns = ["input", "output", "text_type"]
    
    dataset["output"] = dataset["output"].apply(change_target)
    dataset = dataset[dataset["output"] != 'neutral']

    out_texts_list = [[] for _ in range(len(templates))]
    
    def collate_fn(batch):
        inputs = tokenizer(
            [f["context"] for f in batch], return_tensors='pt',
            padding=True, max_length=args.max_length,
            return_token_type_ids=False
        )
        return inputs
    
    for i, template in enumerate(templates):
        dataset = dataset[['input', 'output', "text_type"]]
        dataset["instruction"] = dataset['text_type'].apply(lambda x: template.format(type=x) + "\nOptions: positive, negative")
        # dataset["instruction"] = dataset['text_type'].apply(lambda x: template.format(type=x) + "\nOptions: negative, positive")
        dataset[["context", "target"]] = dataset.apply(format_example, axis=1, result_type="expand")
        
        dataloader = DataLoader(Dataset.from_pandas(dataset), batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

        log_interval = len(dataloader) // 5

        for idx, inputs in enumerate(tqdm(dataloader)):
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            res = model.generate(**inputs, do_sample=False, max_length=args.max_length, eos_token_id=tokenizer.eos_token_id)#, max_new_tokens=10)
            res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
            tqdm.write(f'{idx}: {res_sentences[0]}')
            # if (idx + 1) % log_interval == 0:
            #     tqdm.write(f'{idx}: {res_sentences[0]}')
            out_text = [o.split("Answer: ")[1] for o in res_sentences]
            out_texts_list[i] += out_text
            torch.cuda.empty_cache()

    for i in range(len(templates)):
        dataset[f"out_text_{i}"] = out_texts_list[i]
        dataset[f"out_text_{i}"] = dataset[f"out_text_{i}"].apply(change_target)
    
    dataset["new_out"] = dataset.apply(vote_output, axis=1, result_type="expand")

    dataset.to_csv('tmp.csv')
    
    for k in [f"out_text_{i}" for i in range(len(templates))] + ["new_out"]:

        acc = accuracy_score(dataset["target"], dataset[k])
        f1_macro = f1_score(dataset["target"], dataset[k], average="macro")
        f1_micro = f1_score(dataset["target"], dataset[k], average="micro")
        f1_weighted = f1_score(dataset["target"], dataset[k], average="weighted")

        print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted (BloombergGPT): {f1_weighted}. ")

    return dataset