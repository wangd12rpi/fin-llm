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
from batch_inference import perform_batch_inference_with_metrics

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
    
def test_fpb(args, model, tokenizer, prompt_fun=None):
    batch_size = args.batch_size
    instructions = load_dataset("financial_phrasebank", "sentences_50agree")
    instructions = instructions["train"].train_test_split(seed=42)['test']
    instructions = instructions.to_pandas()
    instructions.columns = ["input", "output"]
    instructions["output"] = instructions["output"].apply(lambda x: dic[x])

    if prompt_fun is None:
        instructions["instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    else:
        instructions["instruction"] = instructions.apply(prompt_fun, axis=1)

    instructions[["context", "target"]] = instructions.apply(format_example, axis=1, result_type="expand")

    # print example
    print(f"\n\nPrompt example:\n{instructions['context'][0]}\n\n")

    context = instructions['context'].tolist()

    dataset = instructions.copy()  # Assuming instructions is a pandas DataFrame

    #perform batch inference using the refactored function
    dataset, acc, f1_macro, f1_micro, f1_weighted, batch_times, total_execution_time, gpu_memory_usage = perform_batch_inference_with_metrics(
        context, dataset, batch_size, tokenizer, model, change_target
    )

    return instructions


def test_fpb_mlt(args, model, tokenizer):
    batch_size = args.batch_size
    # dataset = load_dataset("financial_phrasebank", "sentences_50agree")
    dataset = load_from_disk(Path(__file__).parent.parent / 'data/financial_phrasebank-sentences_50agree/')
    dataset = dataset["train"]#.select(range(300))
    dataset = dataset.train_test_split(seed=42)['test']
    dataset = dataset.to_pandas()
    dataset.columns = ["input", "output"]
    dataset["output"] = dataset["output"].apply(lambda x: dic[x])
    dataset["text_type"] = dataset.apply(lambda x: 'news', axis=1)
    
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
            res = model.generate(**inputs, do_sample=False, max_length=args.max_length, eos_token_id=tokenizer.eos_token_id, max_new_tokens=10)
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