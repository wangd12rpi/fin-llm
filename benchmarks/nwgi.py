from sklearn.metrics import accuracy_score,f1_score
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import datasets
import torch
from pathlib import Path
from batch_inference import perform_batch_inference_with_metrics

import pandas as pd
from formatPrompt import format_example


dic = {
    'strong negative':"negative",
    'moderately negative':"negative",
    'mildly negative':"neutral",
    'strong positive':"positive",
    'moderately positive':"positive",
    'mildly positive':'neutral',
    'neutral':'neutral',
}

def change_target(x):
    if 'positive' in x or 'Positive' in x:
        return 'positive'
    elif 'negative' in x or 'Negative' in x:
        return 'negative'
    else:
        return 'neutral'

def test_nwgi(args, model, tokenizer, prompt_fun=None):
    batch_size = args.batch_size
    dataset = load_dataset('oliverwang15/news_with_gpt_instructions')['test']
    dataset = pd.DataFrame(dataset)

    dataset['output'] = dataset['label'].apply(lambda x: x)

    if prompt_fun is None:
        dataset["instruction"] = "What is the sentiment of this news? Please choose an answer from {strong negative/moderately negative/mildly negative/neutral/mildly positive/moderately positive/strong positive}."
    else:
        dataset["instruction"] = dataset.apply(prompt_fun, axis=1)
    dataset["input"] = dataset["news"].apply(lambda x: x)

    dataset = dataset[['input', 'output', 'instruction']]
    dataset[["context", "target"]] = dataset.apply(format_example, axis=1, result_type="expand")

    # print example
    print(f"\n\nPrompt example:\n{dataset['context'][0]}\n\n")

    context = dataset['context'].tolist()

    # perform batch inference and calculate metrics
    dataset, acc, f1_macro, f1_micro, f1_weighted, batch_times, total_execution_time, gpu_memory_usage = perform_batch_inference_with_metrics(
        context, dataset, batch_size, tokenizer, model, change_target
    )

    return dataset
