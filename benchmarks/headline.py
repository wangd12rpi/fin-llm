from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import datasets
import torch
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path
from batch_inference import perform_batch_inference_with_metrics
from formatPrompt import format_example

from changeTarget import change_target

import sys
sys.path.append("../")

    
def test_headline(args, model, tokenizer, prompt_fun=None):
    batch_size = args.batch_size
    dataset = load_dataset("FinGPT/fingpt-headline-cls")
    # instructions = load_from_disk(Path(__file__).parent.parent / "data/financial_phrasebank-sentences_50agree/")
    dataset = dataset["test"]    
    # print example
    dataset = dataset.to_pandas()

    dataset[["context","target"]] = dataset.apply(format_example, axis = 1, result_type="expand")

    print(f"\n\nPrompt example:\n{dataset['context'][0]}\n\n")
    context = dataset['context'].tolist()

    dataset, acc, f1_macro, f1_micro, f1_weighted, batch_times, total_execution_time, gpu_memory_usage = perform_batch_inference_with_metrics(
        context, dataset, batch_size, tokenizer, model, change_target
    )

    return dataset

