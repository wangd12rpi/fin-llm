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
from formatPrompt import format_example


dic = {
        0:"negative",
        1:'neutral',
        2:'positive',
    }

with open(Path(__file__).parent / 'sentiment_templates.txt') as f:
    templates = [l.strip() for l in f.readlines()]
    

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
    dataset = instructions.copy()  
    
    #perform batch inference using the refactored function
    dataset, acc, f1_macro, f1_micro, f1_weighted, batch_times, total_execution_time, gpu_memory_usage = perform_batch_inference_with_metrics(
        context, dataset, batch_size, tokenizer, model, change_target
    )

    return {"acc": acc, "f1": f1_weighted}

