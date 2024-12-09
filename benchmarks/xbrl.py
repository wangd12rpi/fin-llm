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
import utils

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
    
    # Load instructions from the file
    instructions = pd.read_json(path_or_buf=path, lines=True)
    context = instructions['context'].tolist()

    # Call the batch inference function
    dataset, acc, f1_macro, f1_micro, f1_weighted, batch_times, total_execution_time, gpu_memory_usage = \
        perform_batch_inference_with_metrics(
            context=context,
            dataset=instructions,
            batch_size=batch_size,
            tokenizer=tokenizer,
            model=model,
            change_target=change_target
        )

    return dataset