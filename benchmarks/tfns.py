import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score,f1_score
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import datasets
import torch
from pathlib import Path

from batch_inference import perform_batch_inference_with_metrics
from formatPrompt import format_example
from changeTarget import change_target

dic = {
    0:"negative",
    1:'positive',
    2:'neutral',
}

def test_tfns(args, model, tokenizer, prompt_fun=None):
    batch_size = args.batch_size
    dataset = load_dataset('zeroshot/twitter-financial-news-sentiment')
    # dataset = load_from_disk(Path(__file__).parent.parent / 'data/twitter-financial-news-sentiment')
    dataset = dataset['validation']
    dataset = dataset.to_pandas()
    dataset['label'] = dataset['label'].apply(lambda x:dic[x])
    
    if prompt_fun is None:
        dataset["instruction"] = 'What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.'
    else:
        dataset["instruction"] = dataset.apply(prompt_fun, axis = 1)

    dataset.columns = ['input', 'output', 'instruction']
    dataset[["context","target"]] = dataset.apply(format_example, axis = 1, result_type="expand")

    # print example
    print(f"\n\nPrompt example:\n{dataset['context'][0]}\n\n")

    context = dataset['context'].tolist()

    # perform batch inference and calculate metrics
    dataset, acc, f1_macro, f1_micro, f1_weighted, batch_times, total_execution_time, gpu_memory_usage = perform_batch_inference_with_metrics(
        context, dataset, batch_size, tokenizer, model, change_target
    )

    return dataset