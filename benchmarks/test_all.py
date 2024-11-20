from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, get_peft_model, LoraConfig, TaskType  # 0.4.0
import torch
import argparse
import re

from fpb import test_fpb, test_fpb_mlt
from fiqa import test_fiqa, test_fiqa_mlt
from tfns import test_tfns
from nwgi import test_nwgi
from headline import test_headline
from ner import test_ner
from xbrl import test_xbrl
# from convfinqa import test_convfinqa
# from fineval import test_fineval
# from finred import test_re

import sys

sys.path.append('../')
from utils import *


def main(args):
    model_name = args.base_model

    bnb_config = BitsAndBytesConfig(
        # load_in_4bit=args.quant_bits == 4,  # Load in 4-bit if quant_bits is 4
        # load_in_8bit=args.quant_bits == 8,  # Load in 8-bit if quant_bits is 8
        load_in_8bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,

    )

    model.model_parallel = True

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              torch_dtype=torch.bfloat16,
                                              trust_remote_code=True,
                                              device_map="auto"
                                              )

    # tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # print(f'pad: {tokenizer.pad_token_id}, eos: {tokenizer.eos_token_id}')

    if args.peft_model != "":
        model = PeftModel.from_pretrained(model, args.peft_model)
    else:
        model.half()

    model = model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        for data in args.dataset.split(','):
            if data == 'fpb':
                results[(args.base_model, args.quant_bits, args.rank, data)] = test_fpb(args, model, tokenizer)
            elif data == 'fiqa':
                results[(args.base_model, args.quant_bits, args.rank, data)] = test_fiqa(args, model, tokenizer)
            elif data == 'tfns':
                results[(args.base_model, args.quant_bits, args.rank, data)] = test_tfns(args, model, tokenizer)
            elif data == 'nwgi':
                results[(args.base_model, args.quant_bits, args.rank, data)] = test_nwgi(args, model, tokenizer)
            elif data == 'headline':
                results[(args.base_model, args.quant_bits, args.rank, data)] = test_headline(args, model, tokenizer)
            elif data == 'ner':
                results[(args.base_model, args.quant_bits, args.rank, data)] = test_ner(args, model, tokenizer)
            elif data == 'xbrl_tags':
                results[(args.base_model, args.quant_bits, args.rank, data)] = test_xbrl(args, model, tokenizer)
            elif data == 'xbrl_value':
                results[(args.base_model, args.quant_bits, args.rank, data)] = test_xbrl(args, model, tokenizer, path="../xbrl/xbrl_value_test.jsonl")
            else:
                raise ValueError('undefined dataset.')

    print("\n*********\nAfter eval:", torch.cuda.max_memory_allocated() // 1024 // 1024 // 1024)

    print('Evaluation Ends.')


def parse_folder_name(folder_name):
  """
  Parses a folder name to extract information about the dataset, model, quantization bits, and rank.
  Assumes no hyphens in the JSONL filename.

  Args:
    folder_name: The folder name in the format 
                 "dataset_name-model_name-quant_bits-r{rank}".

  Returns:
    A tuple containing the dataset name, model name, quantization bits, and rank.
  """
  pattern = r"^(.+?)-(.+?)-(\d+)bits-r(\d+)$"
  match = re.match(pattern, folder_name)
  if match:
    dataset_name = match.group(1)
    model_name = match.group(2)
    quant_bits = int(match.group(3))
    rank = int(match.group(4))
    return dataset_name, model_name, quant_bits, rank
  else:
    return None


def map_dataset_name(jsonl_name):
  if "sentiment" in jsonl_name:
    return "fiqa,fpb,tfns,nwgi"
  elif "headline" in jsonl_name:
    return "headline"
  elif "ner" in jsonl_name:
    return "ner"
  elif "xbrl_tags" in jsonl_name:
    return "xbrl_tags"
  elif "xbrl_value" in jsonl_name:
    return "xbrl_value"
  elif "xbrl_train" in jsonl_name:
    return "xbrl_value,xbrl_tags"
  else:
    return None  


def get_batch_size(model_name):
  if "70b" in model_name.lower():
    return 8
  else:
    return 128


def generate_markdown_tables(results):
    """
    Generates markdown tables for accuracy and F1 scores with datasets in columns.

    Args:
        results: A dictionary containing the accuracy and F1 scores for each dataset,
                 with (model, bits, r, dataset) as keys.
    """

    # Get unique models, bits, ranks, and datasets
    models = sorted(set(model for model, _, _, _ in results.keys()))
    bits = sorted(set(bits for _, bits, _, _ in results.keys()))
    ranks = sorted(set(rank for _, _, rank, _ in results.keys()))
    datasets = sorted(set(dataset for _, _, _, dataset in results.keys()))

    with open("evaluation_results.md", "w") as f:
        f.write("# Evaluation Results\n\n")

        # Accuracy table
        f.write("## Accuracy\n\n")
        f.write("| Model | " + " | ".join(datasets) + " |\n")
        f.write("|------| " + " | ".join(["---"] * len(datasets)) + " |\n")
        for model in models:
            for bit in bits:
                for rank in ranks:        
                    if (bit == -1) != (rank == -1):
                        continue
                        
                    row = f"| {model}-{bit}bits-r{rank} |"
                    for dataset in datasets:
                        key = (model, bit, rank, dataset)
                        if key in results:
                            row += f" {results[key]['acc']:.4f} |"
                        else:
                            row += " - |"
                    
                    f.write(row + "\n")

        f.write("\n\n")  # Add an empty line between tables

        # F1 score table
        f.write("## F1 Score\n\n")
        f.write("| Model | " + " | ".join(datasets) + " |\n")
        f.write("|------| " + " | ".join(["---"] * len(datasets)) + " |\n")
        for model in models:
            for bit in bits:
                for rank in ranks:
                    if (bit == -1) != (rank == -1):
                        continue
                        
                    row = f"| {model}-{bit}bits-r{rank}|"
                    for dataset in datasets:
                        key = (model, bit, rank, dataset)
                        if key in results:
                            row += f" {results[key]['f1']:.4f} |"
                        else:
                            row += " - |"
                    f.write(row + "\n")



if __name__ == "__main__":
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    results = {} 
    finetuned_models_dir = "../finetuned_models"
    # dataset_base = "fiqa,fpb,tfns,nwgi,headline,ner,xbrl_tags,xbrl_value"
    dataset_base = "xbrl_tags,xbrl_value"
    base_models = [
        # "meta-llama/Llama-3.1-70B-Instruct", 
        # "meta-llama/Llama-3.1-8B-Instruct"
    ]
    
    for model_name in base_models:
        args = {
                    "base_model": model_name,
                    "quant_bits": -1,
                    "peft_model": "",
                    "dataset": dataset_base,  
                    "batch_size": get_batch_size(model_name),  
                    "rank": -1
                }
        
        print(f"Running main() for folder: {args}")
        args = argparse.Namespace(**args) 
        main(args)


    
    for folder_name in os.listdir(finetuned_models_dir):
        parsed_info = parse_folder_name(folder_name)
        if parsed_info:
            dataset_name, model_name, quant_bits, rank = parsed_info
            dataset_name = map_dataset_name(dataset_name)

            if dataset_name is None:
                print(f"Skipping folder with unknown dataset: {folder_name}")
                continue

            if "xbrl" not in dataset_name:
                continue
                
            # Construct the full path to the fine-tuned model
            peft_model_path = os.path.join(finetuned_models_dir, folder_name)

            # Determine the batch size based on the model name
            batch_size = get_batch_size(model_name)

            # Set up the arguments for the main function
            args = {
                "base_model": "meta-llama/" + model_name.replace("meta-llama-", ""),
                "quant_bits": quant_bits,
                "peft_model": peft_model_path,
                "dataset": dataset_name,  
                "batch_size": batch_size,  
                "rank": rank
            }
            
            print(f"Running main() for folder: {args}")
            args = argparse.Namespace(**args) 
            main(args)
        else:
            print(f"Skipping invalid folder name: {folder_name}")

    generate_markdown_tables(results=results)

    
