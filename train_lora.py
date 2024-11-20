import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer, AutoModel, AutoConfig# Model,Tokenizer
from transformers import DataCollatorForLanguageModeling  # Datacollator
from transformers import TrainingArguments, Trainer
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast, DataCollatorForSeq2Seq, \
    MistralForCausalLM
from transformers import BitsAndBytesConfig
from deepspeed.pipe import PipelineModule
from torch.utils.tensorboard import SummaryWriter
import datasets
import torch
import pdb
import argparse
from datetime import datetime
from functools import partial
from tqdm import tqdm
import json
import wandb
import benchmarks.utils as utils

def bytes_to_giga_bytes(bytes):
  return bytes / 1024 / 1024 / 1024
    
# Trainer
class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def prediction_step(self, model: torch.nn.Module, inputs, prediction_loss_only: bool, ignore_keys=None):
        with torch.no_grad():
            res = model(
                input_ids=inputs["input_ids"].to(model.device),
                labels=inputs["labels"].to(model.device),
            ).loss
        return (res, None, None)

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))

class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def get_data(args):
    def preprocess(example, max_seq_length):
        prompt = example["context"]
        target = example["target"]
        
        prompt_ids = tokenizer.encode(prompt)
        target_ids = tokenizer.encode(
            target, add_special_tokens=False)
        input_ids = prompt_ids + target_ids + [config.eos_token_id[0]]
        # print(input_ids, "\n\n")
        return {"input_ids": input_ids, "seq_len": len(prompt_ids)}
    
    def load_dataset_jsonl(name):
        with open(name, "r") as f:
            for line in tqdm(f.readlines()):
                example = json.loads(line)
                feature = preprocess(example, args.max_length)
                # feature["input_ids"] = feature["input_ids"]
                yield feature
        
    # dataset_list = load_dataset(args.dataset, args.from_remote)
    # dataset_train = datasets.concatenate_datasets([d['train'] for d in dataset_list]).shuffle(seed=42)

    # # if args.test_dataset:
    # #     dataset_list = load_dataset(args.test_dataset, args.from_remote)
    # dataset_test = datasets.concatenate_datasets([d['test'] for d in dataset_list])

    # dataset = datasets.DatasetDict({'train': dataset_train, 'test': dataset_test})
    # # Display first sample from the training dataset
    # print(dataset['train'][0])
    # # Filter out samples that exceed the maximum token length and remove unused columns
    # dataset = dataset.map(partial(tokenize, args, tokenizer))
    # print('original dataset length: ', len(dataset['train']))
    # dataset = dataset.filter(lambda x: not x['exceed_max_length'])
    # print('filtered dataset length: ', len(dataset['train']))
    # dataset = dataset.remove_columns(['instruction', 'input', 'output', 'exceed_max_length'])
    # print(dataset['train'][0])
    # return dataset
    # return load_jsonl_dataset(args.dataset, tokenizer)
    
    dataset = datasets.Dataset.from_generator(
        lambda: load_dataset_jsonl(args.dataset), num_proc = 64
    )

    dataset = dataset.train_test_split(test_size=0.05)
    return dataset
    

# def data_collator(features: list) -> dict:
#     len_ids = [len(feature["input_ids"]) for feature in features]
#     longest = max(len_ids)
#     input_ids = []
#     labels_list = []
#     for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
#         ids = feature["input_ids"]
#         seq_len = feature["seq_len"]
#         labels = (
#                 [tokenizer.pad_token_id] * (seq_len - 1) + ids[(seq_len - 1):] + [tokenizer.pad_token_id] * (
#                     longest - ids_l)
#         )
#         ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
#         _ids = torch.LongTensor(ids)
#         labels_list.append(torch.LongTensor(labels))
#         input_ids.append(_ids)
#     input_ids = torch.stack(input_ids)
#     labels = torch.stack(labels_list)
#     return {
#         "input_ids": input_ids,
#         "labels": labels,
#     }

def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"] # prompt length
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

def main(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    # LoRA
    from peft import (
        TaskType,
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        prepare_model_for_kbit_training,
        set_peft_model_state_dict,
        PeftModel
    )

    # Model,Tokenizer, Datacollator
    model_name = args.base_model

    # load data
    # dataset = datasets.load_from_disk("./data/dataset_new")

    dataset = get_data(args)

    # Create a timestamp for model saving
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%dT%H%M')

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # world_size = 1
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        # gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # config
    deepspeed_config = args.ds_config
    # import deepspeed
    # deepspeed.init_distributed(dist_backend = "gloo")

    dataset_name = args.dataset.split('/')[-1]
    task_name = f"{dataset_name}-{args.base_model.replace('meta-llama-', '')}-{args.quant_bits}bits-r{args.r}".replace("/", "-")
    
    training_args = TrainingArguments(
        output_dir='./finetuned_models/' + "/" + task_name,

        logging_steps=0.1,
        save_steps=args.eval_steps,
        warmup_ratio=args.warmup_ratio,

        max_steps=args.max_steps,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accu,
        learning_rate=args.learning_rate,
        weight_decay=0.005,

        # ddp_backend = 'gloo',
        # bf16=True,
        fp16=True,
        deepspeed=deepspeed_config,
        torch_compile=False,
        load_best_model_at_end=False,
        evaluation_strategy="steps",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if ddp else None,
        # testing only, comment otherwise
        dataloader_num_workers=64,
        dataloader_pin_memory=True,
        report_to='wandb',
        run_name=task_name,
        max_grad_norm=1.0
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.quant_bits == 4,  # Load in 4-bit if quant_bits is 4
        load_in_8bit=args.quant_bits == 8,  # Load in 8-bit if quant_bits is 8
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        # attn_implementation="flash_attention_2"
    )

    model = prepare_model_for_kbit_training(model)

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.r,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q_proj', "k_proj", 'v_proj'],
        bias='none',
    )

    model = get_peft_model(model, peft_config)
        
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True

    # KVcache inference
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    # model.config.use_cache = False
    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    # Train
    writer = SummaryWriter()
    trainer = ModifiedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        callbacks=[TensorBoardCallback(writer)],
    )
    print("\n*********\nBefore training:", bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

    trainer.train()
    writer.close()
    # save model
    model.save_pretrained(training_args.output_dir)
    print("\n*********\nAfter training:", bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

    


if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--base_model", required=True, type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--batch_size", default=4, type=int, help="The train batch size per device")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The learning rate")
    parser.add_argument("--num_epochs", default=8, type=float, help="The training epochs")
    parser.add_argument("--gradient_steps", default=8, type=float, help="The gradient accumulation steps")
    parser.add_argument("--num_workers", default=8, type=int, help="dataloader workers")
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--warmup_ratio", default=0.05, type=float)
    parser.add_argument("--ds_config", default='./config_.json', type=str)
    parser.add_argument("--scheduler", default='linear', type=str)
    parser.add_argument("--instruct_template", default='default')
    parser.add_argument("--evaluation_strategy", default='steps', type=str)
    parser.add_argument("--load_best_model", default='False', type=bool)
    parser.add_argument("--eval_steps", default=0.1, type=float)
    parser.add_argument("--from_remote", default=False, type=bool)
    parser.add_argument("--grad_accu", default=1, type=int)
    parser.add_argument("--quant_bits", default=8, type=int)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--peft_model", default="", type=str)
    parser.add_argument("--r", default=8, type=int)

    args = parser.parse_args()

    # Login to Weights and Biases
    model_name = args.base_model

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # Run the main function
    main(args)

    run = wandb.init(
        project="fin_finetune_results",
        tags=[args.base_model, args.dataset],
    )

    wandb.config = args
    
