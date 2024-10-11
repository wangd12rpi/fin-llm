import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer, AutoModel  # Model,Tokenizer
from transformers import DataCollatorForLanguageModeling  # Datacollator
from transformers import TrainingArguments, Trainer
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizerFast
from transformers import BitsAndBytesConfig
from deepspeed.pipe import PipelineModule
from torch.utils.tensorboard import SummaryWriter
import datasets
import torch
import pdb

local_rank = int(os.environ["LOCAL_RANK"])
# LoRA
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

# Model,Tokenizer, Datacollator
model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = '/colab_space/yanglet/models--daryl149--Llama-2-7b-chat-hf/snapshots/bbc9b373dacff93e600e4426f2b3d3dd264e90ed'
tokenizer = LlamaTokenizerFast.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Trainer
class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
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


def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
                [tokenizer.pad_token_id] * (seq_len - 1) + ids[(seq_len - 1):] + [tokenizer.pad_token_id] * (
                    longest - ids_l)
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


class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def main():
    # load data
    # dataset = datasets.load_from_disk("./data/dataset_new")
    dataset = datasets.load_from_disk("./data/fingpt_data_train_token")
    dataset = dataset.train_test_split(0.2, shuffle=True, seed=42)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # world_size = 1
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        # gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # config
    deepspeed_config = "./config_.json"
    # import deepspeed
    # deepspeed.init_distributed(dist_backend = "gloo")
    training_args = TrainingArguments(
        output_dir='./llama3.1_7b_out',
        # output_dir='./test',
        logging_steps=200,
        # max_steps=10000,
        num_train_epochs=2 * 4,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=200,
        save_steps=400,
        # ddp_backend = 'gloo',
        fp16=True,
        # bf16=True,
        deepspeed=deepspeed_config,
        torch_compile=False,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if ddp else None,
        # testing only, comment otherwise
        dataloader_num_workers=64,
        dataloader_pin_memory=True
    )

    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type='nf4'
    # )

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)


    # load model
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        # load_in_8bit=True,
        # load_in_4bit=True,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.float16,
        quantization_config = quantization_config
        # cache_dir='/colab_space/yanglet/model_weight/LLM'
    )

    model = prepare_model_for_kbit_training(model)

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,
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

    trainer.train()
    writer.close()
    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
