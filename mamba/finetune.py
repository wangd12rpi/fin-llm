import torch
from transformers import BioGptTokenizer, BioGptForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import json
import os
from load_data import load_train_texts, load_test_texts, load_test_answer_texts



class PubmedQA_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    
    
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig
from trl import SFTTrainer
# Modified model_finetune function
def model_finetune(
    model=None,  # Default will use Mamba model
    tokenizer=None,  # Default will use Mamba tokenizer
    checkpoint_path="checkpoint/fine-tuned-mamba",
    device="cpu"
):

    model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    
    model.to(device)

    # Load your training texts (assuming this is a list of strings)
    train_texts = load_train_texts()  # You should define this function to load your custom train texts

    # Tokenize the texts
    inputs = tokenizer(train_texts, return_tensors='pt', max_length=1024, truncation=True, padding=True)
    train_dataset = PubmedQA_Dataset(inputs)

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Set to False since it's a causal language model
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results/out',
        overwrite_output_dir=True,
        num_train_epochs=5,  # Adjust as necessary
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        fp16=True,  # Use 16-bit precision if supported by your hardware
    )

    # LoRA configuration (if you want to use LoRA like in your first code)
    lora_config = LoraConfig(
        r=8,
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )

    # Trainer with LoRA
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    # Start training
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)


if __name__ == "__main__":
    # Fine-tune the model
    model_finetune(
        checkpoint_path="checkpoint/fine-tuned-mamba",
        device="cuda" if torch.cuda.is_available() else "cpu" 
    )


    
    