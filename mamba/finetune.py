import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig
from trl import SFTTrainer
from load_data import load_train_texts  # Ensure this is implemented to load your data
import argparse


class PubmedQA_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


def model_finetune(
    model_name,
    tokenizer_name,
    checkpoint_path,
    training_args,
    device="cpu"
):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model.to(device)

    # Load and tokenize training data
    train_texts = load_train_texts()
    

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config,
        train_dataset=train_texts,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune a causal language model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model checkpoints.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device.")
    parser.add_argument("--save_steps", type=int, default=500, help="Steps interval to save checkpoints.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Steps interval to log training progress.")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training.")
    parser.add_argument("--model_name", type=str, default="state-spaces/mamba-130m-hf", help="Pretrained model name.")
    parser.add_argument("--tokenizer_name", type=str, default="state-spaces/mamba-130m-hf", help="Tokenizer name.")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint/fine-tuned-mamba", help="Checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")

    args = parser.parse_args()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
    )

    # Fine-tune the model
    model_finetune(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        checkpoint_path=args.checkpoint_path,
        training_args=training_args,
        device=args.device
    )

