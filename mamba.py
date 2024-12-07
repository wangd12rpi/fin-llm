from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer, AutoConfig, AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from datasets import load_dataset, load_from_disk, DatasetDict
from mamba_ssm import MambaLMHeadModel
from torch.cuda.amp import autocast, GradScaler
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
import random


def init_model():
    
    
    model = MambaLMHeadModel.from_pretrained (pretrained_model_name="state-spaces/mamba2-130m")
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba2-130m")
    config = AutoConfig.from_pretrained("state-spaces/mamba2-130m")
    print("Config of mamba2-130m")
    print(config)
    
    model = MambaLMHeadModel.from_pretrained (pretrained_model_name="state-spaces/mamba2-2.7b")
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
    config = AutoConfig.from_pretrained("state-spaces/mamba2-2.7b")
    print("Config of mamba2-2.7b")
    print(config)




def model_training(
    model_name="mamba2-130m",
    dataset_name="FinGPT/fingpt-sentiment-train",  # Default to one dataset
    model=MambaLMHeadModel.from_pretrained(pretrained_model_name="state-spaces/mamba2-130m"),
    tokenizer=AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf"),
    params={"lr": 5e-5, "num_training_steps": 30, "weight_decay": 0.1},
    device=torch.device("cpu")
):
    # Load model
    model.to(device)
    
    # Load parameters
    lr = params["lr"]
    num_training_steps = params["num_training_steps"]
    weight_decay = params["weight_decay"]
    print(f"Training {model_name} on {dataset_name} with parameters: {params}")
    
    # Load dataset
    raw_datasets = load_dataset(dataset_name)
    train_data = raw_datasets["train"]
    test_data = raw_datasets["test"]
    
    # Preprocess the datasets
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_train = train_data.map(tokenize_function, batched=True)
    tokenized_test = test_data.map(tokenize_function, batched=True)
    
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Set batch size based on model size
    if model_name == "mamba2-130m":
        batch_size = 32
    elif model_name == "mamba2-2.7b":
        batch_size = 1
    else:
        print("Invalid model name")
        return
    
    # DataLoader
    train_dataloader = DataLoader(tokenized_train, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(tokenized_test, shuffle=False, batch_size=batch_size)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    num_epoch = num_training_steps // len(train_dataloader) + 1
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    print(f"Number of epochs: {num_epoch}")
    global_step = 0
    model.train()
    
    # Training loop
    with SummaryWriter(log_dir=f"runs/{dataset_name.replace('/', '_')}") as writer:
        for epoch in range(num_epoch):
            for batch_idx, batch in enumerate(tqdm(train_dataloader)):
                if global_step >= num_training_steps:
                    break
                
                inputs = batch["input_ids"].to(device)
                labels = batch["input_ids"].to(device)
                
                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    logits = outputs.logits
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                scaler = GradScaler()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                writer.add_scalar("Loss/train", loss.item(), global_step)
                global_step += 1
                
                if batch_idx % 100 == 0:
                    print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")
    
    # Save model
    save_path = f"checkpoints/{model_name}_{dataset_name.replace('/', '_')}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
    
    return model, tokenizer


def eval(model, tokenizer, device = torch.device("cpu")):
    
    print("Evaluating the model")

    
    
    model.to(device)
    
    test_dataset = load_from_disk('data/mamba2_tokenized_pubmed_abstract_512')['test']
    test_dataset = test_dataset.rename_column('abstract', 'text')
    
    # less samples for faster evaluation
    num_samples = 500

        
    random.seed(0)

    # Randomly sample the indices
    sampled_indices = random.sample(range(len(test_dataset)), num_samples)
    sampled_dataset = test_dataset.select(sampled_indices)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

    test_tokenized = sampled_dataset.map(tokenize_function, batched=True, num_proc=32)
    test_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    test_tokenized = test_tokenized.remove_columns('text')
    
    batch_size = 32
    test_dataloader = DataLoader(test_tokenized, shuffle=False, batch_size=batch_size)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            inputs = batch['input_ids'].to(device)
            labels = batch['input_ids'].to(device)
            
            outputs = model(inputs)
            
            # Get logits
            logits = outputs.logits  # Assuming model outputs a named tuple with logits
            
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Compute the loss using CrossEntropyLoss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Accumulate the loss
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)

    print(f"Perplexity: {perplexity:.4f}")
    print(f"avg_loss: {avg_loss:.4f}")
    
    return perplexity



    