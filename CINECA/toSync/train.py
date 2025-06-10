from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import json

# Verify GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
model_name = "sapienzanlp/Minerva-7B-base-v1.0"
model_path = "./minerva_cache/models--sapienzanlp--Minerva-7B-base-v1.0/snapshots/ff16836b81e75ae299c01fd6c797115c9935907d"
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
model.to(device)
model.gradient_checkpointing_enable()
torch.cuda.empty_cache()

# Load dataset llamafactory/lima
dataset = load_dataset("llamafactory/lima")
dataset = dataset["train"].train_test_split(test_size=0.05)

# Tokenization
def tokenize_function(batch):
    merged_list = []
    for conversation in batch["conversations"]:
        merged = " ".join(turn["value"] for turn in conversation)
        merged_list.append(merged)
    return tokenizer(merged_list, truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# Data collator for LM
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training Args
training_args = TrainingArguments(
    output_dir="./results",
    #evaluation_strategy="epoch",
    save_strategy="epoch",
    optim="adafactor",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=False,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    #tokenizer=tokenizer,
    data_collator=collator
)

# Training
trainer.train()