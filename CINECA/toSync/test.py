from huggingface_hub import push_to_hub_keras
import torch
import torch.nn as nn
import pandas as pd
from datasets import Dataset, load_dataset
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM


# Load the model and tokenizer
model_path = "./results/checkpoint-369"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval().to("cuda")

examples = Dataset.from_csv("test_dataset_ann1.csv")

def format_prompt(sentence):
    messages = [
        {"role": "system", "content": "rispondendo in Italiano riscrivi Le Frasi in Italiano Antico in Italiano Moderno seguendo le indicazioni dello 'user' e rispondendo precisamente alle richieste senza dare spiegazioni"},
        {"role": "user", "content": "frase in Italiano Antico: 'Orlando, che gran tempo inamorato fu de la bella Angelica', sotituisci i termini poco utilizzati o errati"},
        {"role": "assistant", "content": "nuova Frase: 'Orlando che da molto tempo innamorato è della bella Agelica"},
        {"role": "user", "content": "riordina le parole in modo che la frase risulti più scorrevole"},
        {"role": "assistant", "content": "nuova Frase: 'Orlando che è innamorato della bella Angelica da molto tempo'"},
        {"role": "user", "content": "migliora il significato della frase"},
        {"role": "assistant", "content": "nuova Frase: 'Orlando è innamorato della bella Angelica da molto tempo'"},
        {"role": "user", "content": f"Frase in Italiano Antico: '{sentence}', sostituisci i termini poco utilizzati o errati"},
        {"role": "assistant", "content": ""}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

prompts = [format_prompt(s) for s in examples["Sentence"]]
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=180).to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
examples = examples.add_column("Generation", decoded)

# Push the results to the Hugging Face Hub
examples.push_to_hub(repo_id="DaniPet02/atmi-minerva7b-lima", 
                     commit_message="Pushing results of the model inference",
                     token="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")