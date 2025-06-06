import torch
import torch.nn as nn
import numpy as np
import transformers

model_id = "sapienzanlp/Minerva-7B-base-v1.0"

pipeline = transformers.pipeline(
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Translate the following text to Italian: 'Hello, how are you?'"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)

print(outputs[0]["generated_text"][-1])