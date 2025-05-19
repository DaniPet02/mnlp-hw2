# Import Datases to work with Transformers by Hugging-Face
import torch
import pandas as pd
import numpy as np


class EvalModel:
    def __init__(self, model, tokenizer, device, tokenize):
        self.model_name = model
        self.tokenizer = tokenizer
        self.device = device
        self.tokenize = tokenize

    def __call__(self, hf):
        """
        Evaluate the model on a test set and return the results.
        Args:
            standard_token: The test set to evaluate the model on.
            tokenizer: The tokenizer used for the model.
            model: The model to evaluate.
            device: The device to use for evaluation.
        Returns:
            The generated text from the model.
        """

        outputs = []
        model = model.eval()

        with torch.no_grad():
            model.to(self.device)
            hf.map(self.tokenize, batched=True)

            for example in hf:
                example = example.to(self.device)
                ids = example.input_ids.to(self.device)
                attention = example.attention_mask.to(self.device)
                output = self.model.generate(ids, attention_mask=attention, max_length=512)
                output = self.tokenizer.decode(output, skip_special_tokens=True)
                outputs.append(output)
                hf.add_column("Translation", outputs)

        return hf


class GeminiJudge:
    # TODO
    pass