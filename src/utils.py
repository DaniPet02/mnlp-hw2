# Import Datases to work with Transformers by Hugging-Face
from abc import ABC, abstractmethod
from pathlib import Path
from typing import override
from io import StringIO
import datasets
import pandas as pd
import torch
from google import genai
from google.genai import types
from pydantic import BaseModel
from prompts import baseline


class EvaluationGride(BaseModel):
    short_comment: str
    proposed_traslation: str
    score: int


class GenericPrompter(ABC):
    def __init__(self, base_tokenizer) -> None:
        self._tokenizer = base_tokenizer

    @abstractmethod
    def __call__(self, examples):
        pass

    def get_tokenizer(self):
        return self._tokenizer


class StandardPrompt(GenericPrompter):
    """
    standard_prompt function to create the input for the model
    It takes a list of examples and returns a list of strings
    where each string is a prompt for the model.
    """

    def __init__(self, base_tokenizer) -> None:
        super().__init__(base_tokenizer)

    def __call__(self, examples):
        inputs = [
            'translate "' + example + '" to Italian: '
            for example in examples["Sentence"]
        ]
        model_inputs = self.get_tokenizer()(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=120
        )
        return model_inputs


class PromptModel:
    def __init__(self, model, prompter, device="cpu"):
        self.model = model
        self.device = device
        self.prompt = prompter

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

        model = self.model.eval()

        with torch.no_grad():
            model.to(self.device)
            hf = hf.map(self.prompt, batched=True)
            hf = hf.map(self.promptSentence)

        return hf

    def promptSentence(self, example):
        ids_list = example["input_ids"]
        attention_mask_list = example["attention_mask"]

        input_ids_tensor = torch.tensor([ids_list], device=self.device)
        attention_mask_tensor = torch.tensor([attention_mask_list], device=self.device)

        generated_ids = self.model.generate(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            max_new_tokens=120,
            do_sample=True,
            top_k=10,
            top_p=0.95,
        )

        # Use the tokenizer stored in PromptModel directly
        decoded_output = self.prompt.get_tokenizer().decode(
            generated_ids[0], skip_special_tokens=True
        )

        return {"generated_text": decoded_output}


class GeminiJudge:
    def __init__(
        self,
        TOKEN: str,
        model: str = "gemini-2.0-flash",
        contents: str = "",
        log_dir: str = "",
    ):
        self.client = genai.Client(api_key=TOKEN)
        self.model = model
        self.log = Path(log_dir)
        base = baseline
        self.config = types.GenerateContentConfig(
            system_instruction=base,
            max_output_tokens=500,
            response_mime_type="application/json",
            response_schema=list[EvaluationGride],
        )

        response = self.client.models.generate_content(
            model=self.model, config=self.config, contents=contents
        )

        pd.read_json(StringIO(response.text)).to_json(self.log.joinpath("init.json"))

    def judge(self, query: str):
        response = self.client.models.generate_content(
            model=self.model, config=self.config, contents=query
        )

        return response.text
