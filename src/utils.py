# Import Datases to work with Transformers by Hugging-Face
from abc import ABC, abstractmethod
import json
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
import time
import os
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

    def __init__(self, base_tokenizer, max_tokens=120) -> None:
        super().__init__(base_tokenizer)
        self.max_tokens = max_tokens

    def __call__(self, examples):
        inputs = [
            'Traduci "' + example + '" in Italiano Moderno'
            for example in examples["Sentence"]
        ]
        model_inputs = self.get_tokenizer()(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=150
        )
        return model_inputs

class OpusPrompt(GenericPrompter):
    """
    standard_prompt function to create the input for the model
    It takes a list of examples and returns a list of strings
    where each string is a prompt for the model.
    """

    def __init__(self, base_tokenizer, max_tokens=120) -> None:
        super().__init__(base_tokenizer)
        self.max_tokens = max_tokens

    def __call__(self, examples):
        inputs = [
            '>>ita<< ' + example
            for example in examples["Sentence"]
        ]
        model_inputs = self.get_tokenizer()(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=120
        )
        return model_inputs

class LLMPrompt(GenericPrompter):
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
    

class AuthorPrompt(GenericPrompter):
    """
    author_prompt function to create the input for the model
    It takes a list of examples and returns a list of strings
    where each string is a prompt for the model.
    """

    def __init__(self, base_tokenizer) -> None:
        super().__init__(base_tokenizer)

    def __call__(self, examples):
        inputs = [
            'Knowing that ' + example["Author"] + 'wrote this, '
            'translate "' + example["Sentence"] + '" to Italian: '
            for example in examples
        ]
        model_inputs = self.get_tokenizer()(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=120
        )
        return model_inputs


class QuestionPrompt(GenericPrompter):
    """
    question_prompt function to create the input for the model
    It takes a list of examples and returns a list of strings
    where each string is a prompt for the model.
    """

    def __init__(self, base_tokenizer) -> None:
        super().__init__(base_tokenizer)

    def __call__(self, examples):
        inputs = [
            'Can you express this sentence: "' + example["Sentence"] + '" in a more colloquial style? '
            for example in examples
        ]
        model_inputs = self.get_tokenizer()(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=120
        )
        return model_inputs


class Few_Shot_Prompt(GenericPrompter):
    '''
    few_shot_prompt function to create the input for the model
    It takes a list of examples and returns a list of strings
    where each string is a prompt for the model.
    '''

    def __init__(self, base_tokenizer) -> None:
        super().__init__(base_tokenizer)
        
    def __call__(self, examples):
        inputs = [
            'Example 1: Archaic: "Caron, non ti crucciare: vuolsi così colà dove si puote ciò che si vuole, e più non dimandare." \n'
            'Modern: "Caronte, non ti agitare: si vuole così lassù dove è possibile tutto ciò che si vuole, quindi non dire altro." \n'
            'Example 2: Archaic: "Amor, ch\'al cor gentil ratto s\'apprende prese costui de la bella persona che mi fu tolta; e \'l modo ancor m\'offende." \n'
            'Modern: "L\'amore, che si attacca subito al cuore nobile, colpì costui per il bel corpo che mi fu tolto, e il modo ancora mi addolora." \n'
            'Now translate: Archaic: "' + example + '" to Modern: '
            for example in examples["Sentence"]
        ]
        model_inputs = self.get_tokenizer()(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        return model_inputs
    

class Period_Region_Prompt(GenericPrompter):
    """
    period_region_prompt function to create the input for the model
    It takes a list of examples and returns a list of strings
    where each string is a prompt for the model.
    """

    def __init__(self, base_tokenizer) -> None:
        super().__init__(base_tokenizer)

    def __call__(self, examples):
        dates = examples["Date"]
        regions = examples["Region"]
        sentences = examples["Sentence"]
        inputs = [
            'This sentence: "' + "{sentence}" + '" was written in ' + "{date}" + 
            ' in the "' + "{region}" + '" region. Translate it to Modern Italian: '
            for date, region, sentence in zip(dates, regions, sentences)
        ]
        model_inputs = self.get_tokenizer()(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=120
        )
        return model_inputs


class StylePrompt(GenericPrompter):
    """
    style_prompt function to create the input for the model
    It takes a list of examples and returns a list of strings
    where each string is a prompt for the model.
    """

    def __init__(self, base_tokenizer) -> None:
        super().__init__(base_tokenizer)

    def __call__(self, examples):
        inputs = [
            'The following sentence represents an example from the Dolce Stil Novo (sweet new style) literary movement, developed in the 13th and 14th century in Italy: "'
            + example + '" Translate it to modern Italian: '
            for example in examples["Sentence"]
        ]
        model_inputs = self.get_tokenizer()(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=240
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
            max_new_tokens=512,
            do_sample=True,
            top_k=100,            # aumento della diversità controllando le parole candidate
            top_p=0.95,          # campionamento nucleus per ulteriori controlli sulla varietà
            temperature=0.8,     # riduce la casualità e aumenta la coerenza
            repetition_penalty=1.2,  # penalizza ripetizioni
            num_return_sequences=1,  # numero di risposte generate
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

        if self.log.joinpath("gemini_log.json").exists():
            os.remove(self.log.joinpath("gemini_log.json"))
        try:
            pd.read_json(StringIO(response.text)).to_json(self.log.joinpath("gemini_log.json"))
        except (json.JSONDecodeError, ValueError) as e:
            error_log_path = self.log.joinpath(f"panic_gen{time.time():.f2}.txt")
            with open(error_log_path, "w", encoding="utf-8") as f:
                f.write("--- Errore di Parsing JSON ---\n")
                f.write(f"Tipo di errore: {type(e).__name__}\n")
                f.write("\n-----------------------------------------------------\n")
      

    def judge(self, query: str):
        response = self.client.models.generate_content(
            model=self.model, config=self.config, contents=query
        )

        try:
            pd.read_json(StringIO(response.text)).to_json(self.log.joinpath("gemini_log.json"))
        except (json.JSONDecodeError, ValueError) as e:
            error_log_path = self.log.joinpath(f"panic_gen{time.time():.f2}.txt")
            with open(error_log_path, "w", encoding="utf-8") as f:
                f.write("--- Errore di Parsing JSON ---\n")
                f.write(f"Tipo di errore: {type(e).__name__}\n")
                f.write("\n-----------------------------------------------------\n")

        return response.text
