from abc import ABC, abstractmethod
from datasets import Dataset
from typing import Dict, List, Union
from transformers import AutoTokenizer

class GenericPrompter(ABC):
    def __init__(self, base_tokenizer):
        self._tokenizer = base_tokenizer
        self._tokenizer.pad_token = base_tokenizer.eos_token

    @abstractmethod
    def _tokenize(self, examples):
        "Tokenize conform `examples`"
        pass

    @abstractmethod
    def __call__(self, examples:Dataset) -> Dataset:
        pass

    def get_tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

