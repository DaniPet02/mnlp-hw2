from abc import ABC, abstractmethod
from typing import *
from pandas import Series
from transformers import AutoTokenizer, AutoModelForCausalLM

class Judge(ABC):
    def __init__(self, model:AutoModelForCausalLM, tokenizer:AutoTokenizer, conf:Dict[str, Any]) -> None:
        super()
        self._model = model 
        self._tokenizer = tokenizer
        self.conf = conf

    @abstractmethod
    def judge(self, batch_prompts:List[str]) -> Series:
        pass 

    def get_judge(self):
        return self._model
    
    def get_tokeinizer(self):
        return self.tokenizer

    @abstractmethod
    def prompt(self, inputs:Dict[str, Any]) -> List[Dict[str,str]]:
        pass 