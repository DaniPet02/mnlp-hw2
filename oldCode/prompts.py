from typing import Dict, List
from GenericPrompter import GenericPrompter


class LLMPrompt(GenericPrompter):
    """LLM Few-shots"""

    def __init__(self, base_tokenizer):
        super().__init__(base_tokenizer)
        
    def _tokenize(self, examples: List[Dict[str, str]]) -> Dict:
        tokenizer = self.get_tokenizer()
        chat_tk = tokenizer.apply_chat_template(examples, padding=True, max_length=tokenizer.model_max_length)
        return tokenizer(chat_tk, return_tensors="pt", padding=True, max_length=tokenizer.model_max_length)
    
    def __call__(self, examples) -> Dict:
        chat = [
            [
                {"role": "system",   "content": "Sei un traduttore esperto di Italiano Antico "},
                {"role": "user",     "content": "Traduci 'La corte era in gran fermento.' in Italiano Moderno"},
                {"role": "assistant","content": "Italiano Antico: 'La corte era in gran fermento.' Italiano Moderno: 'La corte era molto agitata.'"},
                {"role": "user",      "content": f"Traduci '{example}' in Italiano Moderno"},
                {"role": "assistant", "content": ""}
            ] for example in examples["Sentence"] ]

        model_inputs = self._tokenize(chat)
        return model_inputs