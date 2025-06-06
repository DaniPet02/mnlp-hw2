import torch
from abc import abstractmethod
from torch.utils.data import DataLoader
from typing import List, Dict 
from Judge  import Judge
from tqdm.auto import tqdm

class GenericPrometheus(Judge):
    def __init__(self, model, tokenizer, conf):
        super().__init__(model, tokenizer, conf)

    @abstractmethod
    def prompt(self, inputs) -> List[Dict[str,str]]:
        pass
        
    

    def tokenize(self, hf_dataset):
        return hf_dataset.map(self.prompt, batched=True)

    def judge(self, tokens, batch_size=1):
        # Tokenizza e formatta il dataset
        
        tokens.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        # Crea un DataLoader PyTorch
        loader = DataLoader(tokens, batch_size=batch_size)

        model = self.get_judge()
        tokenizer = self.get_tokenizer()
        model.eval()

        all_outputs = []

        for batch in tqdm(loader, desc="🔥⚖️Judging⚖️🔥"):
            ids = batch["input_ids"]
            att_m = batch["attention_mask"]
            if torch.cuda.is_available():
                ids = ids.to(self.device)
                att_m = att_m.to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=ids,
                    attention_mask=att_m,
                    max_new_tokens=680,
                    do_sample=True
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_outputs.extend(decoded)

        return all_outputs
    
    
    def parse_responses(responses):
        
        scores = []
        for r in responses:
            match = re.search(r"\[RESULT\]\s*([1-5])", text)
            scores.append(int(match.group(1)) if match else None)
        return scores
    
