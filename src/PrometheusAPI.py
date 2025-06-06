import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from Judge import Judge

from typing import *
class PrometheusJudge(Judge):
    def __init__(self, device:str="cpu"):
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,  # or torch.float16 if bfloat16 not supported
        )
        model = AutoModelForCausalLM.from_pretrained("prometheus-eval/prometheus-7b-v2.0", quantization_config=bnb_config)
        tokenizer = AutoTokenizer.from_pretrained("prometheus-eval/prometheus-7b-v2.0")
        conf = {
            "max_new_tokens": 1000,
            "do_sample":True
        }
        super().__init__(model, tokenizer, conf)

    def prompt(self, inputs) -> List[Dict[str,str]]:
        ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
        
        # Customize the prompt for Italian Tranlation - substitute {instructions-rubric-reference_answers} with correct instructions
        ABSOLUTE_PROMPT ="""###Task Description:
        An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score from 1 to 5, and a score rubric representing an evaluation criteria are given.
        1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
        2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
        3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
        4. Please do not generate any other opening, closing, and explanations.

        ###The instruction to evaluate:
        {instruction} 

        ###Response to evaluate:
        {response}

        ###Reference Answer (Score 5):
        Italiano Antico: 'Orlando, che gran tempo inamorato fu de la bella Angelica.'
        Italiano Moderno (Score 1): 'Orlando, che gran tempo è, inamorato fu de la bella Angelica.'
        Italiano Moderno (Score 2): 'Orlando, inamorato fu gran tempo de la bella Angelica.'
        Italiano Moderno (Score 3): 'Orlando da gran tempo è inamorato della bella Angelica.'
        Italiano Moderno (Score 4): 'Orlando è da grande tempo innamorato della bella Angelica.'
        Italiano Moderno (Score 5): 'Orlando è innamorato della bella Angelica da molto tempo.'

        ###Score Rubrics:
        1: Completely Unacceptable, description: Translation bears no resemblance to the original meaning. 
        Output is gibberish, nonsensical, or entirely irrelevant.
        2: Severe Errors, description: Translation contains critical semantic and/or syntactic errors, 
        significant omissions, or unwarranted additions that distort the core message. 
        The output is unnatural and clearly not human-like.
        3: Partially Incorrect / Lackluster, description: Translation conveys a portion of the original meaning but is marred by 
        noticeable errors (e.g., typos, minor semantic inaccuracies, awkward phrasing). 
        While understandable, it lacks polish and accuracy.
        4: Good Translation, description: Translation is largely accurate and faithful to the original meaning. 
        It is fluent, comprehensible, and semantically sound. 
        Minor stylistic deviations from the original may be present, but overall quality is high.
        5: Perfect / Near-Native Translation, description: Translation is accurate, fluent, complete, and coherent, 
        perfectly capturing the meaning, nuance, and style of the original text. 
        It reads as if originally written in the target language.

        ###Feedback: """
      
        messages = [
            {"role": "user", "content": ABS_SYSTEM_PROMPT + "\n\n" + ABSOLUTE_PROMPT.format(instruction=prompt, response=response)} # Fill the prompt with your data
            for prompt, response in zip(inputs["prompt"],inputs["response"])]
        
        return messages
    
    def judge(self, batch_prompts):
        
        m = self.prompt(batch_prompts)
        tokens = self.get_tokeinizer().apply_chat_template(m, return_tensors="pt", tokenize=True)
        j = self.get_judge().generate(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"], max_new_tokens=1000, do_sample=True)
        decoded = self.get_tokeinizer().batch_decode(j)

        return decoded