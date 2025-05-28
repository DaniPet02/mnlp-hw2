from transformers import AutoModelForCausalLM, AutoTokenizer
from Judge import Judge

from typing import *
class PRometheusJudge(Judge):
    def __init__(self, device:str="cpu"):
        
        model = AutoModelForCausalLM.from_pretrained("prometheus-eval/prometheus-7b-v2.0")
        tokenizer = AutoTokenizer.from_pretrained("prometheus-eval/prometheus-7b-v2.0")
        conf = {
            "max_new_tokens": 1000,
            "do_sample":True
        }
        super().__init__(model, tokenizer, conf)

    def prompt(self, inputs) -> List[Dict[str,str]]:
        ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
        
        #TODO: Customize the prompt for Italian Tranlation - substitute {instructions-rubric-reference_answers} with correct instructions
        ABSOLUTE_PROMPT ="""###Task Description:
        An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
        2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
        3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
        4. Please do not generate any other opening, closing, and explanations.

        ###The instruction to evaluate:
        {instruction} 

        ###Response to evaluate:
        {response}

        ###Reference Answer (Score 5):
        {reference_answer}

        ###Score Rubrics:
        {rubric}

        ###Feedback: """

      
        messages = [
            {"role": "user", "content": ABS_SYSTEM_PROMPT + "\n\n" + ABSOLUTE_PROMPT.format(instruction=prompt, response=response)} # Fill the prompt with your data
            for prompt, response in zip(inputs["prompt"],inputs["response"]) ]
        
        return messages
    
    def judge(self, batch_prompts):
        
        m = self.prompt(batch_prompts)
        tokens = self.get_tokeinizer().apply_chat_template(m, return_tensors="pt", tokenize=True)
        j = self.get_judge().generate(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"], max_new_tokens=1000, do_sample=True)
        decoded = self.get_tokeinizer().batch_decode(j)

        return decoded
 