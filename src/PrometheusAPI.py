from typing import Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from GenericPrometheus import GenericPrometheus

class PrometheusJudge(GenericPrometheus):
    def __init__(self, device:str="cpu"):
        self.device = device
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,  # or torch.float16 if bfloat16 not supported
        )
        model = AutoModelForCausalLM.from_pretrained("prometheus-eval/prometheus-7b-v2.0",device_map=device, quantization_config=bnb_config)
        tokenizer = AutoTokenizer.from_pretrained("prometheus-eval/prometheus-7b-v2.0")
        conf = {
            "max_new_tokens": 680,
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
            f"{ABS_SYSTEM_PROMPT} \n\n {ABSOLUTE_PROMPT.format(instruction=prompt, response=response)}" # Fill the prompt with your data
            for prompt, response in zip(inputs["Prompt"],inputs["Translation(Generated)"])]
        
        return self.get_tokenizer()(messages, padding=True, max_length=680, truncation=True)

class MPrometheusJudge(GenericPrometheus):
    def __init__(self, device:str="cpu"):
        self.device = device
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,  # or torch.float16 if bfloat16 not supported
        )
        model = AutoModelForCausalLM.from_pretrained("Unbabel/M-Prometheus-7B",device_map=device, quantization_config=bnb_config)
        tokenizer = AutoTokenizer.from_pretrained("Unbabel/M-Prometheus-7B")
        conf = {
            "max_new_tokens": 680,
            "do_sample":True
        }
        super().__init__(model, tokenizer, conf)

    def prompt(self, inputs) -> List[Dict[str,str]]:
        ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
        
        # Customize the prompt for Italian Tranlation - substitute {instructions-rubric-reference_answers} with correct instructions
        ABSOLUTE_PROMPT ="""###Task Description: An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given. 
        1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general. 
        2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric. 
        3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)" 
        4. Please do not generate any other opening, closing, and explanations.

        ###The instruction to evaluate:
        Translate the following text from Archaic Italian to Modern Italian: {instruction}

        ###Response to evaluate:
        {response}

        ###Reference Answer (Score 5):
        Italiano Antico: 'Orlando, che gran tempo inamorato fu de la bella Angelica.'
        Italiano Moderno (Score 1): 'Orlando, che gran tempo è, inamorato fu de la bella Angelica.'
        Italiano Moderno (Score 2): 'Orlando, inamorato fu gran tempo de la bella Angelica.'
        Italiano Moderno (Score 3): 'Orlando da gran tempo è inamorato della bella Angelica.'
        Italiano Moderno (Score 4): 'Orlando è da grande tempo innamorato della bella Angelica.'
        Italiano Moderno (Score 5): 'Orlando è innamorato della bella Angelica da molto tempo.'

        ###Score Rubrics: [Accuracy, Fluency, Style]
        Score 1: The translation contains major errors that significantly alter the meaning of the source text. It is barely comprehensible and reads like a poor machine translation. The style is completely inconsistent with the source text.
        Score 2: The translation has several inaccuracies that affect the overall meaning. It is difficult to read and understand, with frequent awkward phrasings. The style only occasionally matches the source text.
        Score 3: The translation is mostly accurate but has some minor errors that don't significantly alter the meaning. It is generally understandable but lacks natural flow in some parts. The style is somewhat consistent with the source text.
        Score 4: The translation is accurate with only a few negligible errors. It reads naturally for the most part, with occasional minor awkwardness. The style largely matches that of the source text.
        Score 5: The translation is highly accurate, conveying the full meaning of the source text. It reads as fluently as an original text in the target language. The style perfectly captures the tone and register of the source text.

        ###Feedback:
        """
      
        messages = [
            f"{ABS_SYSTEM_PROMPT} \n\n {ABSOLUTE_PROMPT.format(instruction=prompt, response=response)}" # Fill the prompt with your data
            for prompt, response in zip(inputs["Prompt"],inputs["Translation(Generated)"])]
        
        return self.get_tokeinizer()(messages, padding=True, max_length=680, truncation=True)