class StandardPrompt(GenericPrompter):
    """
    standard_prompt function to create the input for the model
    It takes a list of examples and returns a list of strings
    where each string is a prompt for the model.
    """

    def __init__(self, base_tokenizer, max_tokens=120) -> None:
        super().__init__(base_tokenizer)
        self.max_tokens = max_tokens

    def shot(self, raw:bool=False):
        s = "Translate Old Italian to Modern Italian\n"
        return s if raw else self._tokenize(s)
    
    def _tokenize(self, examples):
        return self.get_tokenizer()(
            examples, return_tensors="pt", padding=True, truncation=True, max_length=120
        )
    def __call__(self, examples):
        inputs = [
            self.shot(True) + f'Translate "{example}" in Modern Italian idioama'
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
    def shot(self, raw:bool=False):
        s= "\n"
        return s if raw else self._tokenize(s)
    
    def _tokenize(self, samples):
        self.get_tokenizer()(
            samples, return_tensors="pt", padding=True, truncation=True, max_length=120
        )
        
    def __call__(self, examples):
        inputs = [
            self.shot(True) + '>>ita<< ' + example
            for example in examples["Sentence"]
        ]
        model_inputs = self._tokenize(inputs)
        return model_inputs



class AuthorPrompt(GenericPrompter):
    """
    author_prompt function to create the input for the model
    It takes a list of examples and returns a list of strings
    where each string is a prompt for the model.
    """

    def __init__(self, base_tokenizer) -> None:
        super().__init__(base_tokenizer)

    def _tokenize(self, examples):
        self.get_tokenizer()(
            examples, return_tensors="pt", padding=True, truncation=True, max_length=120
        )
    def shot(self, raw:bool=False):
        s = ""
        return s if raw else self._tokenize(s)
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

    def shot(self, raw:bool=False):
        s = "Become QA bot\n"
        return s if raw else self._tokenize(s)
    
    def _tokenize(self, examples):
        self.get_tokenizer()(
            examples, return_tensors="pt", padding=True, truncation=True, max_length=120
        )
    def __call__(self, examples):
        inputs = [
            self.shot(True) + 'Can you express this sentence: "' + example["Sentence"] + '" in a more colloquial style? '
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
    
    def shot(self, raw:bool=False):
        s = 'Example 1: Archaic: "Caron, non ti crucciare: vuolsi così colà dove si puote ciò che si vuole, e più non dimandare." \n'\
            'Modern: "Caronte, non ti agitare: si vuole così lassù dove è possibile tutto ciò che si vuole, quindi non dire altro." \n'\
            'Example 2: Archaic: "Amor, ch\'al cor gentil ratto s\'apprende prese costui de la bella persona che mi fu tolta; e \'l modo ancor m\'offende." \n'\
            'Modern: "L\'amore, che si attacca subito al cuore nobile, colpì costui per il bel corpo che mi fu tolto, e il modo ancora mi addolora." \n'
        
        return s if raw else self._tokenize(s)
    def _tokenize(self, examples):
        return self.get_tokenizer()(
            examples, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )

    def __call__(self, examples):
        inputs = [
            'Now translate: Archaic: "' + example + '" to Modern: '
            for example in examples["Sentence"]
        ]
        model_inputs = self._tokenize(inputs)
        return model_inputs
class Period_Region_Prompt(GenericPrompter):
    """
    period_region_prompt function to create the input for the model
    It takes a list of examples and returns a list of strings
    where each string is a prompt for the model.
    """

    def shot():
        return ""

    def __init__(self, base_tokenizer) -> None:
        super().__init__(base_tokenizer)

    def _tokenize(self, examples):
        return self.get_tokenizer()(
            examples, return_tensors="pt", padding=True, truncation=True, max_length=120
        )

    def __call__(self, examples):
        dates = examples["Date"]
        regions = examples["Region"]
        sentences = examples["Sentence"]
        inputs = [
            f'This sentence: "{sentence}" was written in {date} in the "{region}" region. Translate it to Modern Italian: '
            for date, region, sentence in zip(dates, regions, sentences)
        ]
        model_inputs = self.get_tokenizer()(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=120
        )
        return model_inputs
#class StylePrompt(GenericPrompter):
    """
    style_prompt function to create the input for the model
    It takes a list of examples and returns a list of strings
    where each string is a prompt for the model.
    """

    def __init__(self, base_tokenizer) -> None:
        super().__init__(base_tokenizer)

    
    def shot(self):
        return self._tokenize("")
    
    def _tokenize(self, examples):
        self.get_tokenizer()(
            examples, return_tensors="pt", padding=True, truncation=True, max_length=120
        )
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
