import torch

class Prompt2Model:
    def __init__(self, model, prompter, device="cpu"):
        """
        Initializes PromptModel.

        Args:
            model: Hugging Face pre-trained model (es. from AutoModelForCausalLM).
            prompter: An object or a function responsible for prompt formatting
                      and tokenization. Must have a method `get_tokenizer()`
                      returning an instance of the Hugging Face tokenizer, and must
                      be callable so that, when used with `dataset.map(prompter, batched=True)`,
                      returns a dictionary with at least "input_ids" and "attention_mask".
            device (str): Device on which execute the model ("cpu" or "cuda").
            default_generation_batch_size (int): Predefined Batch Dimension for generation.
        """
        self.model = model.to(device)
        self.device = device
        self.prompter = prompter
        self.model.eval() # Sets the model in evaluation mode by default

    def _generate_text_batched(self, examples_batch):
        """
        Internal method to generate text for a batch of examples.
        Called by dataset.map(..., batched=True).
        """
        tokenizer = self.prompter.get_tokenizer()

        if tokenizer.pad_token_id is None:
            # print("Warning: tokenizer.pad_token_id is not set. eos_token_id is used for padding of inputs.")
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # examples_batch['input_ids'] list of lists of ID token from previous mapping.
        # examples_batch['attention_mask'] lista of lists of attention masks.
        
        if not examples_batch["input_ids"]: # Handles the case of an empty batch
            return {"generated_text": []}

        # Does padding of the input sequences internal to the current batch
        # to the max length present in that specific batch
        max_len_in_batch = max(len(ids) for ids in examples_batch["input_ids"])

        padded_input_ids_list = []
        padded_attention_mask_list = []

        for ids, mask in zip(examples_batch["input_ids"], examples_batch["attention_mask"]):
            padding_length = max_len_in_batch - len(ids)
            
            current_padded_ids = ids + [tokenizer.pad_token_id] * padding_length
            current_padded_mask = mask + [0] * padding_length # Padding of attention mask with 0
            
            padded_input_ids_list.append(current_padded_ids)
            padded_attention_mask_list.append(current_padded_mask)

        if not padded_input_ids_list:
             return {"generated_text": []}

        input_ids_tensor = torch.tensor(padded_input_ids_list, device=self.device, dtype=torch.long)
        attention_mask_tensor = torch.tensor(padded_attention_mask_list, device=self.device, dtype=torch.long)

        # Executes generation
        # Note: pad_token_id in model.generate is used for padding *during* generation,
        # if generated sequences end at different times or if num_return_sequences > 1.
        # It is often set to tokenizer.eos_token_id for causal models.
        generated_ids_batch_tensor = self.model.generate(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            max_new_tokens=120,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decodes the generated sequences.
        # generated_ids_batch_tensor is usually a tensor of input_ids for CausalLM models.
        # tokenizer.batch_decode handles the decoding.
        decoded_outputs = tokenizer.batch_decode(
            generated_ids_batch_tensor, skip_special_tokens=True
        )

        return {"generated_text": decoded_outputs}
    
    def _prepare_chat_inputs_batched(self, examples_batch, conversations_column_name):
        """
        Internal method to prepare inputs for a batch of conversations
        using tokenizer.apply_chat_template.
        Called by dataset.map(..., batched=True).
        """
        
        # examples_batch is a dictionary of lists
        # examples_batch[conversations_column_name] is a list of conversations history
        # Example:
        # [
        #   [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], # conversation 1
        #   [{"role": "user", "content": "Altra domanda..."}]                             # conversation 2
        # ]
        list_of_conversations = examples_batch[conversations_column_name]

        if not list_of_conversations:
            return {"input_ids": [], "attention_mask": []}

        # apply_chat_template can process a batch of conversations.
        # Tokenizes, formats, adds prompt of generation, does padding and truncation.
        batch_tokenized_inputs = self.tokenizer.apply_chat_template(
            list_of_conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding="longest",  # Does padding to the longest sequence in the current batch
            truncation=True,    # Truncates sequences if they exceed the max length of the model
            max_length=self.tokenizer.model_max_length if self.tokenizer.model_max_length is not None else 512, # Max len of the model or default
            return_tensors=None, # Returns lists of lists (for compatibility with .map)
            return_dict=True      # Returns a dictionary with 'input_ids' and 'attention_mask'
        )
        
        return {
            "input_ids": batch_tokenized_inputs["input_ids"],
            "attention_mask": batch_tokenized_inputs["attention_mask"],
        }
    
    def causal_prompt(self, hf_dataset, generation_batch_size = 8):
        """
        Processes a dataset to generate text using the model.

        Args:
            hf_dataset (datasets.Dataset): An object Dataset of Hugging Face.
            We presume that this dataset contains the text to be processed (or the necessary fields for the prompter).
            generation_batch_size (Optional[int]): Dimension of the batch for the generation phase.
            If given, overwrites the predefined of the instance.

        Returns:
            datasets.Dataset: A Hugging Face Dataset with an added column "generated_text".
        """

        # Ensure the model is in evaluation mode
        self.model.eval()

        with torch.no_grad():
            # Phase 1: Apply the prompter (prompt formatting and tokenization).
            # self.prompter is a function/callable that takes a batch of examples
            # and returns 'input_ids' and 'attention_mask' tokenized.
            # print("Fase 1: Application of the prompter (tokenization/formatting)...")
            
            # Note: the batch_size for this first mapping can also be specified if necessary:
            # depends on how computationally intensive is self.prompter.
            processed_hf_dataset = hf_dataset.map(
                self.prompter, # Tokenization/formatting function of the prompt
                batched=True 
            )

            # Phase 2: Generate text using the model in batches.
            # print(f"Fase 2: Generation of the text with batch dimension {generation_batch_size}...")
            generations_dataset = processed_hf_dataset.map(
                self._generate_text_batched,
                batched=True,
                batch_size=generation_batch_size
            )

        return generations_dataset