import torch

class Prompt2Model:
    def __init__(self, model, prompter, device="cpu"):
        """
        Inizializza PromptModel.

        Args:
            model: Il modello Hugging Face pre-addestrato (es. da AutoModelForCausalLM).
            prompter: Un oggetto o una funzione responsabile della formattazione del prompt
                      e della tokenizzazione. Deve avere un metodo `get_tokenizer()`
                      che restituisce l'istanza del tokenizer Hugging Face, e deve
                      essere un callable che, quando usato con `dataset.map(prompter, batched=True)`,
                      restituisce un dizionario con almeno "input_ids" e "attention_mask".
            device (str): Il device su cui eseguire il modello ("cpu" o "cuda").
            default_generation_batch_size (int): Dimensione batch predefinita per la generazione.
        """
        self.model = model.to(device)
        self.device = device
        self.prompter = prompter
        self.model.eval() # Imposta il modello in modalità valutazione di default

    def _generate_text_batched(self, examples_batch):
        """
        Metodo interno per generare testo per un batch di esempi.
        Chiamato da dataset.map(..., batched=True).
        """
        tokenizer = self.prompter.get_tokenizer()

        if tokenizer.pad_token_id is None:
            # print("Avviso: tokenizer.pad_token_id non è impostato. Si utilizza eos_token_id per il padding degli input.")
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # examples_batch['input_ids'] è una lista di liste di ID token dalla mappatura precedente.
        # examples_batch['attention_mask'] è una lista di liste di maschere di attenzione.
        
        if not examples_batch["input_ids"]: # Gestisce il caso di un batch vuoto
            return {"generated_text": []}

        # Effettua il padding delle sequenze all'interno del batch corrente
        # alla lunghezza massima presente in questo specifico batch.
        max_len_in_batch = max(len(ids) for ids in examples_batch["input_ids"])

        padded_input_ids_list = []
        padded_attention_mask_list = []

        for ids, mask in zip(examples_batch["input_ids"], examples_batch["attention_mask"]):
            padding_length = max_len_in_batch - len(ids)
            
            current_padded_ids = ids + [tokenizer.pad_token_id] * padding_length
            current_padded_mask = mask + [0] * padding_length # Padding della maschera di attenzione con 0
            
            padded_input_ids_list.append(current_padded_ids)
            padded_attention_mask_list.append(current_padded_mask)

        if not padded_input_ids_list:
             return {"generated_text": []}

        input_ids_tensor = torch.tensor(padded_input_ids_list, device=self.device, dtype=torch.long)
        attention_mask_tensor = torch.tensor(padded_attention_mask_list, device=self.device, dtype=torch.long)

        # Esegue la generazione
        # Nota: pad_token_id in model.generate è usato per il padding *durante* la generazione,
        # se le sequenze generate terminano in momenti diversi o se num_return_sequences > 1.
        # È spesso impostato su tokenizer.eos_token_id per i modelli causali.
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

        # Decodifica le sequenze generate.
        # generated_ids_batch_tensor di solito include gli input_ids originali per i modelli CausalLM.
        # tokenizer.batch_decode gestisce la decodifica.
        decoded_outputs = tokenizer.batch_decode(
            generated_ids_batch_tensor, skip_special_tokens=True
        )

        return {"generated_text": decoded_outputs}
    def _prepare_chat_inputs_batched(self, examples_batch, conversations_column_name):
        """
        Metodo interno per preparare gli input per un batch di conversazioni
        utilizzando tokenizer.apply_chat_template.
        Chiamato da dataset.map(..., batched=True).
        """
        # examples_batch è un dizionario di liste.
        # examples_batch[conversations_column_name] è una lista di intere cronologie di conversazioni.
        # Esempio:
        # [
        #   [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], # conversazione 1
        #   [{"role": "user", "content": "Altra domanda..."}]                             # conversazione 2
        # ]
        list_of_conversations = examples_batch[conversations_column_name]

        if not list_of_conversations:
            return {"input_ids": [], "attention_mask": []}

        # apply_chat_template può processare un batch di conversazioni.
        # Tokenizza, formatta, aggiunge prompt di generazione, effettua padding e troncatura.
        batch_tokenized_inputs = self.tokenizer.apply_chat_template(
            list_of_conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding="longest",  # Effettua padding alla sequenza più lunga nel batch corrente
            truncation=True,    # Tronca le sequenze se superano la lunghezza massima del modello
            max_length=self.tokenizer.model_max_length if self.tokenizer.model_max_length is not None else 512, # Max len del modello o un default
            return_tensors=None, # Restituisce liste di liste (per compatibilità con .map)
            return_dict=True      # Restituisce un dizionario con 'input_ids' e 'attention_mask'
        )
        
        return {
            "input_ids": batch_tokenized_inputs["input_ids"],
            "attention_mask": batch_tokenized_inputs["attention_mask"],
        }
    
    def causal_prompt(self, hf_dataset, generation_batch_size = 8):
        """
        Elabora un dataset per generare testo utilizzando il modello.

        Args:
            hf_dataset (datasets.Dataset): Un oggetto Dataset di Hugging Face.
                                           Si presume che questo dataset contenga il testo
                                           da elaborare (o i campi necessari per il prompter).
            generation_batch_size (Optional[int]): Dimensione del batch per la fase di generazione.
                                                   Se fornito, sovrascrive quello predefinito dell'istanza.

        Returns:
            datasets.Dataset: Un Dataset di Hugging Face con una colonna aggiunta "generated_text".
        """
    

        # Assicurati che il modello sia in modalità valutazione
        self.model.eval()

        with torch.no_grad():
            # Fase 1: Applica il prompter (formattazione del prompt e tokenizzazione).
            # self.prompter è una funzione/callable che prende un batch di esempi
            # e restituisce 'input_ids' e 'attention_mask' tokenizzati.
            # print("Fase 1: Applicazione del prompter (tokenizzazione/formattazione)...")
            
            # Nota: La batch_size per questa prima mappa può anche essere specificata se necessario.
            # dipende da quanto è intensivo dal punto di vista computazionale self.prompter.
            processed_hf_dataset = hf_dataset.map(
                self.prompter, # Questa dovrebbe essere la funzione di tokenizzazione/formattazione del prompt
                batched=True 
            )
            
        

            # Fase 2: Genera testo usando il modello in batch.
            # print(f"Fase 2: Generazione del testo con dimensione batch {generation_batch_size}...")
            generations_dataset = processed_hf_dataset.map(
                self._generate_text_batched,
                batched=True,
                batch_size=generation_batch_size
            )

        return generations_dataset
        
        