import os
from pathlib import Path
from typing import Dict, Union
from matplotlib import pyplot as plt
import pandas as pd
import json
# Import Datases to work with Transformers by Hugging-Face

# Imports for Transformers
import os
import torch
from tqdm.auto import tqdm
from transformers import TrainerCallback

class Report(TrainerCallback):
    """
    Personalized callback to draw loss and metrics graphs.
    """
    def __init__(self, plotting_dir="./training_plots"):
        self.plotting_dir = plotting_dir
        self.log_history = []
        os.makedirs(self.plotting_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Event called after logging the last metrics.
        Collects loss and metrics data.
        """
        if logs is not None:
            self.log_history.append(logs)

    def on_train_end(self, args, state, control, **kwargs):
        """
        Event called at the end of training.
        Draws and saves the graphs.
        """
        print("Training done. Generating graphs...")

        train_losses = []
        eval_losses = []
        eval_metrics = {}
        global_steps = []
        epochs = []

        for log in self.log_history:
            # Collect training loss (recorded at logging_steps)
            if 'loss' in log:
                train_losses.append(log['loss'])
                global_steps.append(log.get('step', None)) # Use 'step' if available
            # Collect evaluation metrics (recorded at evaluation_strategy)
            elif 'eval_loss' in log:
                eval_losses.append(log['eval_loss'])
                epochs.append(log.get('epoch', None)) # Use 'epoch' if available
                for key, value in log.items():
                    if key.startswith('eval_') and key != 'eval_loss' and isinstance(value, (int, float)):
                        if key not in eval_metrics:
                            eval_metrics[key] = []
                        eval_metrics[key].append(value)

        # Remove None from global_steps if not uniformely available
        if not all(step is None for step in global_steps):
            # Only filters log containing step for training loss
            train_logs_with_step = [(log['loss'], log['step']) for log in self.log_history if 'loss' in log and 'step' in log]
            train_losses = [log[0] for log in train_logs_with_step]
            global_steps = [log[1] for log in train_logs_with_step]
        else:
            global_steps = list(range(len(train_losses))) # Use range if steps are not logged

        # Remove None from epochs if not uniformely available
        if not all(epoch is None for epoch in epochs):
            # Only filter log containing epoch for eval metrics
            eval_logs_with_epoch = [(log['eval_loss'], log['epoch'], {k:v for k,v in log.items() if k.startswith('eval_') and k != 'eval_loss'}) for log in self.log_history if 'eval_loss' in log and 'epoch' in log]
            eval_losses = [log[0] for log in eval_logs_with_epoch]
            epochs = [log[1] for log in eval_logs_with_epoch]
            eval_metrics = {k: [log[2][k] for log in eval_logs_with_epoch if k in log[2]] for k in eval_metrics.keys()}

        else:
             epochs = list(range(len(eval_losses))) # Use range if epochs are not logged
             # Ensure metrics have same length
             for key in eval_metrics:
                 eval_metrics[key] = eval_metrics[key][:len(epochs)]

        # Plot loss
        if train_losses or eval_losses:
            plt.figure(figsize=(10, 6))
            if train_losses:
                plt.plot(global_steps, train_losses, label='Training Loss')
            if eval_losses:
                plt.plot(epochs, eval_losses, label='Validation Loss')
            plt.xlabel('Step (Training Loss) / Epoch (Validation Loss)')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.plotting_dir, "loss_graph.png"))
            # plt.show()
            plt.close()

        # Plot evaluation metrics
        for metric_name, metric_values in eval_metrics.items():
            if metric_values:
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, metric_values, label=metric_name)
                plt.xlabel('Epoch')
                plt.ylabel(metric_name.replace('eval_', '').capitalize())
                plt.title(f'Validation Metric: {metric_name.replace("eval_", "").capitalize()}')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.plotting_dir, f"{metric_name}.png"))
                # plt.show()
                plt.close()

from datasets import Dataset
def generate_and_save(
    model,
    tokenizer,
    tokenized_dataset: Dataset,
    output_prefix: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
    include_prompt: bool = True,
    format: str = "csv",
    config: Union[Dict[str, Union[int, str]], None] = None,
):
    """
    Generate translations for each example in `tokenized_dataset` and save a CSV/JSONL
    with columns ["Original", "Translation(Generated)"].

    Args:
        model              : HuggingFace seq2seq model
        tokenizer          : corresponding tokenizer
        tokenized_dataset  : HuggingFace Dataset with fields "input_ids" & "attention_mask"
        output_prefix      : output file prefix
        device             : device to run on, e.g. "cuda" or "cpu"
        batch_size         : generation batch size
        include_prompt     : if True, keep the full generated text; if False, strip the prompt from output
        format             : "csv" or "jsonl"
        config             : additional kwargs for model.generate()

    Returns:
        pandas.DataFrame with columns ["Original", "Translation(Generated)"]
    """
    # Move model to device and set evaluation model
    model.eval()

    # Ensure dataset is in PyTorch format
    #tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    loader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    rows = []

    for batch in tqdm(loader, desc="Generating", dynamic_ncols=True):
        
        target = batch["Target"]
        origin = batch["Sentence"]
        input_ids = torch.as_tensor(batch["input_ids"], device=device, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.as_tensor(batch["attention_mask"], device=device, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            preds = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **(config or {})
            )

        decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        decoded_outputs = tokenizer.batch_decode(preds, skip_special_tokens=True)

        for orr,targ, src,pred in zip(origin,target, decoded_inputs, decoded_outputs):
            if include_prompt:
                rows.append([src,orr,targ,pred,"", "", "", ""])
            else:
                trimmed_pred = pred[len(src):].strip()
                rows.append([src,orr,targ,trimmed_pred, "", "", "", ""])

    df = pd.DataFrame(rows, columns=["Prompt","Original","Target","Translation(Generated)","User_Score","Judge_Score(Prometheus)", "Judge_Score(Gemini)", "Judge_Score(GPT)"])

    filename = f"{output_prefix}({model.__class__.__name__}).{format}"
    if format == "csv":
        df.to_csv(filename, index=False)
    elif format == "jsonl":
        df.to_json(filename, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError("Invalid format: use 'csv' or 'jsonl'")

    print(f"Saved to {filename}")
    return df



def jsonline(df, out_file:str|Path):
    """
    Saves a DataFrame Pandas in a file JSON Lines (JSONL).

    Args:
        df (pd.DataFrame): DataFrame to save.
        nome_file_output (str): Name of the file to save the DataFrame in (es. 'dati.jsonl').
    """
    try:
        with open(out_file, 'w', encoding='utf-8') as f:
            for record in df.to_dict(orient='records'):
                json_record = json.dumps(record, ensure_ascii=False)
                f.write(json_record + '\n')
        print(f"DataFrame salvato con successo in '{out_file}'")
        return f
    except Exception as e:
        print(f"Si è verificato un errore durante il salvataggio del DataFrame: {e}")

def csvline(df:pd.DataFrame, out_file:str|Path):
    df.to_csv(path_or_buf=out_file, sep=",", index=False, quoting=1, encoding='utf-8')

def txtline(lines, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
