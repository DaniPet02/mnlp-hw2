import os
from matplotlib import pyplot as plt
# Import Datases to work with Transformers by Hugging-Face

# Imports for Transformers
import matplotlib.pyplot as plt
import os
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