import pytorch_lightning as pl
import csv
import os

class LossMetricsLogger(pl.Callback):
    def __init__(self, train_metrics=None, val_metrics=None,
                 train_filename="train_metrics.csv", val_filename="val_metrics.csv"):
        """
        Logs specified training and validation metrics into separate CSV files.
        """
        super().__init__()
        self.train_metrics = train_metrics or []
        self.val_metrics = val_metrics or []
        self.train_filename = train_filename
        self.val_filename = val_filename

        # Write headers if files do not exist
        if not os.path.exists(self.train_filename) and self.train_metrics:
            with open(self.train_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch"] + self.train_metrics)

        if not os.path.exists(self.val_filename) and self.val_metrics:
            with open(self.val_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch"] + self.val_metrics)

    def on_train_epoch_end(self, trainer, pl_module):
        row = [trainer.current_epoch + 1]
        for metric in self.train_metrics:
            value = trainer.callback_metrics.get(metric)
            if value is not None and hasattr(value, "item"):
                value = value.item()
            row.append(value)
        if self.train_metrics:
            with open(self.train_filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

    def on_validation_epoch_end(self, trainer, pl_module):
        row = [trainer.current_epoch + 1]
        for metric in self.val_metrics:
            value = trainer.callback_metrics.get(metric)
            if value is not None and hasattr(value, "item"):
                value = value.item()
            row.append(value)
        if self.val_metrics:
            with open(self.val_filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)