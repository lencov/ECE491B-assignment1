import os
import csv
import time

class ExperimentLogger:
    """
    A simple experiment logger that writes training and validation metrics to a CSV file.
    
    Each log entry includes:
      - iteration number
      - elapsed wallclock time (in seconds)
      - training loss
      - validation loss
      - learning rate

    Example usage:
      logger = ExperimentLogger("logs/experiment_log.csv")
      # In your training loop:
      logger.log(iteration, train_loss, val_loss, lr)
    """
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.start_time = time.time()
        # Create the directory if needed.
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Open the file and write the header.
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "elapsed_time", "train_loss", "val_loss", "learning_rate"])

    def log(self, iteration: int, train_loss: float, val_loss: float, learning_rate: float):
        elapsed_time = time.time() - self.start_time
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([iteration, elapsed_time, train_loss, val_loss, learning_rate])