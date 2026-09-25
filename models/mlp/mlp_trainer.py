import logging
from pathlib import Path

import numpy as np
from torchinfo import summary

from utils import save_checkpoint

def predict(model, pred_dl, metric_fn, cuda=False):
    """Run model prediction over `pred_dl` and compute a tracking metric.

    Args:
        model (nn.Module): Trained PyTorch model used for inference.
        pred_dl (DataLoader): Dataloader yielding (input, target) pairs.
        metric_fn (callable): Function taking (outputs, targets) and returning a scalar.
        cuda (bool): If True, move batches to CUDA device.

    Returns:
        tuple: `(predictions, ground_truth)` where both are concatenated numpy arrays.
    """
    model.eval()

    predictions = []
    ground_truth = []
    tracking_metric = RunningAverage()

    for pred_batch, target_batch in pred_dl:
        if cuda:
            pred_batch, target_batch = pred_batch.cuda(non_blocking=True), target_batch.cuda(non_blocking=True)
        output_batch = model(pred_batch)
        metric = metric_fn(output_batch, target_batch)
        tracking_metric.update(metric.item())

        output_batch = output_batch.data.detach().numpy()
        target_batch = target_batch.data.detach().numpy()

        predictions.append(output_batch)
        ground_truth.append(target_batch)
    average_score = tracking_metric()
    logging.info(f"Prediction completed. Average Score: {average_score:.4f}")
    return np.concatenate(predictions, axis=0), np.concatenate(ground_truth, axis=0)

def train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metric_fn, num_epochs, save_dir=None, writer=None, cuda=False):
    """Train `model` using `train_dl` and evaluate on `val_dl` each epoch.

    Saves best and last model summaries to `save_dir` when provided and logs
    training/evaluation metrics to `writer` if available.

    Args:
        model (nn.Module): PyTorch model to train.
        train_dl (DataLoader): Training dataloader.
        val_dl (DataLoader): Validation dataloader.
        optimizer: Optimizer with `zero_grad`/`step` methods.
        loss_fn: Loss function taking (outputs, targets) and returning scalar.
        metric_fn: Metric function for monitoring (e.g., MAE).
        num_epochs (int): Number of training epochs.
        save_dir (str or Path, optional): Directory to write model summaries/checkpoints.
        writer (SummaryWriter, optional): TensorBoard writer for logging.
        cuda (bool): Whether to run batches on CUDA.
    """

    best_val_loss = float('inf')
    best_epoch = None

    for epoch in range(num_epochs):
        train_loss, train_metric = train(model, optimizer, loss_fn, metric_fn, train_dl, cuda=cuda)
        if writer:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Metric/Train", train_metric, epoch)
        
        val_loss, val_metric = evaluate(model, loss_fn, metric_fn, val_dl, cuda=cuda)
        if writer:
            writer.add_scalar("Loss/Eval", val_loss, epoch)
            writer.add_scalar("Metric/Eval", val_metric, epoch)
        is_best = val_loss <= best_val_loss

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_summary = str(summary(model))

            if save_dir is not None:
                save_dir = Path(save_dir)
                best_summary_file = save_dir / "best_model_summary.txt"
                with open(best_summary_file, "w") as f:
                    f.write(best_model_summary)
                save_checkpoint(state=model, is_best=True, save_dir=save_dir)

        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Metric: {val_metric:.4f}")

    last_model_summary = str(summary(model))
    if save_dir is not None:
        last_summary_file = save_dir / "last_model_summary.txt"
        with open(last_summary_file, "w") as f:
            f.write(last_model_summary)
        save_checkpoint(state=model, is_best=False, save_dir=save_dir)

    logging.info(f"Training completed. Best Loss: {best_val_loss:.4f} at epoch: {best_epoch}")

def train(model, optimizer, loss_fn, metric_fn, train_dl, cuda=False):
    """Single epoch training loop.

    Args:
        model (nn.Module): Model to train.
        optimizer: Optimizer instance.
        loss_fn: Loss function.
        metric_fn: Metric function to compute monitoring metric.
        train_dl (DataLoader): Training dataloader.
        cuda (bool): Whether to move batches to CUDA.

    Returns:
        tuple: `(avg_loss, avg_metric)` for the epoch.
    """
    model.train()

    train_loss = RunningAverage()
    train_metric = RunningAverage()
    
    for train_batch, target_batch in train_dl:
        if cuda:
            train_batch, target_batch = train_batch.cuda(non_blocking=True), target_batch.cuda(non_blocking=True)
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, target_batch)
        metric = metric_fn(output_batch, target_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item())
        train_metric.update(metric.item())
    return train_loss(), train_metric()

def evaluate(model, loss_fn, metric_fn, val_dl, cuda=False):
    """Evaluate `model` on `val_dl` without gradient updates.

    Args:
        model (nn.Module): Model to evaluate.
        loss_fn: Loss function.
        metric_fn: Metric function.
        val_dl (DataLoader): Validation dataloader.
        cuda (bool): Whether to move batches to CUDA.

    Returns:
        tuple: `(avg_loss, avg_metric)` over the validation set.
    """
    model.eval()

    val_loss = RunningAverage()
    val_metric = RunningAverage()
    for val_batch, target_batch in val_dl:
        if cuda:
            val_batch, target_batch = val_batch.cuda(non_blocking=True), target_batch.cuda(non_blocking=True)
        output_batch = model(val_batch)
        loss = loss_fn(output_batch, target_batch)
        metric = metric_fn(output_batch, target_batch)
        val_loss.update(loss.item())
        val_metric.update(metric.item())
    return val_loss(), val_metric()

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)