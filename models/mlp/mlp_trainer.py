import logging
from pathlib import Path

import numpy as np
from torchinfo import summary

from utils import save_checkpoint

def predict(model, pred_dl, metric_fn, cuda=False):
    model.eval()

    predictions = []
    ground_truth = []
    tracking_metric = []

    for pred_batch, target_batch in pred_dl:
        if cuda:
            pred_batch, target_batch = pred_batch.cuda(non_blocking=True), target_batch.cuda(non_blocking=True)
        output_batch = model(pred_batch)
        metric = metric_fn(output_batch, target_batch)

        output_batch = output_batch.data.detach().numpy()
        target_batch = target_batch.data.detach().numpy()
        batch_metric = metric.item().detach().numpy()

        predictions.append(output_batch)
        ground_truth.append(target_batch)
        tracking_metric.append(batch_metric)

    return np.concatenate(predictions, axis=0), np.concatenate(ground_truth, axis=0), np.concatenate(tracking_metric, axis=0)

def train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metric_fn, num_epochs, save_dir=None, writer=None, cuda=False):

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