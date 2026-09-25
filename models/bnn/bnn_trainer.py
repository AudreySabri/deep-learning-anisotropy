import logging

import torch
import pyro

def SVITrainer(svi, dataloader, num_epochs, device, writer=None):
    """Train a Pyro model using Stochastic Variational Inference (SVI).

    Args:
        svi: A configured Pyro SVI instance.
        dataloader: Iterable of (input, target) batches used for training.
        num_epochs (int): Number of training epochs.
        device (str or torch.device): Device to run training on.
        writer (optional): TensorBoard `SummaryWriter` for logging metrics.

    Returns:
        None

    Side effects:
        Logs per-epoch metrics and updates the Pyro parameter store.
    """
    pyro.clear_param_store()
    losses = []
    mae_scores = []
    last_loss = None
    last_mae = None
    best_loss = float('inf')
    best_mae = float('inf')
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_mae = 0.0
        for x_train, y_train in dataloader:
            x_train, y_train = x_train.to(device), y_train.to(device)
            loss = svi.step(x_train, y_train)
            with torch.no_grad():
                guide_trace = pyro.poutine.trace(svi.guide).get_trace(x_train, y_train)
                model_trace = pyro.poutine.replay(svi.model, trace=guide_trace)
                mu = model_trace(x_train, None)
                mae = torch.abs(mu - y_train).mean()

            epoch_loss += loss
            epoch_mae += mae.item()

        avg_loss = epoch_loss / len(dataloader)
        avg_mae = epoch_mae / len(dataloader)   

        losses.append(avg_loss)
        mae_scores.append(avg_mae)

        if writer:
            writer.add_scalar("Loss/Train", avg_loss, epoch)
            writer.add_scalar("MAE/Train", avg_mae, epoch)

        last_loss = avg_loss
        last_mae = avg_mae
        if avg_loss < best_loss:
            best_loss = avg_loss
        if avg_mae < best_mae:
            best_mae = avg_mae

        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {last_loss:.4f}, MAE: {last_mae:.4f}")
    logging.info(f"Training completed. Best Loss: {best_loss:.4f}, Best MAE: {best_mae:.4f}")