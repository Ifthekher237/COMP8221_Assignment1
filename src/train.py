from pathlib import Path
import time

import torch
import torch.nn.functional as F


def noise_prediction_loss(model, x_start, q_sample_fn, timesteps):
    """
    Compute simplified DDPM/DDIM-style noise prediction MSE loss.

    This samples a random timestep, adds Gaussian noise with q_sample_fn,
    predicts that noise with the model, and returns the MSE objective.
    """
    model_device = next(model.parameters()).device
    x_start = x_start.to(model_device)
    batch_size = x_start.shape[0]
    t = torch.randint(0, timesteps, (batch_size,), device=x_start.device).long()
    noise = torch.randn_like(x_start)
    x_noisy = q_sample_fn(x_start=x_start, t=t, noise=noise)
    predicted_noise = model(x_noisy, t)
    return F.mse_loss(predicted_noise, noise)


def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, config=None, path=None):
    """Save model, optimizer, training history, and configuration."""
    if path is None:
        path = config
        config = None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "config": config,
        },
        path,
    )


def load_checkpoint(model, optimizer, path, map_location):
    """Load a checkpoint if present and restore model and optimizer state."""
    path = Path(path)
    if not path.exists():
        print(f"No checkpoint found at: {path}")
        return None

    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded checkpoint from: {path}")
    return checkpoint


def run_validation_loss(model, val_loader, loss_fn, device, max_batches=20):
    """Compute average validation loss over a limited number of batches."""
    was_training = model.training
    model.eval()
    losses = []

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            images = images.to(device)
            loss = loss_fn(model, images)
            losses.append(loss.item())

    if was_training:
        model.train()

    return sum(losses) / len(losses)


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
    max_batches=None,
    grad_clip=1.0,
):
    """Train for one epoch or a limited number of batches and return average loss."""
    model.train()
    losses = []

    for batch_idx, (images, _) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, images)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        losses.append(loss.item())

    return sum(losses) / len(losses)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    num_epochs,
    save_checkpoint_fn=None,
    checkpoint_path=None,
    train_losses=None,
    val_losses=None,
    max_train_batches=None,
    max_val_batches=20,
):
    """Train the model while tracking train and validation losses."""
    if train_losses is None:
        train_losses = []
    if val_losses is None:
        val_losses = []

    for epoch in range(num_epochs):
        start_time = time.perf_counter()

        avg_train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            max_batches=max_train_batches,
        )
        avg_val_loss = run_validation_loss(
            model,
            val_loader,
            loss_fn,
            device,
            max_batches=max_val_batches,
        )
        elapsed_time = time.perf_counter() - start_time

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Time: {elapsed_time:.2f} sec"
        )

        if save_checkpoint_fn is not None and checkpoint_path is not None:
            save_checkpoint_fn(
                model,
                optimizer,
                epoch + 1,
                train_losses,
                val_losses,
                checkpoint_path,
            )

    return train_losses, val_losses
