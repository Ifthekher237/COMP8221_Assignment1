from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from torchvision.utils import make_grid


def unnormalize_to_01(x):
    """Convert image tensors from [-1, 1] to [0, 1] for visualization."""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def save_loss_history_csv(train_losses, val_losses, output_path):
    """Save epoch, training loss, and validation loss history as a CSV file."""
    if len(train_losses) == 0 or len(val_losses) == 0:
        raise ValueError("train_losses and val_losses must not be empty.")
    if len(train_losses) != len(val_losses):
        raise ValueError("train_losses and val_losses must have the same length.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history = pd.DataFrame({
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses,
    })
    history.to_csv(output_path, index=False)
    return output_path


def plot_loss_curve(train_losses, val_losses, output_path):
    """Plot and save the training and validation loss curve."""
    if len(train_losses) == 0 or len(val_losses) == 0:
        raise ValueError("train_losses and val_losses must not be empty.")
    if len(train_losses) != len(val_losses):
        raise ValueError("train_losses and val_losses must have the same length.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Training Loss")
    plt.plot(epochs, val_losses, marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    return output_path


def save_image_grid(images, output_path, title=None, nrow=4, figsize=(8, 8)):
    """Save a grid of image tensors expected to already be in [0, 1]."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = images.detach().cpu().clamp(0.0, 1.0)
    grid = make_grid(images, nrow=nrow, padding=2)

    plt.figure(figsize=figsize)
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    return output_path


def save_reverse_diffusion_grid(
    intermediates,
    output_path,
    title="Reverse DDIM Sampling Process: xT to x0",
    max_images=8,
):
    """Save a horizontal grid showing selected reverse diffusion intermediates."""
    if len(intermediates) == 0:
        raise ValueError("intermediates must contain at least one tensor.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_images = min(max_images, len(intermediates))
    if num_images == 1:
        selected_indices = [0]
    else:
        selected_indices = [
            round(i * (len(intermediates) - 1) / (num_images - 1))
            for i in range(num_images)
        ]
        selected_indices = list(dict.fromkeys(selected_indices))
        if selected_indices[-1] != len(intermediates) - 1:
            selected_indices.append(len(intermediates) - 1)

    selected_images = []
    for index in selected_indices:
        image = intermediates[index].detach().cpu()
        if image.ndim == 4:
            image = image[0]
        selected_images.append(unnormalize_to_01(image))

    fig, axes = plt.subplots(1, len(selected_images), figsize=(14, 3))
    if len(selected_images) == 1:
        axes = [axes]

    for position, (ax, image, index) in enumerate(zip(axes, selected_images, selected_indices)):
        ax.imshow(image.permute(1, 2, 0).numpy())
        if index == 0:
            label = "start"
        elif index == len(intermediates) - 1:
            label = "final"
        else:
            label = f"step {position}"
        ax.set_title(label)
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    return output_path, selected_indices
