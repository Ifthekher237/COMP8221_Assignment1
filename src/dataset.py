import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_cifar10_transforms():
    """Return train and test transforms that scale CIFAR-10 images to [-1, 1]."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return train_transform, test_transform


def get_cifar10_dataloaders(config, seed=42):
    """Create reproducible CIFAR-10 train, validation, and test DataLoaders."""
    train_transform, test_transform = get_cifar10_transforms()

    train_dataset_full = datasets.CIFAR10(
        root=config["data_dir"],
        train=True,
        download=True,
        transform=train_transform,
    )

    val_dataset_full = datasets.CIFAR10(
        root=config["data_dir"],
        train=True,
        download=True,
        transform=test_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=config["data_dir"],
        train=False,
        download=True,
        transform=test_transform,
    )

    train_size = int(config["train_val_split"] * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size

    train_dataset, _ = random_split(
        train_dataset_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    _, val_dataset = random_split(
        val_dataset_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": config["pin_memory"],
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
