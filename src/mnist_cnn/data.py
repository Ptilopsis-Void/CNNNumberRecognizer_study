from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import Settings


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def get_data_loaders(
    settings: Settings,
    *,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Build deterministic MNIST training and test data loaders."""

    settings.ensure_directories()
    transform = build_transform()
    train_dataset = datasets.MNIST(
        root=str(settings.data_dir),
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=str(settings.data_dir),
        train=False,
        download=download,
        transform=transform,
    )

    generator = torch.Generator().manual_seed(settings.seed)
    loader_options = {
        "batch_size": settings.batch_size,
        "num_workers": settings.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if settings.num_workers > 0:
        loader_options["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_options,
    )
    return train_loader, test_loader

