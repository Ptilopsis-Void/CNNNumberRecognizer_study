from __future__ import annotations

from math import ceil, sqrt
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from PIL import Image

from .data import MNIST_MEAN, MNIST_STD


def show_sample_images(
    data_loader: object,
    *,
    count: int = 8,
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    images, labels = next(iter(data_loader))
    count = max(1, min(count, len(images)))
    columns = ceil(sqrt(count))
    rows = ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)

    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= count:
            continue
        image = images[index].squeeze().mul(MNIST_STD[0]).add(MNIST_MEAN[0]).clamp(0, 1)
        axis.imshow(image.cpu().numpy(), cmap="gray")
        axis.set_title(f"Label: {labels[index].item()}")

    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def plot_training_history(
    train_losses: Sequence[float],
    train_accuracies: Sequence[float],
    test_accuracies: Sequence[float],
    *,
    output_path: Path,
    show: bool = False,
) -> None:
    epochs = range(1, len(train_losses) + 1)
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4.5))

    loss_axis.plot(epochs, train_losses, marker="o")
    loss_axis.set(title="Training Loss", xlabel="Epoch", ylabel="Loss")
    loss_axis.grid(alpha=0.3)

    accuracy_axis.plot(epochs, train_accuracies, marker="o", label="Train")
    accuracy_axis.plot(epochs, test_accuracies, marker="o", label="Test")
    accuracy_axis.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    accuracy_axis.grid(alpha=0.3)
    accuracy_axis.legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def show_prediction_image(
    processed_image: Image.Image,
    *,
    digit: int,
    confidence: float,
) -> None:
    """Display the centered 28 x 28 image used by the model."""

    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(processed_image, cmap="gray")
    axis.set_title(f"Prediction: {digit} ({confidence * 100:.2f}%)")
    axis.axis("off")
    figure.tight_layout()
    plt.show()
    plt.close(figure)
