from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import torch
from torch import nn

from .config import DEFAULT_SETTINGS, Settings
from .data import get_data_loaders
from .model import CNN, create_model
from .runtime import resolve_device, set_random_seed
from .visualization import plot_training_history, show_sample_images


@dataclass(slots=True)
class Metrics:
    loss: float
    accuracy: float


def _run_training_epoch(
    model: nn.Module,
    data_loader: object,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
    log_interval: int,
) -> Metrics:
    model.train()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    for batch_index, (images, labels) in enumerate(data_loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        sample_count += batch_size

        if log_interval > 0 and batch_index % log_interval == 0:
            print(
                f"Epoch {epoch}/{total_epochs} | Batch {batch_index}/{len(data_loader)} "
                f"| Loss {loss.item():.4f}"
            )

    return Metrics(loss=loss_sum / sample_count, accuracy=100.0 * correct / sample_count)


def evaluate_model(
    model: nn.Module,
    data_loader: object,
    criterion: nn.Module,
    device: torch.device,
) -> Metrics:
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            batch_size = labels.size(0)
            loss_sum += criterion(logits, labels).item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size

    return Metrics(loss=loss_sum / sample_count, accuracy=100.0 * correct / sample_count)


def _save_checkpoint(
    model: nn.Module,
    destination: Path,
    *,
    epoch: int,
    accuracy: float,
    num_classes: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "test_accuracy": accuracy,
            "num_classes": num_classes,
        },
        temporary_path,
    )
    temporary_path.replace(destination)


def train_model(
    settings: Settings = DEFAULT_SETTINGS,
    *,
    show_samples: bool = False,
    show_plots: bool = False,
) -> tuple[CNN, dict[str, object]]:
    """Train the CNN, retain the best checkpoint, and persist reproducible metrics."""

    if settings.epochs < 1:
        raise ValueError("epochs 必须大于 0。")
    if settings.batch_size < 1:
        raise ValueError("batch_size 必须大于 0。")

    settings.ensure_directories()
    set_random_seed(settings.seed)
    device = resolve_device(settings.device)
    train_loader, test_loader = get_data_loaders(settings)

    if show_samples:
        show_sample_images(
            train_loader,
            count=8,
            output_path=settings.plot_dir / "samples.png",
            show=True,
        )

    model = create_model(settings.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    history: dict[str, object] = {
        "settings": {
            "batch_size": settings.batch_size,
            "learning_rate": settings.learning_rate,
            "epochs": settings.epochs,
            "seed": settings.seed,
            "device": str(device),
        },
        "epochs": [],
    }
    best_accuracy = -1.0
    started_at = time.perf_counter()

    print(f"开始训练 | device={device} | epochs={settings.epochs} | batch={settings.batch_size}")
    for epoch in range(1, settings.epochs + 1):
        train_metrics = _run_training_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=settings.epochs,
            log_interval=settings.log_interval,
        )
        test_metrics = evaluate_model(model, test_loader, criterion, device)
        epoch_record = {
            "epoch": epoch,
            "train": asdict(train_metrics),
            "test": asdict(test_metrics),
        }
        history["epochs"].append(epoch_record)

        print(
            f"Epoch {epoch}/{settings.epochs} | "
            f"train_loss={train_metrics.loss:.4f} | "
            f"train_acc={train_metrics.accuracy:.2f}% | "
            f"test_acc={test_metrics.accuracy:.2f}%"
        )
        if test_metrics.accuracy > best_accuracy:
            best_accuracy = test_metrics.accuracy
            _save_checkpoint(
                model,
                settings.model_path,
                epoch=epoch,
                accuracy=best_accuracy,
                num_classes=settings.num_classes,
            )

    history["best_test_accuracy"] = best_accuracy
    history["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
    settings.history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    epoch_rows = history["epochs"]
    plot_training_history(
        [row["train"]["loss"] for row in epoch_rows],
        [row["train"]["accuracy"] for row in epoch_rows],
        [row["test"]["accuracy"] for row in epoch_rows],
        output_path=settings.plot_dir / "training_history.png",
        show=show_plots,
    )
    print(f"最佳模型已保存：{settings.model_path}")
    return model, history

