from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps
import torch
from torchvision import transforms

from .config import DEFAULT_SETTINGS, Settings
from .data import MNIST_MEAN, MNIST_STD
from .model import CNN, create_model
from .runtime import resolve_device


InvertMode = Literal["auto", "yes", "no"]


@dataclass(frozen=True, slots=True)
class Prediction:
    digit: int
    confidence: float
    probabilities: tuple[float, ...]


def load_model(settings: Settings = DEFAULT_SETTINGS) -> tuple[CNN, torch.device]:
    model_path = settings.model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到模型文件：{model_path}。请先运行 train。")

    device = resolve_device(settings.device)
    model = create_model(settings.num_classes).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    state_dict = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def _border_mean(image: Image.Image) -> float:
    width, height = image.size
    border = (
        image.crop((0, 0, width, 1)).tobytes()
        + image.crop((0, height - 1, width, height)).tobytes()
        + image.crop((0, 0, 1, height)).tobytes()
        + image.crop((width - 1, 0, width, height)).tobytes()
    )
    return sum(border) / max(1, len(border))


def prepare_image(
    image_path: str | Path,
    *,
    invert: InvertMode = "auto",
) -> tuple[torch.Tensor, Image.Image]:
    """Convert an arbitrary digit image to a centered MNIST-style tensor."""

    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"找不到图片：{path}")
    if invert not in {"auto", "yes", "no"}:
        raise ValueError("invert 必须是 auto、yes 或 no。")

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")

    should_invert = invert == "yes" or (invert == "auto" and _border_mean(image) > 127)
    if should_invert:
        image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)

    foreground = image.point(lambda value: 255 if value > 20 else 0)
    bounding_box = foreground.getbbox()
    if bounding_box is None:
        raise ValueError("图片中没有检测到有效笔画。")

    digit = image.crop(bounding_box)
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    offset = ((28 - digit.width) // 2, (28 - digit.height) // 2)
    canvas.paste(digit, offset)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )
    return transform(canvas).unsqueeze(0), canvas


def predict_image(
    image_path: str | Path,
    settings: Settings = DEFAULT_SETTINGS,
    *,
    invert: InvertMode = "auto",
) -> tuple[Prediction, Image.Image]:
    model, device = load_model(settings)
    return predict_with_model(model, device, image_path, invert=invert)


def predict_with_model(
    model: CNN,
    device: torch.device,
    image_path: str | Path,
    *,
    invert: InvertMode = "auto",
) -> tuple[Prediction, Image.Image]:
    """Predict one image with an already loaded model."""

    image_tensor, processed_image = prepare_image(image_path, invert=invert)
    image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        probabilities = torch.softmax(model(image_tensor), dim=1).squeeze(0).cpu()
    digit = int(probabilities.argmax().item())
    return (
        Prediction(
            digit=digit,
            confidence=float(probabilities[digit].item()),
            probabilities=tuple(float(value) for value in probabilities.tolist()),
        ),
        processed_image,
    )
