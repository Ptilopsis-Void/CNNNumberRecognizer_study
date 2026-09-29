from __future__ import annotations

import random

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a requested device and fail clearly when it is unavailable."""

    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请改用 --device cpu。")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device 必须是 auto、cpu 或 cuda。")
    return torch.device(requested)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

