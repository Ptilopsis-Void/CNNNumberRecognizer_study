from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class Settings:
    """Central runtime configuration with paths anchored to the project root."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    model_path: Path = PROJECT_ROOT / "artifacts" / "models" / "cnn_model.pth"
    history_path: Path = PROJECT_ROOT / "artifacts" / "training_history.json"
    plot_dir: Path = PROJECT_ROOT / "artifacts" / "plots"
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 10
    num_classes: int = 10
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    log_interval: int = 100

    def with_overrides(self, **changes: object) -> "Settings":
        return replace(self, **changes)

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_SETTINGS = Settings()

