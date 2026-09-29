from __future__ import annotations

from pathlib import Path

from .config import DEFAULT_SETTINGS, Settings


def _clean_dragged_path(raw_value: str) -> Path:
    """Normalize a path pasted or dragged into a Windows terminal."""

    value = raw_value.strip()
    if value.startswith("&"):
        value = value[1:].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return Path(value.strip()).expanduser()


def _ask_use_existing_model(model_path: Path) -> bool:
    """Ask whether an existing checkpoint should be reused."""

    print(f"检测到已有模型：{model_path}")
    while True:
        choice = input("是否直接使用已有模型？(Y/n)：").strip().lower()
        if choice in {"", "y", "yes", "是"}:
            return True
        if choice in {"n", "no", "否"}:
            return False
        print("请输入 y 或 n；直接按 Enter 默认使用已有模型。")


def _show_training_results(settings: Settings, history: dict[str, object]) -> None:
    from .data import get_data_loaders
    from .visualization import plot_training_history, show_sample_images

    print("\n训练完成，正在生成样本图和训练曲线……")
    train_loader, _ = get_data_loaders(settings, download=False)
    print("将显示 MNIST 样本图，关闭窗口后会继续显示训练曲线。")
    show_sample_images(
        train_loader,
        count=8,
        output_path=settings.plot_dir / "samples.png",
        show=True,
    )

    epoch_rows = history["epochs"]
    print("将显示训练曲线，关闭窗口后即可输入自己的图片。")
    plot_training_history(
        [row["train"]["loss"] for row in epoch_rows],
        [row["train"]["accuracy"] for row in epoch_rows],
        [row["test"]["accuracy"] for row in epoch_rows],
        output_path=settings.plot_dir / "training_history.png",
        show=True,
    )


def _print_prediction(prediction: object) -> None:
    ranking = sorted(
        enumerate(prediction.probabilities),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    print("\n" + "=" * 45)
    print(f"预测数字：{prediction.digit}")
    print(f"置信度：{prediction.confidence * 100:.2f}%")
    print(
        "候选结果："
        + ", ".join(
            f"{digit}={probability * 100:.2f}%" for digit, probability in ranking
        )
    )
    print("=" * 45)


def run_interactive_workflow(settings: Settings = DEFAULT_SETTINGS) -> int:
    """Train, visualize, then repeatedly predict user-supplied digit images."""

    from .prediction import load_model, predict_with_model
    from .training import train_model
    from .visualization import show_prediction_image

    print("=" * 58)
    print("CNN 手写数字识别：自动训练与交互预测")
    print("=" * 58)
    print("程序会先检查是否已经存在训练好的模型。\n")

    try:
        use_existing_model = settings.model_path.is_file() and _ask_use_existing_model(
            settings.model_path
        )
        if use_existing_model:
            print("已选择使用现有模型，本次跳过训练。")
        else:
            if settings.model_path.is_file():
                print("已选择重新训练模型，原模型将在训练完成后更新。")
            else:
                print("没有检测到已有模型，将自动开始训练。")
            print("训练时间取决于电脑性能，请耐心等待。\n")
            _, history = train_model(settings, show_samples=False, show_plots=False)
            _show_training_results(settings, history)
        model, device = load_model(settings)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"\n程序无法继续：{error}")
        return 1

    print("\n模型已经准备完成。")
    while True:
        raw_path = input(
            "请把手写数字图片拖到终端窗口中，然后按 Enter（直接按 Enter 退出）：\n> "
        )
        if not raw_path.strip():
            print("程序已结束。")
            return 0

        image_path = _clean_dragged_path(raw_path)
        try:
            prediction, processed_image = predict_with_model(
                model,
                device,
                image_path,
                invert="auto",
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            print(f"图片读取失败：{error}")
            print("请重新拖入一张 PNG、JPG 或 JPEG 图片。\n")
            continue

        _print_prediction(prediction)
        print("正在显示模型实际识别的 28 × 28 图片，关闭窗口后可继续。")
        show_prediction_image(
            processed_image,
            digit=prediction.digit,
            confidence=prediction.confidence,
        )

        continue_choice = input("是否继续预测另一张图片？(y/N)：").strip().lower()
        if continue_choice not in {"y", "yes", "是"}:
            print("预测完成，程序已结束。")
            return 0
