from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from .config import DEFAULT_SETTINGS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CNN 手写数字识别")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练并评估模型")
    train_parser.add_argument("--epochs", type=int, default=DEFAULT_SETTINGS.epochs)
    train_parser.add_argument("--batch-size", type=int, default=DEFAULT_SETTINGS.batch_size)
    train_parser.add_argument("--learning-rate", type=float, default=DEFAULT_SETTINGS.learning_rate)
    train_parser.add_argument("--num-workers", type=int, default=DEFAULT_SETTINGS.num_workers)
    train_parser.add_argument("--seed", type=int, default=DEFAULT_SETTINGS.seed)
    train_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train_parser.add_argument("--show-samples", action="store_true")
    train_parser.add_argument("--show-plots", action="store_true")

    predict_parser = subparsers.add_parser("predict", help="预测一张手写数字图片")
    predict_parser.add_argument("image", type=Path)
    predict_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    predict_parser.add_argument("--invert", choices=["auto", "yes", "no"], default="auto")
    predict_parser.add_argument("--top-k", type=int, default=3)
    predict_parser.add_argument("--show", action="store_true", help="显示归一化前的处理结果")

    demo_parser = subparsers.add_parser("demo", help="保存或显示 MNIST 样本")
    demo_parser.add_argument("--count", type=int, default=8)
    demo_parser.add_argument("--show", action="store_true")
    demo_parser.add_argument("--batch-size", type=int, default=DEFAULT_SETTINGS.batch_size)
    return parser


def _run_train(args: argparse.Namespace) -> int:
    from .training import train_model

    settings = DEFAULT_SETTINGS.with_overrides(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )
    train_model(settings, show_samples=args.show_samples, show_plots=args.show_plots)
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    from .prediction import predict_image

    if not 1 <= args.top_k <= DEFAULT_SETTINGS.num_classes:
        raise ValueError("top-k 必须在 1 到 10 之间。")
    settings = DEFAULT_SETTINGS.with_overrides(device=args.device)
    prediction, processed_image = predict_image(args.image, settings, invert=args.invert)
    ranking = sorted(
        enumerate(prediction.probabilities),
        key=lambda item: item[1],
        reverse=True,
    )[: args.top_k]

    print(f"预测数字：{prediction.digit}")
    print(f"置信度：{prediction.confidence * 100:.2f}%")
    print("候选结果：" + ", ".join(f"{digit}={probability * 100:.2f}%" for digit, probability in ranking))
    if args.show:
        from .visualization import show_prediction_image

        show_prediction_image(
            processed_image,
            digit=prediction.digit,
            confidence=prediction.confidence,
        )
    return 0


def _run_demo(args: argparse.Namespace) -> int:
    from .data import get_data_loaders
    from .visualization import show_sample_images

    settings = DEFAULT_SETTINGS.with_overrides(batch_size=max(args.batch_size, args.count))
    train_loader, _ = get_data_loaders(settings)
    destination = settings.plot_dir / "samples.png"
    show_sample_images(train_loader, count=args.count, output_path=destination, show=args.show)
    print(f"样本图已保存：{destination}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if not arguments:
        from .interactive import run_interactive_workflow

        return run_interactive_workflow()

    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        if args.command == "train":
            return _run_train(args)
        if args.command == "predict":
            return _run_predict(args)
        if args.command == "demo":
            return _run_demo(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(1, f"错误：{error}\n")
    parser.error(f"未知命令：{args.command}")
    return 2
