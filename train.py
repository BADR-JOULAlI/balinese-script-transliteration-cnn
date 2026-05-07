"""Command-line training entry point for Balinese script classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L

from src.data import BalineseDataModule
from src.models.callbacks import build_training_callbacks
from src.models.lightning_module import BalineseClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN for Balinese script recognition.")
    parser.add_argument(
        "--backbone",
        choices=["vgg16", "resnet50", "inception_v3"],
        default="resnet50",
        help="Torchvision backbone to fine-tune.",
    )
    parser.add_argument("--data-dir", default="data/raw", help="Path to the ImageFolder dataset.")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints", help="Directory for model checkpoints.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for all dataloaders.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--max-epochs", type=int, default=30, help="Maximum number of training epochs.")
    parser.add_argument("--accelerator", default="auto", help="Lightning accelerator, e.g. auto, cpu, gpu.")
    parser.add_argument("--devices", default="auto", help="Lightning devices argument, e.g. auto, 1, 2.")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(42, workers=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data_module = BalineseDataModule(
        data_dir=args.data_dir,
        backbone=args.backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42,
    )
    data_module.setup()
    data_module.print_class_summary()

    model = BalineseClassifier(
        num_classes=data_module.num_classes,
        backbone=args.backbone,
        learning_rate=args.learning_rate,
        pretrained=not args.no_pretrained,
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=build_training_callbacks(checkpoint_dir),
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()