"""Evaluate a trained checkpoint with imbalance-aware classification metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data import BalineseDataModule
from src.models.lightning_module import BalineseClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("reports/test_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = BalineseDataModule(
        args.data_dir, backbone="resnet18", image_size=args.image_size,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    data.setup()
    model = BalineseClassifier.load_from_checkpoint(args.checkpoint, map_location=device).to(device).eval()
    confusion = torch.zeros(data.num_classes, data.num_classes, dtype=torch.int64)
    with torch.inference_mode():
        for images, labels in data.test_dataloader():
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            indices = labels * data.num_classes + predictions
            confusion += torch.bincount(indices, minlength=data.num_classes**2).reshape_as(confusion)

    true_positive = confusion.diag().float()
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    recall = true_positive / support.clamp_min(1)
    precision = true_positive / predicted.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    per_class = [
        {
            "dataset_index": index, "class_name": name, "support": int(support[index]),
            "precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]),
        }
        for index, name in enumerate(data.class_names)
    ]
    result = {
        "checkpoint": str(args.checkpoint),
        "samples": int(support.sum()),
        "accuracy": float(true_positive.sum() / support.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"per_class", "confusion_matrix"}}, indent=2))


if __name__ == "__main__":
    main()
