"""Convert DeepLontar YOLO annotations into a leakage-safe classification dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("data/deeplontar/images"))
    parser.add_argument("--labels", type=Path, default=Path("data/deeplontar/labels"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--max-class-id", type=int, default=54)
    parser.add_argument("--min-source-groups", type=int, default=3)
    return parser.parse_args()


def annotation_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_groups(labels_dir: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for label_path in labels_dir.glob("*.txt"):
        groups[annotation_hash(label_path)].append(label_path.stem)
    return groups


def is_original(stem: str) -> bool:
    return len(stem) >= 2 and stem[-1] in {"a", "b"} and stem[:-1].isdigit()


def classes_by_group(groups: dict[str, list[str]], labels_dir: Path, max_class_id: int) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for group_id, stems in groups.items():
        original = next(stem for stem in stems if is_original(stem))
        result[group_id] = {
            int(line.split()[0]) for line in (labels_dir / f"{original}.txt").read_text().splitlines()
            if line.strip() and 0 <= int(line.split()[0]) <= max_class_id
        }
    return result


def split_groups(
    groups: dict[str, list[str]], group_classes: dict[str, set[int]], valid_classes: set[int],
    seed: int, train_ratio: float, val_ratio: float,
) -> dict[str, str]:
    group_ids = sorted(groups)
    total = len(group_ids)
    train_end = round(total * train_ratio)
    val_end = train_end + round(total * val_ratio)
    rng = random.Random(seed)
    for _ in range(50_000):
        rng.shuffle(group_ids)
        assignment = {
            group_id: "train" if index < train_end else "val" if index < val_end else "test"
            for index, group_id in enumerate(group_ids)
        }
        coverage = {split: set() for split in ("train", "val", "test")}
        for group_id, split in assignment.items():
            coverage[split].update(group_classes[group_id])
        if all(valid_classes <= coverage[split] for split in coverage):
            return assignment
    raise RuntimeError("Could not find a group split covering every retained class.")


def crop_box(image: Image.Image, values: list[float], margin: float) -> Image.Image | None:
    _, x, y, width, height = values
    image_width, image_height = image.size
    box_width, box_height = width * image_width, height * image_height
    pad_x, pad_y = box_width * margin, box_height * margin
    left = max(0, round(x * image_width - box_width / 2 - pad_x))
    top = max(0, round(y * image_height - box_height / 2 - pad_y))
    right = min(image_width, round(x * image_width + box_width / 2 + pad_x))
    bottom = min(image_height, round(y * image_height + box_height / 2 + pad_y))
    if right - left < 2 or bottom - top < 2:
        return None
    return image.crop((left, top, right, bottom)).convert("RGB")


def main() -> None:
    args = parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    groups = build_groups(args.labels)
    group_classes = classes_by_group(groups, args.labels, args.max_class_id)
    source_frequency = Counter(class_id for classes in group_classes.values() for class_id in classes)
    valid_classes = {class_id for class_id, frequency in source_frequency.items() if frequency >= args.min_source_groups}
    split_by_group = split_groups(
        groups, group_classes, valid_classes, args.seed, args.train_ratio, args.val_ratio
    )
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    skipped = Counter()
    manifest_rows: list[dict[str, str | int]] = []

    for group_id, stems in groups.items():
        split = split_by_group[group_id]
        selected_stems = stems if split == "train" else [stem for stem in stems if is_original(stem)]
        for stem in selected_stems:
            image_path = args.images / f"{stem}.jpg"
            label_path = args.labels / f"{stem}.txt"
            if not image_path.exists():
                skipped["missing_image"] += 1
                continue
            with Image.open(image_path) as image:
                for box_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                    parts = line.split()
                    if len(parts) != 5:
                        skipped["invalid_annotation"] += 1
                        continue
                    values = [float(value) for value in parts]
                    class_id = int(values[0])
                    if class_id not in valid_classes:
                        skipped["out_of_schema_class"] += 1
                        continue
                    crop = crop_box(image, values, args.margin)
                    if crop is None:
                        skipped["empty_crop"] += 1
                        continue
                    class_name = f"class_{class_id:02d}"
                    destination_dir = args.output / split / class_name
                    destination_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"{stem.replace(' ', '_')}_{box_index:04d}.jpg"
                    destination = destination_dir / filename
                    crop.save(destination, quality=92)
                    counts[split][class_id] += 1
                    manifest_rows.append({
                        "path": destination.as_posix(), "split": split, "class_id": class_id,
                        "source": stem, "source_group": group_id,
                    })

    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "split", "class_id", "source", "source_group"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {
        "groups": len(groups),
        "samples": {split: sum(counter.values()) for split, counter in counts.items()},
        "class_counts": {split: dict(sorted(counter.items())) for split, counter in counts.items()},
        "skipped": dict(skipped),
        "retained_classes": sorted(valid_classes),
        "excluded_classes": sorted(set(source_frequency) - valid_classes),
        "policy": "augmented variants are train-only; validation/test contain originals only",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
