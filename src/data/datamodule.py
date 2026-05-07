"""PyTorch Lightning DataModule for Balinese script image datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightning as L
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


class BalineseDataModule(L.LightningDataModule):
    """Load Balinese character images with a reproducible 80/10/10 split.

    Expected ImageFolder layout:

    data/raw/
    ├── class_1/
    │   ├── image_001.png
    │   └── ...
    ├── class_2/
    │   ├── image_001.png
    │   └── ...
    └── ...
    """

    def __init__(
        self,
        data_dir: str | Path = "data/raw",
        backbone: str = "resnet50",
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.backbone = backbone.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # InceptionV3 was trained with 299x299 inputs; VGG16/ResNet50 use 224x224.
        self.target_size = 299 if self.backbone == "inception_v3" else 224

        self.train_dataset: Optional[Subset] = None
        self.val_dataset: Optional[Subset] = None
        self.test_dataset: Optional[Subset] = None
        self.class_names: list[str] = []
        self.num_classes = 0

    @property
    def train_transforms(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(self.target_size, scale=(0.8, 1.0)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @property
    def eval_transforms(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((self.target_size, self.target_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.data_dir}. "
                "Expected class folders under data/raw/."
            )

        base_dataset = datasets.ImageFolder(root=self.data_dir)
        train_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.train_transforms)
        eval_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.eval_transforms)

        self.class_names = base_dataset.classes
        self.num_classes = len(self.class_names)

        total_size = len(base_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        generator = torch.Generator().manual_seed(self.seed)
        train_split, val_split, test_split = random_split(
            range(total_size),
            [train_size, val_size, test_size],
            generator=generator,
        )

        self.train_dataset = Subset(train_dataset, train_split.indices)
        self.val_dataset = Subset(eval_dataset, val_split.indices)
        self.test_dataset = Subset(eval_dataset, test_split.indices)

    def train_dataloader(self) -> DataLoader:
        return self._build_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._build_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._build_dataloader(self.test_dataset, shuffle=False)

    def _build_dataloader(self, dataset: Optional[Subset], shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup() before requesting a dataloader.")

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def print_class_summary(self) -> None:
        """Print the number of detected classes and their folder names."""
        if not self.class_names:
            dataset = datasets.ImageFolder(root=self.data_dir)
            self.class_names = dataset.classes
            self.num_classes = len(self.class_names)

        print(f"Detected classes: {self.num_classes}")
        for class_index, class_name in enumerate(self.class_names):
            print(f"{class_index}: {class_name}")