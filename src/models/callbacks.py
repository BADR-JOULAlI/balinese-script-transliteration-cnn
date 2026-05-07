"""Training callbacks for PyTorch Lightning experiments."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def build_training_callbacks(checkpoint_dir: str | Path) -> list:
    """Build callbacks for best-checkpoint saving and overfitting control."""
    return [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
        ),
    ]