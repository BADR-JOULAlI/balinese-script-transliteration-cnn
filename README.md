# Balinese Script Transliteration CNN

A research-oriented Deep Learning project for Balinese script image classification and transliteration using Convolutional Neural Networks with PyTorch Lightning.

This repository follows a clean Research Engineer structure: raw data is kept local, experiments are isolated in notebooks, reusable training code lives under `src/`, model artifacts are excluded from Git, and reports/figures are separated from implementation code.

## Project Overview

The goal of this project is to build a reproducible computer vision pipeline capable of recognizing Balinese script characters from image data and supporting transliteration workflows.

The current implementation provides:

1. A PyTorch Lightning `BalineseDataModule` based on `torchvision.datasets.ImageFolder`.
2. Leakage-safe, source-grouped train/validation/test splits for DeepLontar.
3. ImageNet-style preprocessing for transfer learning backbones.
4. Configurable image sizing, including the evaluated ResNet18 `128x128` baseline.
5. A dynamic `BalineseClassifier` LightningModule whose output layer is built from the detected number of classes.
6. Reproducible preprocessing, training, evaluation, metrics, and executed notebooks.

## Structure

```text
balinese-script-transliteration-cnn/
├── data/                             # Local datasets, ignored by Git
├── models/checkpoints/               # Local checkpoints, ignored by Git
├── notebooks/
│   ├── 01_data_ingestion.ipynb       # YOLO ingestion, grouping and preprocessing
│   ├── 02_model_training.ipynb       # ResNet18 training and learning curves
│   └── 03_evaluation.ipynb           # Test metrics, plots and inference demo
├── reports/
│   ├── training_history.json         # Recorded learning curves
│   └── test_metrics.json             # Per-class metrics and confusion matrix
├── src/
│   ├── data/datamodule.py            # BalineseDataModule
│   └── models/
│       ├── callbacks.py              # Checkpoint and early stopping
│       └── lightning_module.py       # BalineseClassifier
├── prepare_deeplontar.py             # YOLO-to-classification conversion
├── train.py                          # Training CLI
├── evaluate.py                       # Evaluation CLI
├── requirements.txt                  # Runtime and training dependencies
└── requirements-notebook.txt         # Optional notebook dependencies
```

## Tech Stack

The project dependencies are pinned in `requirements.txt`:

- `lightning==2.5.1`
- `torch==2.9.1`
- `torchvision==0.24.1`
- `kaggle==1.6.17`
- `pandas==2.2.3`
- `matplotlib==3.10.0`
- `pillow>=11.0,<13`

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd balinese-script-transliteration-cnn
```

### 2. Create a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For the executed notebooks:

```bash
pip install -r requirements-notebook.txt
```

## Data Layout

The dataset must follow the `ImageFolder` convention:

```text
data/raw/
├── class_1/
│   ├── image_001.png
│   └── image_002.png
├── class_2/
│   ├── image_001.png
│   └── image_002.png
└── ...
```

The entire `data/` directory is ignored by Git to prevent large datasets and local Kaggle downloads from being pushed.

### DeepLontar preparation

Download `DeepLontar.zip` and `DeepLontar_Labels.zip` from the official
[Figshare record](https://doi.org/10.6084/m9.figshare.20103803.v2), extract them
under `data/deeplontar/images` and `data/deeplontar/labels`, then run:

```bash
python prepare_deeplontar.py
```

The converter crops YOLO boxes, retains the declared class IDs, groups the
original and enhanced copies of each manuscript page, and creates leakage-safe
`train/val/test` folders. Enhanced copies are used only for training.

The executed walkthrough is split by responsibility:

1. [`01_data_ingestion.ipynb`](notebooks/01_data_ingestion.ipynb)
2. [`02_model_training.ipynb`](notebooks/02_model_training.ipynb)
3. [`03_evaluation.ipynb`](notebooks/03_evaluation.ipynb)

## DataModule Usage

```python
from src.data import BalineseDataModule

# Uses 224x224 by default for ResNet50/VGG16.
data_module = BalineseDataModule(data_dir="data/raw", backbone="resnet50", batch_size=32)
data_module.setup()
data_module.print_class_summary()

# Uses 299x299 automatically for InceptionV3.
inception_data_module = BalineseDataModule(data_dir="data/raw", backbone="inception_v3", batch_size=32)
```

## Model Usage

```python
import lightning as L

from src.data import BalineseDataModule
from src.models import BalineseClassifier, build_training_callbacks

data_module = BalineseDataModule(
    data_dir="data/processed", backbone="resnet18", image_size=128, batch_size=128
)
data_module.setup()

model = BalineseClassifier(
    num_classes=data_module.num_classes,
    backbone="resnet18",
    learning_rate=3e-4,
)

trainer = L.Trainer(
    max_epochs=8,
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    callbacks=build_training_callbacks("models/checkpoints"),
)
trainer.fit(model, datamodule=data_module)
```

GPU training and evaluation example:

```bash
python train.py --data-dir data/processed --backbone resnet18 --image-size 128 \
  --batch-size 128 --max-epochs 8 --accelerator gpu --devices 1
python evaluate.py --checkpoint models/checkpoints/resnet18-128/<best-checkpoint>.ckpt
```

## Baseline Result

The current ResNet18 baseline was trained on an RTX 5070 Laptop GPU with
128x128 inputs and mixed precision. Evaluation uses 6,706 original character
crops from manuscript groups that never occur in training.

| Metric | Result |
|---|---:|
| Accuracy | 96.76% |
| Macro precision | 92.74% |
| Macro recall | 93.56% |
| Macro F1 | 92.91% |

The complete per-class metrics and confusion matrix are stored in
[`reports/test_metrics.json`](reports/test_metrics.json). Rare classes remain
the main limitation; they need additional independent manuscript samples.

## Reproducibility Notes

To keep experiments reproducible:

- Keep dependency versions pinned in `requirements.txt`.
- Use the `seed` argument in `BalineseDataModule` for deterministic dataset splits.
- Record dataset versions and preprocessing assumptions in notebooks or reports.
- Save training curves, confusion matrices, and comparison plots in `reports/figures/`.
- Avoid committing local datasets, checkpoints, `.h5`, `.ckpt`, `.pt`, or `.pth` files.

## Git Ignore Policy

This project intentionally excludes:

- `data/`
- model weights and checkpoints such as `.h5`, `.ckpt`, `.pt`, `.pth`
- Python cache directories
- Jupyter notebook checkpoints
- virtual environments
- editor and OS metadata

This keeps the repository lightweight and suitable for collaboration.

## License

See [LICENSE](LICENSE) for details.
