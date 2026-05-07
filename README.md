# Balinese Script Transliteration CNN

A research-oriented Deep Learning project for Balinese script image classification and transliteration using Convolutional Neural Networks with PyTorch Lightning.

This repository follows a clean Research Engineer structure: raw data is kept local, experiments are isolated in notebooks, reusable training code lives under `src/`, model artifacts are excluded from Git, and reports/figures are separated from implementation code.

## Project Overview

The goal of this project is to build a reproducible computer vision pipeline capable of recognizing Balinese script characters from image data and supporting transliteration workflows.

The current implementation provides:

1. A PyTorch Lightning `BalineseDataModule` based on `torchvision.datasets.ImageFolder`.
2. Reproducible train/validation/test splitting with an 80/10/10 ratio.
3. ImageNet-style preprocessing for transfer learning backbones.
4. Dynamic image sizing for ResNet/VGG (`224x224`) and InceptionV3 (`299x299`).
5. A dynamic `BalineseClassifier` LightningModule whose output layer is built from the detected number of classes.
6. Recommended Lightning callbacks for checkpointing and early stopping.

## Structure

```text
balinese-script-transliteration-cnn/
├── data/
│   ├── raw/                         # Local raw images, ignored by Git
│   └── processed/                   # Local processed datasets, ignored by Git
├── models/
│   └── checkpoints/                 # Local model checkpoints, ignored by Git
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── reports/
│   └── figures/                     # Evaluation plots and training figures
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── datamodule.py            # BalineseDataModule
│   ├── models/
│   │   ├── __init__.py
│   │   ├── callbacks.py             # ModelCheckpoint and EarlyStopping helpers
│   │   └── lightning_module.py      # BalineseClassifier LightningModule
│   └── utils/
│       └── data_loader.py           # Utility placeholder
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Tech Stack

The project dependencies are pinned in `requirements.txt`:

- `lightning==2.5.1`
- `torch==2.6.0`
- `torchvision==0.21.0`
- `kaggle==1.6.17`
- `pandas==2.2.3`
- `matplotlib==3.10.0`
- `jupyter==1.1.1`

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
from src.models import build_model_from_datamodule, build_training_callbacks

data_module = BalineseDataModule(data_dir="data/raw", backbone="resnet50", batch_size=32)
data_module.setup()

model = build_model_from_datamodule(
    data_module=data_module,
    backbone="resnet50",
    learning_rate=1e-4,
)

trainer = L.Trainer(
    max_epochs=30,
    callbacks=build_training_callbacks("models/checkpoints"),
)
trainer.fit(model, datamodule=data_module)
```

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