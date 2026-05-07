"""LightningModule for Balinese script image classification."""

from __future__ import annotations

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class BalineseClassifier(L.LightningModule):
    """Transfer-learning classifier for Balinese script characters."""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        learning_rate: float = 1e-4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be greater than 0.")

        self.save_hyperparameters()
        self.num_classes = num_classes
        self.backbone = backbone.lower()
        self.learning_rate = learning_rate
        self.model = self._build_backbone(self.backbone, num_classes, pretrained)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(images)
        # InceptionV3 can return an InceptionOutputs object during training.
        return outputs.logits if hasattr(outputs, "logits") else outputs

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, accuracy = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, accuracy = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, accuracy = self._shared_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
        return loss, accuracy

    @staticmethod
    def _build_backbone(backbone: str, num_classes: int, pretrained: bool) -> nn.Module:
        if backbone == "vgg16":
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            model = models.vgg16(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
            return model

        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            return model

        if backbone == "inception_v3":
            weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
            model = models.inception_v3(weights=weights, aux_logits=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            if model.AuxLogits is not None:
                aux_features = model.AuxLogits.fc.in_features
                model.AuxLogits.fc = nn.Linear(aux_features, num_classes)
            return model

        raise ValueError("backbone must be one of: vgg16, resnet50, inception_v3")