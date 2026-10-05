"""Fine-tune the Prithvi EO 2.0 backbone for EuroSAT classification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from terratorch.datasets import HLSBands
from terratorch.registry import BACKBONE_REGISTRY


class SixChannelEuroSAT(Dataset[tuple[torch.Tensor, int]]):
    """Use RGB EuroSAT while preserving Prithvi's six-band input contract.

    EuroSAT's torchvision release is RGB. We repeat RGB as two triplets so the
    backbone receives six channels; this is a controlled RGB transfer baseline,
    not a claim that RGB contains Sentinel-2 spectral information.
    """

    def __init__(self, root: str, image_size: int, download: bool = True) -> None:
        self.base = datasets.EuroSAT(
            root=root,
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]),
            download=download,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rgb, label = self.base[index]
        return torch.cat((rgb, rgb), dim=0), label


def build_backbone(pretrained: bool) -> nn.Module:
    return BACKBONE_REGISTRY.build(
        "prithvi_eo_v2_100_tl",
        pretrained=pretrained,
        bands=[
            HLSBands.BLUE,
            HLSBands.GREEN,
            HLSBands.RED,
            HLSBands.NIR_NARROW,
            HLSBands.SWIR_1,
            HLSBands.SWIR_2,
        ],
    )


def pool_features(features: Any) -> torch.Tensor:
    if isinstance(features, dict):
        features = list(features.values())[-1]
    elif isinstance(features, (list, tuple)):
        features = features[-1]
    if features.ndim == 3:
        return features.mean(dim=1)
    if features.ndim == 4:
        return features.mean(dim=(2, 3))
    raise TypeError(f"Unexpected Prithvi feature shape: {features.shape}")


class PrithviEuroSATClassifier(L.LightningModule):
    def __init__(self, learning_rate: float, weight_decay: float, freeze_epochs: int, pretrained: bool) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = build_backbone(pretrained)
        self.head = nn.LazyLinear(10)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.freeze_epochs = freeze_epochs
        self.set_backbone_trainable(False)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.freeze_epochs:
            self.set_backbone_trainable(True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(pool_features(self.backbone(images)))

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = nn.functional.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        self.log(f"{stage}_accuracy", accuracy, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


def train_eurosat(config: dict[str, Any], data_dir: str, output_dir: str, max_epochs: int) -> None:
    image_size = int(config["training"]["image_size"])
    full_dataset = SixChannelEuroSAT(data_dir, image_size=image_size, download=True)
    validation_size = max(1, int(len(full_dataset) * 0.2))
    train_size = len(full_dataset) - validation_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )
    loader_kwargs = dict(
        batch_size=int(config["training"]["batch_size"]),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    module = PrithviEuroSATClassifier(
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        freeze_epochs=int(config["model"]["freeze_backbone_epochs"]),
        pretrained=bool(config["model"]["pretrained"]),
    )
    output_path = Path(output_dir)
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=config["training"].get("precision", "16-mixed") if torch.cuda.is_available() else "32-true",
        accumulate_grad_batches=int(config["training"]["gradient_accumulation_steps"]),
        gradient_clip_val=1.0,
        default_root_dir=output_path,
        callbacks=[
            ModelCheckpoint(dirpath=output_path / "checkpoints", monitor="val_accuracy", mode="max", save_top_k=1),
            EarlyStopping(monitor="val_accuracy", mode="max", patience=5),
        ],
    )
    trainer.fit(module, DataLoader(train_dataset, shuffle=True, **loader_kwargs), DataLoader(val_dataset, **loader_kwargs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sen1floods.yaml")
    parser.add_argument("--data-dir", default="data/eurosat")
    parser.add_argument("--output-dir", default="outputs/eurosat")
    parser.add_argument("--max-epochs", type=int, default=20)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    L.seed_everything(int(config["seed"]), workers=True)
    train_eurosat(config, args.data_dir, args.output_dir, args.max_epochs)


if __name__ == "__main__":
    main()
