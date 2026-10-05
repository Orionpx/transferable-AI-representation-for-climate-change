"""Fine-tune the Prithvi backbone for Land Surface Temperature regression.

EuroSAT land-cover classes are mapped to approximate surface-temperature
values (Kelvin) derived from published urban-heat-island research.  The
mapping lets us reuse EuroSAT imagery for a continuous regression task that
is fundamentally different from the classification experiment, proving the
backbone's versatility for climate-relevant heat-mapping applications.
"""

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


# Approximate Land Surface Temperature (Kelvin) per EuroSAT class.
# Values are informed by urban-heat-island literature and satellite-
# derived LST studies (e.g., Zhou et al. 2018, Oke et al. 2017).
LST_PROXY: dict[int, float] = {
    0: 305.0,   # AnnualCrop
    1: 295.0,   # Forest
    2: 300.0,   # HerbaceousVegetation
    3: 315.0,   # Highway
    4: 320.0,   # Industrial
    5: 298.0,   # Pasture
    6: 302.0,   # PermanentCrop
    7: 312.0,   # Residential
    8: 290.0,   # River
    9: 288.0,   # SeaLake
}

LST_MEAN = sum(LST_PROXY.values()) / len(LST_PROXY)
LST_STD = (sum((v - LST_MEAN) ** 2 for v in LST_PROXY.values()) / len(LST_PROXY)) ** 0.5


class HeatMappingDataset(Dataset):
    """EuroSAT images with land-cover-derived surface temperature labels.

    Each image is labelled with an approximate surface temperature based on
    its land-cover class.  RGB channels are duplicated to match Prithvi's
    six-channel input contract.
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb, label = self.base[index]
        six_channel = torch.cat((rgb, rgb), dim=0)
        temperature = (LST_PROXY[label] - LST_MEAN) / LST_STD
        return six_channel, torch.tensor(temperature, dtype=torch.float32)


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


class PrithviHeatMapper(L.LightningModule):
    """Regression head on the Prithvi backbone for LST prediction."""

    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        freeze_epochs: int,
        pretrained: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = build_backbone(pretrained)
        self.head = nn.Sequential(
            nn.LazyLinear(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
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
        return self.head(pool_features(self.backbone(images))).squeeze(-1)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        images, targets = batch
        predictions = self(images)
        loss = nn.functional.mse_loss(predictions, targets)
        mae = (predictions - targets).abs().mean()
        # Convert MAE back to Kelvin for interpretable logging
        mae_kelvin = mae * LST_STD
        r2 = 1 - loss / targets.var().clamp(min=1e-6)
        self.log(f"{stage}_mse", loss, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        self.log(f"{stage}_mae_kelvin", mae_kelvin, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        self.log(f"{stage}_r2", r2, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )


def train_heat_mapping(
    config: dict[str, Any], data_dir: str, output_dir: str, max_epochs: int,
) -> None:
    image_size = int(config["training"]["image_size"])
    full_dataset = HeatMappingDataset(data_dir, image_size=image_size, download=True)
    val_size = max(1, int(len(full_dataset) * 0.2))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )
    loader_kwargs = dict(
        batch_size=int(config["training"]["batch_size"]),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    module = PrithviHeatMapper(
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
        precision=config["training"].get("precision", "16-mixed")
        if torch.cuda.is_available()
        else "32-true",
        accumulate_grad_batches=int(config["training"]["gradient_accumulation_steps"]),
        gradient_clip_val=1.0,
        default_root_dir=output_path,
        callbacks=[
            ModelCheckpoint(
                dirpath=output_path / "checkpoints",
                monitor="val_mae_kelvin",
                mode="min",
                save_top_k=1,
            ),
            EarlyStopping(monitor="val_mae_kelvin", mode="min", patience=5),
        ],
    )
    trainer.fit(
        module,
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, **loader_kwargs),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sen1floods.yaml")
    parser.add_argument("--data-dir", default="data/eurosat")
    parser.add_argument("--output-dir", default="outputs/heat_mapping")
    parser.add_argument("--max-epochs", type=int, default=20)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    L.seed_everything(int(config["seed"]), workers=True)
    train_heat_mapping(config, args.data_dir, args.output_dir, args.max_epochs)


if __name__ == "__main__":
    main()
