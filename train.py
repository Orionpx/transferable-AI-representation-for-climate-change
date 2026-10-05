"""Train Prithvi on flood segmentation with Lightning and AMP."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn

from src.dataset import build_dataloaders
from src.model import DiceCrossEntropyLoss, PrithviFloodModel


class FloodLightningModule(L.LightningModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        model_config = config["model"]
        training_config = config["training"]
        self.model = PrithviFloodModel(
            backbone_name=model_config["backbone_name"],
            num_input_channels=len(config["bands"]["indices"]),
            num_classes=int(model_config["num_classes"]),
            decoder_channels=int(model_config["decoder_channels"]),
            pretrained=bool(model_config["pretrained"]),
            freeze_backbone=True,
        )
        self.loss_fn: nn.Module = DiceCrossEntropyLoss()
        self.freeze_backbone_epochs = int(model_config["freeze_backbone_epochs"])
        self.learning_rate = float(training_config["learning_rate"])
        self.backbone_learning_rate = float(training_config["backbone_learning_rate"])
        self.weight_decay = float(training_config["weight_decay"])

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.freeze_backbone_epochs:
            self.model.set_backbone_trainable(True)
            self.log("backbone_unfrozen", 1.0, prog_bar=True)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        logits = self.model(batch["image"])
        loss = self.loss_fn(logits, batch["mask"])
        predictions = logits.argmax(dim=1)
        intersection = ((predictions == 1) & (batch["mask"] == 1)).sum().float()
        union = ((predictions == 1) | (batch["mask"] == 1)).sum().float()
        predicted_positive = (predictions == 1).sum().float()
        actual_positive = (batch["mask"] == 1).sum().float()
        iou = intersection / (union + 1e-6)
        f1 = 2 * intersection / (predicted_positive + actual_positive + 1e-6)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, batch_size=batch["image"].shape[0])
        self.log(f"{stage}_mean_iou", iou, prog_bar=True, on_epoch=True, batch_size=batch["image"].shape[0])
        self.log(f"{stage}_f1_score", f1, prog_bar=True, on_epoch=True, batch_size=batch["image"].shape[0])
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        backbone_parameters = list(self.model.backbone.parameters())
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        decoder_parameters = [
            parameter for parameter in self.model.parameters() if id(parameter) not in backbone_ids
        ]
        return torch.optim.AdamW(
            [
                {"params": decoder_parameters, "lr": self.learning_rate},
                {"params": backbone_parameters, "lr": self.backbone_learning_rate},
            ],
            weight_decay=self.weight_decay,
        )


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sen1floods.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    L.seed_everything(int(config["seed"]), workers=True)
    train_loader, val_loader = build_dataloaders(config)
    module = FloodLightningModule(config)
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    trainer = L.Trainer(
        max_epochs=int(config["training"]["max_epochs"]),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=config["training"].get("precision", "16-mixed") if torch.cuda.is_available() else "32-true",
        accumulate_grad_batches=int(config["training"]["gradient_accumulation_steps"]),
        gradient_clip_val=1.0,
        log_every_n_steps=int(config["training"]["log_every_n_steps"]),
        callbacks=[
            ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_mean_iou", mode="max", save_top_k=1),
            EarlyStopping(monitor="val_mean_iou", mode="max", patience=int(config["training"]["early_stopping_patience"])),
        ],
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
