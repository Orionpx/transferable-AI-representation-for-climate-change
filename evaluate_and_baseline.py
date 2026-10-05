"""Evaluate Prithvi and train a six-channel U-Net baseline on Sen1Floods11."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import FloodSegmentationDataset
from src.model import DiceCrossEntropyLoss, PrithviFloodModel
from train import FloodLightningModule


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def dataset_from_config(config: dict[str, Any], split_key: str, augment: bool = False) -> FloodSegmentationDataset:
    root = Path(config["paths"]["data_root"])
    bands = config["bands"]
    return FloodSegmentationDataset(
        image_dir=root / config["paths"]["image_dir"],
        mask_dir=root / config["paths"]["mask_dir"],
        split_file=root / config["paths"][split_key],
        band_indices=bands["indices"],
        mean=bands["mean"],
        std=bands["std"],
        image_size=int(config["training"]["image_size"]),
        augment=augment,
    )


def binary_metrics(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    prediction = logits.argmax(dim=1)
    intersection = ((prediction == 1) & (target == 1)).sum().float()
    union = ((prediction == 1) | (target == 1)).sum().float()
    predicted_positive = (prediction == 1).sum().float()
    actual_positive = (target == 1).sum().float()
    iou = intersection / (union + 1e-6)
    f1 = 2 * intersection / (predicted_positive + actual_positive + 1e-6)
    return float(iou), float(f1)


def load_prithvi_checkpoint(config: dict[str, Any], checkpoint_path: str) -> nn.Module:
    module = FloodLightningModule(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    module.load_state_dict(state_dict, strict=True)
    return module.model


@torch.inference_mode()
def evaluate_prithvi(
    config: dict[str, Any],
    checkpoint_path: str,
    output_dir: str = "outputs/evaluation",
    save_examples: int = 12,
) -> dict[str, float]:
    """Evaluate a Prithvi checkpoint on the held-out test split and save 1x3 grids."""
    dataset = dataset_from_config(config, "test_split")
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_prithvi_checkpoint(config, checkpoint_path).to(device).eval()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ious: list[float] = []
    f1_scores: list[float] = []
    saved = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
        iou, f1 = binary_metrics(logits, masks)
        ious.append(iou)
        f1_scores.append(f1)

        predictions = logits.argmax(dim=1).cpu().numpy()
        image_array = images.cpu().numpy()
        mask_array = masks.cpu().numpy()
        for image, mask, prediction in zip(image_array, mask_array, predictions):
            if saved >= save_examples:
                break
            rgb = np.moveaxis(image[[2, 1, 0]], 0, -1)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
            figure, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(rgb)
            axes[0].set_title("RGB Input")
            axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Ground Truth")
            axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("Prediction")
            for axis in axes:
                axis.axis("off")
            figure.tight_layout()
            figure.savefig(output_path / f"example_{saved:04d}.png", dpi=150)
            plt.close(figure)
            saved += 1

    metrics = {"mean_iou": float(np.mean(ious)), "f1_score": float(np.mean(f1_scores))}
    with (output_path / "test_metrics.yaml").open("w", encoding="utf-8") as metrics_file:
        yaml.safe_dump(metrics, metrics_file)
    return metrics


class UNetLightningModule(L.LightningModule):
    """ResNet-34 U-Net baseline with a first layer configured for six channels."""

    def __init__(self, learning_rate: float, weight_decay: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=6,
            classes=2,
        )
        self.loss_fn = DiceCrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        logits = self.model(batch["image"])
        loss = self.loss_fn(logits, batch["mask"])
        iou, f1 = binary_metrics(logits, batch["mask"])
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch["image"].shape[0])
        self.log(f"{stage}_mean_iou", iou, on_epoch=True, prog_bar=True, batch_size=batch["image"].shape[0])
        self.log(f"{stage}_f1_score", f1, on_epoch=True, prog_bar=True, batch_size=batch["image"].shape[0])
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


def train_unet_baseline(config: dict[str, Any], output_dir: str = "outputs/baseline") -> None:
    """Train the U-Net baseline with the same AMP and accumulation policy as Prithvi."""
    train_dataset = dataset_from_config(config, "train_split", augment=True)
    val_dataset = dataset_from_config(config, "val_split")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    output_path = Path(output_dir)
    module = UNetLightningModule(
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    trainer = L.Trainer(
        max_epochs=int(config["training"]["max_epochs"]),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=config["training"].get("precision", "16-mixed") if torch.cuda.is_available() else "32-true",
        accumulate_grad_batches=int(config["training"]["gradient_accumulation_steps"]),
        gradient_clip_val=1.0,
        default_root_dir=output_path,
        callbacks=[
            ModelCheckpoint(dirpath=output_path / "checkpoints", monitor="val_mean_iou", mode="max", save_top_k=1),
            EarlyStopping(monitor="val_mean_iou", mode="max", patience=int(config["training"]["early_stopping_patience"])),
        ],
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["evaluate", "baseline"], required=True)
    parser.add_argument("--config", default="configs/sen1floods.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/epoch=24-step=400.ckpt")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()
    config = load_config(args.config)
    L.seed_everything(int(config["seed"]), workers=True)
    if args.mode == "evaluate":
        print(evaluate_prithvi(config, args.checkpoint, args.output_dir))
    else:
        train_unet_baseline(config, args.output_dir)


if __name__ == "__main__":
    main()
