"""Compare Prithvi and U-Net as the Sen1Floods11 training fraction changes."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader, Subset

from evaluate_and_baseline import UNetLightningModule, dataset_from_config
from train import FloodLightningModule


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_fraction(
    model_name: str,
    config: dict[str, Any],
    fraction: float,
    max_epochs: int,
    seed: int,
) -> float:
    random_generator = random.Random(seed)
    full_train = dataset_from_config(config, "train_split", augment=True)
    selected_count = max(1, int(len(full_train) * fraction))
    selected_indices = random_generator.sample(range(len(full_train)), selected_count)
    train_dataset = Subset(full_train, selected_indices)
    val_dataset = dataset_from_config(config, "val_split")
    batch_size = int(config["training"]["batch_size"])
    train_loader = make_loader(train_dataset, batch_size, shuffle=True)
    val_loader = make_loader(val_dataset, batch_size, shuffle=False)

    if model_name == "prithvi":
        module = FloodLightningModule(config)
    else:
        module = UNetLightningModule(
            learning_rate=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=config["training"].get("precision", "16-mixed") if torch.cuda.is_available() else "32-true",
        accumulate_grad_batches=int(config["training"]["gradient_accumulation_steps"]),
        gradient_clip_val=1.0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        callbacks=[EarlyStopping(monitor="val_mean_iou", mode="max", patience=min(3, max_epochs))],
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    metric = trainer.callback_metrics.get("val_mean_iou")
    if metric is None:
        raise RuntimeError(f"No validation IoU was logged for {model_name} at fraction {fraction}")
    return float(metric.detach().cpu())


def run_experiment(config: dict[str, Any], fractions: list[float], max_epochs: int, output_csv: str) -> None:
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        for model_name in ("prithvi", "unet"):
            L.seed_everything(int(config["seed"]), workers=True)
            print(f"Training {model_name} with {fraction:.0%} of the train split")
            iou = train_one_fraction(model_name, config, fraction, max_epochs, int(config["seed"]))
            rows.append({"model": model_name, "fraction": fraction, "val_mean_iou": iou})

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=["model", "fraction", "val_mean_iou"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved data-efficiency results to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sen1floods.yaml")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.10, 0.25, 0.50])
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--output-csv", default="outputs/data_efficiency/results.csv")
    args = parser.parse_args()
    run_experiment(load_config(args.config), args.fractions, args.max_epochs, args.output_csv)


if __name__ == "__main__":
    main()
