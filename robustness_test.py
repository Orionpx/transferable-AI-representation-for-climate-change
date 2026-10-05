"""Evaluate Prithvi under synthetic missing-data and sensor-noise corruption."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from evaluate_and_baseline import UNetLightningModule, binary_metrics, dataset_from_config, load_config, load_prithvi_checkpoint


def evaluate_dataset(model: torch.nn.Module, dataset, batch_size: int) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = next(model.parameters()).device
    ious: list[float] = []
    f1_scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
            iou, f1 = binary_metrics(logits, masks)
            ious.append(iou)
            f1_scores.append(f1)
    return {"mean_iou": float(np.mean(ious)), "f1_score": float(np.mean(f1_scores))}


def load_unet_checkpoint(checkpoint_path: str) -> torch.nn.Module:
    """Load a U-Net baseline checkpoint for robustness comparison."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    module = UNetLightningModule(learning_rate=1e-4, weight_decay=1e-4)
    module.load_state_dict(state_dict, strict=True)
    return module.model


def run_robustness(
    config: dict[str, Any],
    checkpoint: str,
    output_csv: str,
    baseline_checkpoint: str | None = None,
) -> None:
    clean_dataset = dataset_from_config(config, "test_split")
    corrupted_dataset = dataset_from_config(config, "test_split")
    corrupted_dataset.transform = A.Compose([
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(16, 64),
            hole_width_range=(16, 64),
            fill=0,
            p=1.0,
        ),
        A.GaussNoise(std_range=(0.05, 0.15), mean_range=(0.0, 0.0), p=1.0),
        A.Resize(int(config["training"]["image_size"]), int(config["training"]["image_size"])),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(config["training"]["batch_size"])

    # ---------- Evaluate Prithvi ----------
    prithvi_model = load_prithvi_checkpoint(config, checkpoint).to(device)
    prithvi_clean = evaluate_dataset(prithvi_model, clean_dataset, batch_size)
    prithvi_corrupted = evaluate_dataset(prithvi_model, corrupted_dataset, batch_size)
    del prithvi_model
    torch.cuda.empty_cache()

    rows = [
        {"model": "prithvi", "condition": "clean", **prithvi_clean, "iou_delta": 0.0},
        {
            "model": "prithvi",
            "condition": "corrupted",
            **prithvi_corrupted,
            "iou_delta": prithvi_corrupted["mean_iou"] - prithvi_clean["mean_iou"],
        },
    ]

    # ---------- Evaluate U-Net baseline (if provided) ----------
    if baseline_checkpoint:
        unet_model = load_unet_checkpoint(baseline_checkpoint).to(device)
        unet_clean = evaluate_dataset(unet_model, clean_dataset, batch_size)
        unet_corrupted = evaluate_dataset(unet_model, corrupted_dataset, batch_size)
        del unet_model
        torch.cuda.empty_cache()
        rows.extend([
            {"model": "unet", "condition": "clean", **unet_clean, "iou_delta": 0.0},
            {
                "model": "unet",
                "condition": "corrupted",
                **unet_corrupted,
                "iou_delta": unet_corrupted["mean_iou"] - unet_clean["mean_iou"],
            },
        ])

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(
            result_file,
            fieldnames=["model", "condition", "mean_iou", "f1_score", "iou_delta"],
        )
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Saved robustness results to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sen1floods.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/epoch=24-step=400.ckpt")
    parser.add_argument(
        "--baseline-checkpoint", default=None,
        help="Path to U-Net baseline checkpoint for robustness comparison",
    )
    parser.add_argument("--output-csv", default="outputs/robustness/results.csv")
    args = parser.parse_args()
    run_robustness(
        load_config(args.config), args.checkpoint, args.output_csv, args.baseline_checkpoint,
    )


if __name__ == "__main__":
    main()
