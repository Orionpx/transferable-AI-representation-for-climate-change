"""Visualise Prithvi backbone representations across tasks with t-SNE.

Extracts pooled features from the pretrained Prithvi backbone for samples
from both the Sen1Floods11 flood dataset and the EuroSAT land-cover dataset,
then projects them into 2-D with t-SNE.  The resulting scatter plot shows
whether the foundation model learns a shared feature space that is useful
across diverse downstream climate tasks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from evaluate_and_baseline import dataset_from_config
from transfer_eurosat import SixChannelEuroSAT, build_backbone


EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "Vegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


def pool_features(features: Any) -> torch.Tensor:
    """Global-average-pool Prithvi outputs to a single feature vector."""
    if isinstance(features, dict):
        features = list(features.values())[-1]
    elif isinstance(features, (list, tuple)):
        features = features[-1]
    if features.ndim == 3:
        return features.mean(dim=1)
    if features.ndim == 4:
        return features.mean(dim=(2, 3))
    raise TypeError(f"Unexpected feature shape: {features.shape}")


@torch.inference_mode()
def extract_features(
    backbone: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 500,
    return_labels: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run the backbone on `loader` and return pooled features."""
    features_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    collected = 0
    for batch in loader:
        if isinstance(batch, dict):
            images = batch["image"]
            batch_labels = batch.get("mask")
        else:
            images, batch_labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda",
        ):
            feats = pool_features(backbone(images))
        features_list.append(feats.cpu().numpy())
        if return_labels and batch_labels is not None:
            if batch_labels.ndim > 1:
                batch_labels = torch.zeros(images.shape[0], dtype=torch.long)
            labels_list.append(batch_labels.numpy())
        collected += images.shape[0]
        if collected >= max_samples:
            break
    all_features = np.concatenate(features_list, axis=0)[:max_samples]
    all_labels = np.concatenate(labels_list, axis=0)[:max_samples] if labels_list else None
    return all_features, all_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sen1floods.yaml")
    parser.add_argument("--eurosat-dir", default="data/eurosat")
    parser.add_argument("--output-dir", default="outputs/visualizations")
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    config_path = args.config
    with Path(config_path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = int(config["training"]["image_size"])
    batch_size = int(config["training"]["batch_size"])

    # Load pretrained backbone (no task head)
    backbone = build_backbone(pretrained=True).to(device).eval()

    # ---------- Extract flood-task features ----------
    flood_dataset = dataset_from_config(config, "train_split")
    flood_loader = DataLoader(flood_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    flood_features, _ = extract_features(backbone, flood_loader, device, args.max_samples)
    print(f"Extracted {len(flood_features)} flood features")

    # ---------- Extract EuroSAT features ----------
    eurosat_dataset = SixChannelEuroSAT(args.eurosat_dir, image_size=image_size, download=True)
    eurosat_loader = DataLoader(eurosat_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    eurosat_features, eurosat_labels = extract_features(
        backbone, eurosat_loader, device, args.max_samples, return_labels=True,
    )
    print(f"Extracted {len(eurosat_features)} EuroSAT features")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ==================== PLOT 1: Cross-Task t-SNE ====================
    all_features = np.concatenate([flood_features, eurosat_features], axis=0)
    task_labels = np.array(
        ["Flood Segmentation"] * len(flood_features)
        + ["Land-Cover Classification"] * len(eurosat_features)
    )

    print(f"Running t-SNE on {len(all_features)} samples …")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings = tsne.fit_transform(all_features)

    fig, ax = plt.subplots(figsize=(10, 8))
    colours = {"Flood Segmentation": "#1E88E5", "Land-Cover Classification": "#FB8C00"}
    for task_name, colour in colours.items():
        mask = task_labels == task_name
        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=colour, label=task_name, alpha=0.55, s=18, edgecolors="none",
        )
    ax.set_title("t-SNE of Prithvi Backbone Features — Cross-Task", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(fontsize=12, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path / "tsne_cross_task.png", dpi=200)
    plt.close(fig)
    print(f"Saved -> {output_path / 'tsne_cross_task.png'}")

    # ==================== PLOT 2: EuroSAT class-level t-SNE ====================
    if eurosat_labels is not None:
        tsne2 = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        euro_emb = tsne2.fit_transform(eurosat_features)

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        cmap = plt.get_cmap("tab10", 10)
        for cls_idx in range(10):
            mask = eurosat_labels == cls_idx
            if mask.sum() == 0:
                continue
            ax2.scatter(
                euro_emb[mask, 0], euro_emb[mask, 1],
                c=[cmap(cls_idx)], label=EUROSAT_CLASSES[cls_idx],
                alpha=0.6, s=18, edgecolors="none",
            )
        ax2.set_title(
            "t-SNE of Prithvi Features — EuroSAT Classes", fontsize=14, fontweight="bold",
        )
        ax2.set_xlabel("t-SNE Dimension 1")
        ax2.set_ylabel("t-SNE Dimension 2")
        ax2.legend(fontsize=9, loc="best", framealpha=0.9, ncol=2)
        fig2.tight_layout()
        fig2.savefig(output_path / "tsne_eurosat_classes.png", dpi=200)
        plt.close(fig2)
        print(f"Saved -> {output_path / 'tsne_eurosat_classes.png'}")

    print("Visualisation complete.")


if __name__ == "__main__":
    main()
