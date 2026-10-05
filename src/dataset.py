"""Rasterio-backed Sentinel-2 dataset and DataLoaders for flood segmentation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset


class FloodSegmentationDataset(Dataset[dict[str, torch.Tensor]]):
    """Load paired multi-band GeoTIFF images and binary flood masks."""

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        split_file: str | Path | None,
        band_indices: list[int],
        mean: list[float],
        std: list[float],
        image_size: int,
        augment: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.split_file = Path(split_file) if split_file else None
        self.band_indices = band_indices
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
        if len(self.band_indices) != len(mean) or len(mean) != len(std):
            raise ValueError("band_indices, mean, and std must have the same length")

        self.pairs = self._read_pairs()
        if not self.pairs:
            raise FileNotFoundError(f"No image/mask pairs found using {self.split_file}")

        transforms: list[Any] = [A.Resize(image_size, image_size)]
        if augment:
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Resize(image_size, image_size),
            ]
        self.transform = A.Compose(transforms)

    def _read_pairs(self) -> list[tuple[Path, Path]]:
        if self.split_file is None:
            split_pairs = [(path.name, path.name) for path in sorted(self.image_dir.rglob("*.tif"))]
        else:
            with self.split_file.open(newline="", encoding="utf-8") as split_handle:
                split_pairs = [tuple(row[:2]) for row in csv.reader(split_handle) if len(row) >= 2]

        pairs = []
        for image_name, mask_name in split_pairs:
            image_name = image_name.replace("_S1Hand.tif", "_S2Hand.tif")
            image_path = self.image_dir / image_name
            mask_path = self.mask_dir / mask_name
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {image_name}: {mask_path}")
            if not image_path.exists():
                raise FileNotFoundError(f"Missing Sentinel-2 image: {image_path}")
            pairs.append((image_path, mask_path))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        with rasterio.open(image_path) as image_src:
            # Rasterio indexes bands from 1. These six bands are B2, B3, B4,
            # B8, B11, and B12, mapped by YAML to Prithvi's six inputs.
            image = image_src.read(self.band_indices).astype(np.float32)
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1).astype(np.uint8)

        # Scaling reflectance to [0, 1] is common for uint16 Sentinel products.
        if image.max(initial=0) > 1.5:
            image /= 10000.0
        image = (image - self.mean) / np.maximum(self.std, 1e-6)
        transformed = self.transform(image=image.transpose(1, 2, 0), mask=mask)
        image = transformed["image"].transpose(2, 0, 1).astype(np.float32)
        mask = (transformed["mask"] > 0).astype(np.int64)
        return {"image": torch.from_numpy(image), "mask": torch.from_numpy(mask)}


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Build train/validation loaders from the YAML config."""
    root = Path(config["paths"]["data_root"])
    bands = config["bands"]
    size = int(config["training"]["image_size"])
    common = dict(
        image_dir=root / config["paths"]["image_dir"],
        mask_dir=root / config["paths"]["mask_dir"],
        band_indices=bands["indices"],
        mean=bands["mean"],
        std=bands["std"],
        image_size=size,
    )
    train_dataset = FloodSegmentationDataset(
        split_file=root / config["paths"]["train_split"],
        augment=True,
        **common,
    )
    val_dataset = FloodSegmentationDataset(
        split_file=root / config["paths"]["val_split"],
        augment=False,
        **common,
    )
    loader_args = dict(
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(config["training"].get("num_workers", 0)) > 0,
    )
    return (
        DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_args),
        DataLoader(val_dataset, shuffle=False, **loader_args),
    )
