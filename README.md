# Prithvi transferability on Sen1Floods11

A memory-conscious PyTorch Lightning experiment for evaluating a Prithvi Earth-observation backbone on binary flood segmentation.

## Layout

```text
.
|-- configs/
|   `-- sen1floods.yaml
|-- data/                         # local dataset, ignored by git
|-- outputs/                      # checkpoints and logs, ignored by git
|-- src/
|   |-- __init__.py
|   |-- dataset.py                # Rasterio dataset and DataLoaders
|   `-- model.py                  # Prithvi backbone and segmentation decoder
|-- requirements.txt
|-- train.py                      # Prithvi flood segmentation training
|-- evaluate_and_baseline.py      # Prithvi evaluation & U-Net baseline
|-- data_efficiency_runner.py     # 10%/25%/50% data fraction comparison
|-- robustness_test.py            # Corruption robustness (Prithvi & U-Net)
|-- heat_mapping.py               # LST regression transfer experiment
|-- transfer_eurosat.py           # EuroSAT classification transfer
`-- visualize_representations.py  # t-SNE feature visualisation
```

## Dataset layout

Place paired files under the configured directories. Image and mask files are matched by stem:

```text
data/sen1floods11/
|-- data/flood_events/HandLabeled/S2Hand/*.tif
|-- data/flood_events/HandLabeled/LabelHand/*.tif
`-- splits/flood_handlabeled/*.csv
```

The recipe reads six Sentinel-2 channels from the downloaded 13-band files in this order: B2, B3, B4, B8, B11, B12. The split CSVs contain `S1Hand` names; the loader maps those stems to the corresponding `S2Hand` images and `LabelHand` masks.

## Run

```powershell
python -m pip install -r requirements.txt
python train.py --config configs/sen1floods.yaml
```

The Prithvi registry key can differ between Terratorch releases. Check the installed registry and update `model.backbone_name` if `prithvi_eo_v2_100` is not available.
