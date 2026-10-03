# Sentinel-2 Forest Change Detection

An end-to-end remote sensing and machine learning project for detecting forest cover change from Sentinel-2 satellite imagery.

The project covers the full pipeline from satellite data acquisition and preprocessing to semantic segmentation and bitemporal change detection. It compares a CNN-based forest segmentation approach with a traditional NDVI baseline.

## Notebooks

### 1. Data Acquisition and Visualization

`Task_1.ipynb`

- Queries Sentinel-2 Level-2A imagery from the Copernicus Data Space.
- Selects spatially overlapping tiles from 2018 and 2024.
- Downloads multispectral bands through the Copernicus S3 API.
- Builds and visualizes RGB composites.

### 2. Forest Segmentation

`Task_2.ipynb`

- Builds a PyTorch data pipeline for 10-band Sentinel-2 image patches.
- Uses a pretrained ResNet-18 Sentinel-2 encoder.
- Implements and compares:
  - FCN baseline
  - U-Net-style decoder with skip connections
- Evaluates segmentation using mean Intersection over Union (mIoU).

The skip-connection model achieved a test mIoU of **0.938**, compared with **0.773** for the FCN baseline.

### 3. Bitemporal Change Detection

`Task_3.ipynb`

- Applies the trained CNN to Sentinel-2 imagery from 2018 and 2024.
- Processes **8,281 × 10-band × 120×120** image patches per year.
- Compares CNN-based segmentation with an NDVI threshold baseline.
- Reconstructs full-tile forest masks and analyzes spatial changes.
- Performs quantitative and qualitative comparison of the two methods.

Estimated forest-cover reduction:

- **NDVI:** 8.13%
- **CNN:** 7.95%

## Setup

This project uses **Python 3.12** and [`uv`](https://docs.astral.sh/uv/).

Install dependencies:

```bash
# CPU
uv sync --python 3.12 --extra cpu

# or CUDA 12.4
uv sync --python 3.12 --extra cuda124
```

Start Jupyter Lab:

```bash
uv run --extra cpu jupyter lab

# for CUDA
uv run --extra cuda124 jupyter lab
```

Run the notebooks in order.

## Tech Stack

Python, PyTorch, PyTorch Lightning, NumPy, Rasterio, GeoPandas, Matplotlib, Copernicus Data Space API

## Notes

This project was developed as part of a university machine learning mini-project. The reported forest-change estimates are model-based results and were not validated against independent ground-truth change data.
