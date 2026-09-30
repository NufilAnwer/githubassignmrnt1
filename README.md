# Fashion-ANN-Pipeline

End-to-End ML Versioning with Git, DVC & Google Drive

## Overview

A fully-connected Artificial Neural Network (ANN) that classifies Fashion-MNIST images into 10 clothing categories.  
Target: **≥ 85% test accuracy**.  
The pipeline is reproducible end-to-end via a single `dvc repro` call, with every artifact versioned through DVC and stored on Google Drive.

## Project Structure

```
fashion-ann-pipeline/
├── src/
│   ├── prepare.py       # Download Fashion-MNIST → data/raw/
│   ├── preprocess.py    # Normalize + split → data/processed/
│   ├── train.py         # Build & train ANN → models/
│   └── evaluate.py      # Metrics & confusion matrix → metrics.json
├── data/
│   ├── raw/             # DVC-tracked raw .npy files
│   └── processed/       # DVC-tracked processed .npy files
├── models/              # DVC-tracked model.h5 + history.csv
├── params.yaml          # Central hyperparameters
├── dvc.yaml             # Pipeline stage definitions
├── dvc.lock             # Auto-generated lock file
├── metrics.json         # Evaluation metrics (DVC metric)
└── requirements.txt     # Python dependencies
```

## Dataset

**Fashion-MNIST** — 70,000 grayscale 28×28 images across 10 clothing categories.  
Loaded automatically via `tf.keras.datasets.fashion_mnist.load_data()`.

| Label | Class |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Quick Start

### 1. Create Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure DVC Remote (Google Drive)

```bash
dvc remote add -d gdrive_storage gdrive://<YOUR_FOLDER_ID>
```

### 4. Run the Full Pipeline

```bash
dvc repro
```

### 5. Push Artifacts to Google Drive

```bash
dvc push
```

## Pipeline Stages

| Stage | Script | Inputs | Outputs |
|-------|--------|--------|---------|
| prepare | src/prepare.py | — | data/raw/ |
| preprocess | src/preprocess.py | data/raw/ | data/processed/ |
| train | src/train.py | data/processed/ | models/ |
| evaluate | src/evaluate.py | models/, data/processed/ | metrics.json |

## Hyperparameters (params.yaml)

All hyperparameters are centralized in `params.yaml`. Edit them and re-run `dvc repro` to track changes.

## Author

Nufil — Assignment 3: End-to-End ML Versioning (MSc MLOps)
