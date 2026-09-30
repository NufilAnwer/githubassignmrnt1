"""
src/preprocess.py
-----------------
Stage 2 — Normalize pixel values and split off a validation set.

Reads hyperparameters from params.yaml:
    preprocess.test_size  — fraction of training data used for validation
    preprocess.seed       — random seed for reproducible split

Run standalone:
    python src/preprocess.py
"""

import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split


RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
PARAMS_FILE   = "params.yaml"


def load_params():
    with open(PARAMS_FILE, "r") as f:
        params = yaml.safe_load(f)
    return params["preprocess"]


def preprocess():
    params = load_params()
    val_size = params["test_size"]
    seed     = params["seed"]

    print("[preprocess] Loading raw arrays...")
    x_train_raw = np.load(os.path.join(RAW_DIR, "x_train.npy"))
    y_train_raw = np.load(os.path.join(RAW_DIR, "y_train.npy"))
    x_test_raw  = np.load(os.path.join(RAW_DIR, "x_test.npy"))
    y_test_raw  = np.load(os.path.join(RAW_DIR, "y_test.npy"))

    # Normalize pixel values from [0, 255] to [0, 1]
    print("[preprocess] Normalizing pixel values to [0, 1]...")
    x_train_norm = x_train_raw.astype("float32") / 255.0
    x_test_norm  = x_test_raw.astype("float32")  / 255.0

    # Split training data into train / validation sets
    print(f"[preprocess] Splitting validation set (size={val_size}, seed={seed})...")
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_norm, y_train_raw,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_raw,
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    np.save(os.path.join(PROCESSED_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "x_val.npy"),   x_val)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(PROCESSED_DIR, "x_test.npy"),  x_test_norm)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test_raw)

    print(f"[preprocess] Saved processed arrays to '{PROCESSED_DIR}/'")
    print(f"  x_train : {x_train.shape}  |  x_val : {x_val.shape}  |  x_test : {x_test_norm.shape}")


if __name__ == "__main__":
    preprocess()
