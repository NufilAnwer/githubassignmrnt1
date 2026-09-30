"""
src/prepare.py
--------------
Stage 1 — Download Fashion-MNIST and save raw arrays to data/raw/.

No hyperparameters are needed at this stage.
Run standalone:
    python src/prepare.py
"""

import os
import numpy as np
import tensorflow as tf


RAW_DIR = os.path.join("data", "raw")


def download_and_save():
    """Download Fashion-MNIST via Keras and persist raw arrays as .npy files."""
    print("[prepare] Loading Fashion-MNIST from Keras datasets...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    os.makedirs(RAW_DIR, exist_ok=True)

    np.save(os.path.join(RAW_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(RAW_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(RAW_DIR, "x_test.npy"),  x_test)
    np.save(os.path.join(RAW_DIR, "y_test.npy"),  y_test)

    print(f"[prepare] Saved raw arrays to '{RAW_DIR}/'")
    print(f"  x_train shape : {x_train.shape}")
    print(f"  y_train shape : {y_train.shape}")
    print(f"  x_test  shape : {x_test.shape}")
    print(f"  y_test  shape : {y_test.shape}")


if __name__ == "__main__":
    download_and_save()
