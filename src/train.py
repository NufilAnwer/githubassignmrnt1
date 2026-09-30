"""
src/train.py
------------
Stage 3 — Build and train a fully-connected ANN on Fashion-MNIST.

Architecture:
    Flatten → Dense(ReLU) → Dropout → Dense(10, Softmax)

Reads hyperparameters from params.yaml:
    train.dense_units    — number of units in the hidden Dense layer
    train.dropout_rate   — Dropout fraction
    train.learning_rate  — Adam optimizer learning rate
    train.epochs         — number of training epochs
    train.batch_size     — mini-batch size

Outputs:
    models/model.h5      — saved Keras model
    models/history.csv   — per-epoch training metrics

Run standalone:
    python src/train.py
"""

import os
import csv
import yaml
import numpy as np
import tensorflow as tf


PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = "models"
PARAMS_FILE   = "params.yaml"


def load_params():
    with open(PARAMS_FILE, "r") as f:
        params = yaml.safe_load(f)
    return params["train"]


def build_model(input_shape, dense_units, dropout_rate, learning_rate):
    """Build a Sequential ANN: Flatten → Dense(ReLU) → Dropout → Dense(10, Softmax)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(dense_units, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="fashion_ann")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history(history, path):
    """Write Keras History object to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow(["epoch"] + list(history.history.keys()))
        for epoch, values in enumerate(zip(*history.history.values()), start=1):
            writer.writerow([epoch] + list(values))


def train():
    params = load_params()
    dense_units   = params["dense_units"]
    dropout_rate  = params["dropout_rate"]
    learning_rate = params["learning_rate"]
    epochs        = params["epochs"]
    batch_size    = params["batch_size"]

    # Reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    print("[train] Loading processed data...")
    x_train = np.load(os.path.join(PROCESSED_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    x_val   = np.load(os.path.join(PROCESSED_DIR, "x_val.npy"))
    y_val   = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))

    print(f"[train] Building ANN (dense_units={dense_units}, dropout={dropout_rate}, lr={learning_rate})...")
    model = build_model(
        input_shape=(28, 28),
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )
    model.summary()

    print(f"[train] Training for {epochs} epochs (batch_size={batch_size})...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path   = os.path.join(MODELS_DIR, "model.h5")
    history_path = os.path.join(MODELS_DIR, "history.csv")

    model.save(model_path)
    save_history(history, history_path)

    print(f"[train] Model saved  → {model_path}")
    print(f"[train] History saved → {history_path}")

    # Print final epoch metrics
    final_acc     = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"[train] Final train accuracy : {final_acc:.4f}")
    print(f"[train] Final val   accuracy : {final_val_acc:.4f}")


if __name__ == "__main__":
    train()
