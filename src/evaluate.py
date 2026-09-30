"""
src/evaluate.py
---------------
Stage 4 — Evaluate the trained model and write metrics.

Outputs:
    metrics.json          — test loss & accuracy (DVC-tracked metric)
    models/confusion_matrix.png — confusion matrix heatmap

Run standalone:
    python src/evaluate.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend — no display required
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report


PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = "models"
METRICS_FILE  = "metrics.json"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def plot_confusion_matrix(cm, save_path):
    """Save a color-coded confusion matrix as a PNG image."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix — Fashion-MNIST",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Confusion matrix saved → {save_path}")


def evaluate():
    print("[evaluate] Loading model and test data...")
    model  = tf.keras.models.load_model(os.path.join(MODELS_DIR, "model.h5"))
    x_test = np.load(os.path.join(PROCESSED_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    print("[evaluate] Running inference on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    print(f"[evaluate] Test Loss     : {test_loss:.4f}")
    print(f"[evaluate] Test Accuracy : {test_acc:.4f}")

    # Per-class report
    print("\n[evaluate] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Save confusion matrix image
    os.makedirs(MODELS_DIR, exist_ok=True)
    plot_confusion_matrix(cm, os.path.join(MODELS_DIR, "confusion_matrix.png"))

    # Write metrics.json
    metrics = {
        "test_loss":     round(float(test_loss), 6),
        "test_accuracy": round(float(test_acc),  6),
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[evaluate] Metrics written → {METRICS_FILE}")
    print(f"  {metrics}")


if __name__ == "__main__":
    evaluate()
