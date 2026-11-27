"""Evaluate on the held-out test set.   Usage:  python -m src.evaluate"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from . import config
from .data import load_datasets


def main():
    _, _, test_ds, class_names = load_datasets()
    model = tf.keras.models.load_model(config.MODEL_PATH)

    y_true, y_pred = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)[:, 0]
        y_pred.extend((probs > 0.5).astype(int))
        y_true.extend(labels.numpy()[:, 0].astype(int))

    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   digits=4, zero_division=0)
    print(f"Test accuracy: {acc:.4f}\n\n{report}")
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "classification_report.txt").write_text(
        f"Test accuracy: {acc:.4f}\n\n{report}")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "confusion_matrix.png", dpi=150)
    print(f"Saved results to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
