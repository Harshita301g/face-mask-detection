"""Train the mask / no-mask classifier.   Usage:  python -m src.train"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from . import config
from .data import load_datasets
from .model import build_model


def callbacks():
    config.MODEL_DIR.mkdir(exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            config.MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]


def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="binary_crossentropy", metrics=["accuracy"])


def plot_history(histories, path):
    def joined(key):
        return sum((h.history[key] for h in histories), [])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        ax.plot(joined(key), label="train")
        ax.plot(joined("val_" + key), label="validation")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    tf.keras.utils.set_random_seed(config.SEED)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    train_ds, val_ds, _, class_names = load_datasets()
    print("Classes:", class_names)

    model, base = build_model()
    compile_model(model, config.LR_HEAD)
    model.summary()
    histories = [model.fit(train_ds, validation_data=val_ds,
                           epochs=config.EPOCHS_HEAD, callbacks=callbacks())]

    # Phase 2: fine-tune the top layers of MobileNetV2
    base.trainable = True
    for layer in base.layers[:-config.FINE_TUNE_LAYERS]:
        layer.trainable = False
    compile_model(model, config.LR_FINE_TUNE)
    histories.append(model.fit(train_ds, validation_data=val_ds,
                               epochs=config.EPOCHS_FINE_TUNE, callbacks=callbacks()))

    plot_history(histories, config.RESULTS_DIR / "training_curves.png")
    print(f"Best model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
