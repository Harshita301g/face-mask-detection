import json

import tensorflow as tf

from . import config


def load_datasets():
    """Return train, val, test datasets and class names (['with_mask', 'without_mask'])."""
    data_dir = config.DATA_DIR
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found at {data_dir}. See data/README.md.")

    common = dict(
        directory=data_dir,
        validation_split=config.VAL_TEST_SPLIT,
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="binary",
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", **common)
    val_test_ds = tf.keras.utils.image_dataset_from_directory(subset="validation", **common)
    class_names = train_ds.class_names

    n_batches = tf.data.experimental.cardinality(val_test_ds).numpy()
    test_ds = val_test_ds.take(n_batches // 2)
    val_ds = val_test_ds.skip(n_batches // 2)

    auto = tf.data.AUTOTUNE
    config.MODEL_DIR.mkdir(exist_ok=True)
    with open(config.CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)
    return train_ds.prefetch(auto), val_ds.prefetch(auto), test_ds.prefetch(auto), class_names
