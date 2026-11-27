import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from . import config


def build_model():
    """MobileNetV2 + sigmoid head. Input: raw RGB pixels 0-255 (rescaling is inside).

    Output: probability of class index 1 = 'without_mask'.
    """
    inputs = layers.Input(shape=(*config.IMG_SIZE, 3))
    x = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)

    base = MobileNetV2(include_top=False, weights="imagenet",
                       input_shape=(*config.IMG_SIZE, 3))
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs, name="face_mask_mobilenetv2"), base
