"""Model-building utilities for DermAlgoFairness."""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


APPLICATIONS = {
    "ResNet50": tf.keras.applications.ResNet50,
    "DenseNet121": tf.keras.applications.DenseNet121,
    "MobileNetV2": tf.keras.applications.MobileNetV2,
    "EfficientNetV2B0": tf.keras.applications.EfficientNetV2B0,
    "VGG16": tf.keras.applications.VGG16,
}


def get_preprocess_input(keras_application: str):
    """Return the correct Keras preprocess_input function for an application."""
    mapping = {
        "ResNet50": tf.keras.applications.resnet50.preprocess_input,
        "DenseNet121": tf.keras.applications.densenet.preprocess_input,
        "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
        "EfficientNetV2B0": tf.keras.applications.efficientnet_v2.preprocess_input,
        "VGG16": tf.keras.applications.vgg16.preprocess_input,
    }

    if keras_application not in mapping:
        raise ValueError(f"Unsupported keras_application: {keras_application}")

    return mapping[keras_application]


def build_binary_classifier(config: dict[str, Any]) -> Model:
    """Build and compile a binary image classifier from config."""
    model_cfg = config["model"]
    training_cfg = config["training"]

    keras_application = model_cfg["keras_application"]
    if keras_application not in APPLICATIONS:
        raise ValueError(f"Unsupported keras_application: {keras_application}")

    input_size = tuple(model_cfg["input_size"])
    input_shape = (*input_size, 3)

    weights = "imagenet" if model_cfg.get("imagenet_pretrained", True) else None

    base_model = APPLICATIONS[keras_application](
        weights=weights,
        include_top=False,
        input_shape=input_shape,
    )

    trainable_last_layers = int(model_cfg.get("trainable_last_layers", 0))
    if trainable_last_layers <= 0:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_last_layers]:
            layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(int(model_cfg.get("dense_units", 1024)), activation="relu")(x)
    x = Dropout(float(model_cfg.get("dropout", 0.5)))(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    optimizer_name = training_cfg.get("optimizer", "adam").lower()
    learning_rate = float(training_cfg.get("learning_rate", 1e-6))

    if optimizer_name != "adam":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=training_cfg.get("loss", "binary_crossentropy"),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc_roc", curve="ROC"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
        ],
    )

    return model
