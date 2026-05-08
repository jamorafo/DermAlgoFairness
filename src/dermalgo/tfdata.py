"""TensorFlow dataset utilities for image classification."""

from __future__ import annotations

from typing import Callable

import pandas as pd
import tensorflow as tf


def decode_resize_image(path: tf.Tensor, image_size: tuple[int, int]) -> tf.Tensor:
    """Read, decode, resize, and convert one image to float32."""
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    return image


def make_image_dataset(
    df: pd.DataFrame,
    image_size: tuple[int, int],
    batch_size: int,
    preprocess_input: Callable,
    shuffle: bool = False,
    seed: int = 1,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from image paths and binary labels."""
    paths = df["image_path"].astype(str).to_numpy()
    labels = df["label_binary"].astype("float32").to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def load_one(path: tf.Tensor, label: tf.Tensor):
        image = decode_resize_image(path, image_size)
        image = preprocess_input(image)
        return image, label

    dataset = dataset.map(load_one, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
