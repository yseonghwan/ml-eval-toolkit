"""CNN input normalization verifier and intermediate Feature Map diagnostic module."""

from __future__ import annotations

from typing import Any

import numpy as np

from ml_eval_toolkit.spec import CNNDebugReport, LayerActivationStat


def inspect_input_scale(images: np.ndarray) -> dict[str, float | bool]:
    """Verify whether image tensor is properly normalized (e.g., [0.0, 1.0] vs raw [0, 255])."""
    arr = np.asarray(images, dtype=np.float64)
    img_min = float(arr.min())
    img_max = float(arr.max())
    img_mean = float(arr.mean())
    is_normalized = (img_min >= -3.5) and (img_max <= 3.5) and (img_max > img_min)
    return {
        "is_normalized": bool(is_normalized),
        "min": round(img_min, 4),
        "max": round(img_max, 4),
        "mean": round(img_mean, 4),
    }


def analyze_feature_maps(
    feature_maps: np.ndarray,
    layer_name: str,
    dead_threshold: float = 1e-6,
) -> LayerActivationStat:
    """Compute channel-wise sparsity and activation statistics for a 4D feature map (B, H, W, C)."""
    arr = np.asarray(feature_maps, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D tensor (batch, H, W, channels), got {arr.ndim}D.")

    batch, height, width, channels = arr.shape
    channel_max = arr.max(axis=(0, 1, 2))
    dead_channels = int(np.sum(np.abs(channel_max) <= dead_threshold))
    dead_ratio = dead_channels / float(channels)
    zero_ratio = float(np.mean(np.abs(arr) <= dead_threshold))

    return LayerActivationStat(
        layer_name=layer_name,
        shape=(batch, height, width, channels),
        mean_activation=round(float(arr.mean()), 5),
        std_activation=round(float(arr.std()), 5),
        max_activation=round(float(arr.max()), 5),
        dead_channel_ratio=round(dead_ratio, 4),
        zero_activation_ratio=round(zero_ratio, 4),
    )


def run_cnn_diagnostic_report(
    input_batch: np.ndarray,
    layer_outputs: dict[str, np.ndarray],
) -> CNNDebugReport:
    """Generate a structured CNNDebugReport from input images and intermediate layer tensors."""
    scale_info = inspect_input_scale(input_batch)
    warnings: list[str] = []

    if not scale_info["is_normalized"]:
        warnings.append(
            f"Input scale anomaly detected: range [{scale_info['min']}, {scale_info['max']}]. "
            "Divide raw pixel values by 255.0 or apply channel standardization."
        )

    stats: list[LayerActivationStat] = []
    for name, fmap in layer_outputs.items():
        stat = analyze_feature_maps(fmap, layer_name=name)
        stats.append(stat)
        if stat.dead_channel_ratio >= 0.5:
            warnings.append(
                f"Layer '{name}' has {stat.dead_channel_ratio * 100:.1f}% dead channels."
            )

    return CNNDebugReport(
        is_input_normalized=bool(scale_info["is_normalized"]),
        input_min=float(scale_info["min"]),
        input_max=float(scale_info["max"]),
        input_mean=float(scale_info["mean"]),
        layer_stats=stats,
        warning_messages=warnings,
    )


def build_deterministic_fashion_cnn(seed: int = 42, dropout_rate: float = 0.4) -> Any:
    """Build the Fashion-MNIST CNN model with deterministic op config (jit_compile=False)."""
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(32, kernel_size=3, activation="relu", padding="same", name="conv2d"),
            keras.layers.MaxPooling2D(2, name="max_pooling2d"),
            keras.layers.Conv2D(
                64, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_1"
            ),
            keras.layers.MaxPooling2D(2, name="max_pooling2d_1"),
            keras.layers.Flatten(name="flatten"),
            keras.layers.Dense(100, activation="relu", name="dense"),
            keras.layers.Dropout(dropout_rate, name="dropout"),
            keras.layers.Dense(10, activation="softmax", name="dense_1"),
        ]
    )
    # Disable XLA jit_compile to avoid MaxPooling2D determinism error in TensorFlow
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )
    return model
