"""Strict I/O Data Specifications for human and LLM tool interoperability."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ClusterEvalResult:
    """Quantitative evaluation output for a single K-Means configuration."""

    k: int
    scale_method: str
    inertia: float
    silhouette_score: float
    iterations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LayerActivationStat:
    """Diagnostic statistics for a 4D convolutional feature map tensor (B, H, W, C)."""

    layer_name: str
    shape: tuple[int, int, int, int]
    mean_activation: float
    std_activation: float
    max_activation: float
    dead_channel_ratio: float
    zero_activation_ratio: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        return data


@dataclass(frozen=True)
class CNNDebugReport:
    """End-to-end diagnostic report for input normalization and layer activations."""

    is_input_normalized: bool
    input_min: float
    input_max: float
    input_mean: float
    layer_stats: list[LayerActivationStat]
    warning_messages: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_input_normalized": self.is_input_normalized,
            "input_min": self.input_min,
            "input_max": self.input_max,
            "input_mean": self.input_mean,
            "layer_stats": [stat.to_dict() for stat in self.layer_stats],
            "warning_messages": self.warning_messages,
        }
