"""Typed project configuration loaded from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _pair(values: list[float] | tuple[float, float]) -> tuple[float, float]:
    if len(values) != 2 or values[0] >= values[1]:
        raise ValueError(f"Expected increasing [low, high] pair, got {values!r}")
    return float(values[0]), float(values[1])


@dataclass(frozen=True)
class AcquisitionConfig:
    center_frequency_hz: float
    sample_rate_hz: float
    receive_bandwidth_hz: float
    frames_per_recording: int
    range_bins: int
    fft_size: int
    inband_bins: int
    window_size: int
    window_stride: int
    channel_order: tuple[str, ...]
    retained_channel_indices: tuple[int, ...]


@dataclass(frozen=True)
class BoundsConfig:
    gap_mm: tuple[float, float]
    skin_mm: tuple[float, float]
    fat_mm: tuple[float, float]
    caliper_mm: tuple[float, float]
    grid_step_mm: float


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int
    hidden_channels: int
    permittivity_scale: tuple[float, float]
    conductivity_scale: tuple[float, float]
    interface_residual: tuple[float, float]
    reflection_residual: tuple[float, float]
    final_response_residual: tuple[float, float]
    system_amplitude: tuple[float, float]
    system_bandwidth_ghz: tuple[float, float]
    system_phase_rad: tuple[float, float]
    system_delay_ns: tuple[float, float]
    system_chirp: tuple[float, float]


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    validation_fraction: float
    patience: int
    reconstruction_weight: float
    calibration_weight: float
    reconstruction_normalization: str


@dataclass(frozen=True)
class ProjectConfig:
    acquisition: AcquisitionConfig
    bounds: BoundsConfig
    model: ModelConfig
    training: TrainingConfig


def _construct(cls: type, raw: dict[str, Any], pair_fields: set[str] | None = None):
    values = dict(raw)
    for field in pair_fields or set():
        values[field] = _pair(values[field])
    for field in ("channel_order", "retained_channel_indices"):
        if field in values:
            values[field] = tuple(values[field])
    return cls(**values)


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a project configuration."""
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)

    acquisition = _construct(AcquisitionConfig, raw["acquisition"])
    bounds = _construct(
        BoundsConfig,
        raw["bounds"],
        {"gap_mm", "skin_mm", "fat_mm", "caliper_mm"},
    )
    model = _construct(
        ModelConfig,
        raw["model"],
        {
            "permittivity_scale",
            "conductivity_scale",
            "interface_residual",
            "reflection_residual",
            "final_response_residual",
            "system_amplitude",
            "system_bandwidth_ghz",
            "system_phase_rad",
            "system_delay_ns",
            "system_chirp",
        },
    )
    training = _construct(TrainingConfig, raw["training"])

    if acquisition.fft_size < acquisition.range_bins:
        raise ValueError("fft_size cannot be smaller than range_bins")
    if acquisition.inband_bins % 2 != 1:
        raise ValueError("inband_bins must be odd so the carrier bin is retained")
    if len(acquisition.retained_channel_indices) != 2:
        raise ValueError("retained_channel_indices must contain exactly two channels")
    if training.reconstruction_normalization not in {"none", "energy"}:
        raise ValueError("reconstruction_normalization must be 'none' or 'energy'")
    if not 0.0 < training.validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    return ProjectConfig(acquisition, bounds, model, training)
