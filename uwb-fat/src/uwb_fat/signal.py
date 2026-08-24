"""Signal preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AcquisitionConfig

CROSS_CHANNEL_INDICES = (1, 2)  # TX0-RX1 and TX1-RX0 in canonical order.


@dataclass(frozen=True)
class PreprocessedRecording:
    response: np.ndarray  # [window, cross-channel, frequency], complex64
    frequencies_hz: np.ndarray  # [frequency], float64


def window_average(
    cir: np.ndarray,
    window_size: int = 100,
    stride: int = 100,
) -> np.ndarray:
    """Average complex frames without assuming phase-aligned coherent gain.

    Args:
        cir: Canonical array with shape [frames, channels, range_bins].
    """
    cir = np.asarray(cir)
    if cir.ndim != 3:
        raise ValueError(f"Expected [frames, channels, bins], got shape {cir.shape}")
    if not np.iscomplexobj(cir):
        raise TypeError("CIR must be complex; magnitude-only input discards required phase")
    if cir.shape[0] < window_size:
        raise ValueError(f"Need at least {window_size} frames, got {cir.shape[0]}")
    starts = range(0, cir.shape[0] - window_size + 1, stride)
    return np.stack([cir[start : start + window_size].mean(axis=0) for start in starts])


def inband_fft(
    averaged_cir: np.ndarray,
    sample_rate_hz: float,
    center_frequency_hz: float,
    receive_bandwidth_hz: float,
    fft_size: int,
    expected_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert CIR windows to ordered RF bins inside the receiver passband."""
    if averaged_cir.shape[-1] > fft_size:
        raise ValueError("fft_size cannot be smaller than the CIR range-bin count")
    spectrum = np.fft.fftshift(np.fft.fft(averaged_cir, n=fft_size, axis=-1), axes=-1)
    baseband = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate_hz))
    mask = np.abs(baseband) <= receive_bandwidth_hz / 2.0 + 1e-6
    if int(mask.sum()) != expected_bins:
        raise ValueError(
            "Receiver passband selected "
            f"{int(mask.sum())} bins, but the configured model expects {expected_bins}. "
            "Set an FFT size consistent with the model configuration."
        )
    return spectrum[..., mask].astype(np.complex64), center_frequency_hz + baseband[mask]


def preprocess_cir(cir: np.ndarray, config: AcquisitionConfig) -> PreprocessedRecording:
    """Apply window averaging, cross-channel selection, FFT, and passband restriction."""
    cir = np.asarray(cir)
    if cir.ndim != 3:
        raise ValueError("CIR must have canonical shape [frames, channels, range_bins]")
    if cir.shape[1] != len(config.channel_order):
        raise ValueError(
            f"Expected {len(config.channel_order)} channels in "
            f"{config.channel_order}, got {cir.shape[1]}"
        )
    if cir.shape[2] != config.range_bins:
        raise ValueError(f"Expected {config.range_bins} range bins, got {cir.shape[2]}")
    averaged = window_average(cir, config.window_size, config.window_stride)
    selected = averaged[:, config.retained_channel_indices, :]
    response, frequencies = inband_fft(
        selected,
        config.sample_rate_hz,
        config.center_frequency_hz,
        config.receive_bandwidth_hz,
        config.fft_size,
        config.inband_bins,
    )
    return PreprocessedRecording(response, frequencies)


@dataclass
class ComplexStandardizer:
    """Training-fold-only scalar normalization for complex responses."""

    scale: float = 1.0

    def fit(self, response: np.ndarray) -> ComplexStandardizer:
        response = np.asarray(response)
        energy = float(np.mean(np.abs(response) ** 2))
        if not np.isfinite(energy) or energy <= 0.0:
            raise ValueError("Cannot normalize an empty, non-finite, or zero-energy response")
        self.scale = float(np.sqrt(energy))
        return self

    def transform(self, response: np.ndarray) -> np.ndarray:
        return np.asarray(response) / self.scale

    def inverse_transform(self, response: np.ndarray) -> np.ndarray:
        return np.asarray(response) * self.scale
