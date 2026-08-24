"""Training objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import ModelOutput


@dataclass(frozen=True)
class LossTerms:
    total: torch.Tensor
    reconstruction: torch.Tensor
    calibration: torch.Tensor


def physics_informed_loss(
    output: ModelOutput,
    observed: torch.Tensor,
    caliper_mm: torch.Tensor,
    reconstruction_weight: float = 1.0,
    calibration_weight: float = 1.0,
    reconstruction_normalization: str = "energy",
) -> LossTerms:
    squared_residual = torch.abs(output.reconstructed - observed) ** 2
    per_observation = squared_residual.mean(dim=(1, 2))
    if reconstruction_normalization == "energy":
        energy = torch.abs(observed).square().mean(dim=(1, 2)).clamp_min(1e-8)
        per_observation = per_observation / energy
    elif reconstruction_normalization != "none":
        raise ValueError("reconstruction_normalization must be 'none' or 'energy'")
    reconstruction = per_observation.mean()
    calibration = torch.mean((output.caliper_mm - caliper_mm) ** 2)
    total = reconstruction_weight * reconstruction + calibration_weight * calibration
    return LossTerms(total, reconstruction, calibration)
