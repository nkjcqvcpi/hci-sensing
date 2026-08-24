"""Small physics-inspired neural model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import BoundsConfig, ModelConfig
from .physics import multilayer_reflection, system_response


def _bounded(raw: torch.Tensor, limits: tuple[float, float]) -> torch.Tensor:
    low, high = limits
    return low + (high - low) * torch.sigmoid(raw)


def _complex_pairs(values: torch.Tensor) -> torch.Tensor:
    return torch.complex(values[..., 0], values[..., 1])


class SignalEncoder(nn.Module):
    """Conv1D frequency encoder from two complex channels to a 64-D latent."""

    def __init__(self, hidden_channels: int = 16, latent_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(4, hidden_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, latent_dim),
            nn.GELU(),
        )

    def forward(self, response: torch.Tensor) -> torch.Tensor:
        if response.ndim != 3 or response.shape[1] != 2:
            raise ValueError(
                f"Expected [batch, 2, frequency] complex response, got {response.shape}"
            )
        features = torch.cat((response.real, response.imag), dim=1)
        return self.network(features)


@dataclass
class Condition:
    gap_mm: torch.Tensor
    skin_mm: torch.Tensor
    fat_mm: torch.Tensor
    epsilon_scales: torch.Tensor
    conductivity_scales: torch.Tensor
    interface_residuals: torch.Tensor
    reflection_residual: torch.Tensor
    system_parameters: torch.Tensor
    final_response_residual: torch.Tensor


@dataclass
class ModelOutput:
    reconstructed: torch.Tensor
    skin_mm: torch.Tensor
    fat_mm: torch.Tensor
    caliper_mm: torch.Tensor
    condition: Condition


class CorrectionHead(nn.Module):
    """Bounded physical parameters and residual corrections.

    The manuscript omits the exact tensor shapes. This head gives each named correction an
    explicit bounded representation while keeping the analytic TMM dominant.
    """

    OUTPUTS = 31

    def __init__(self, latent_dim: int, model_config: ModelConfig, bounds: BoundsConfig):
        super().__init__()
        self.model_config = model_config
        self.bounds = bounds
        self.network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, self.OUTPUTS),
        )

    def forward(self, latent: torch.Tensor) -> Condition:
        raw = self.network(latent)
        cursor = 0

        thickness = raw[:, cursor : cursor + 3]
        cursor += 3
        gap = _bounded(thickness[:, 0], self.bounds.gap_mm)
        skin = _bounded(thickness[:, 1], self.bounds.skin_mm)
        fat = _bounded(thickness[:, 2], self.bounds.fat_mm)

        dielectric = raw[:, cursor : cursor + 6].reshape(-1, 3, 2)
        cursor += 6
        epsilon_scales = _bounded(dielectric[..., 0], self.model_config.permittivity_scale)
        conductivity_scales = _bounded(dielectric[..., 1], self.model_config.conductivity_scale)

        interface = raw[:, cursor : cursor + 6].reshape(-1, 3, 2)
        cursor += 6
        interface = _bounded(interface, self.model_config.interface_residual)
        interface_residuals = _complex_pairs(interface)

        reflection = raw[:, cursor : cursor + 2]
        cursor += 2
        reflection = _bounded(reflection, self.model_config.reflection_residual)
        reflection_residual = _complex_pairs(reflection)

        system_raw = raw[:, cursor : cursor + 10].reshape(-1, 2, 5)
        cursor += 10
        system_parameters = torch.stack(
            (
                _bounded(system_raw[..., 0], self.model_config.system_amplitude),
                _bounded(system_raw[..., 1], self.model_config.system_bandwidth_ghz),
                _bounded(system_raw[..., 2], self.model_config.system_phase_rad),
                _bounded(system_raw[..., 3], self.model_config.system_delay_ns),
                _bounded(system_raw[..., 4], self.model_config.system_chirp),
            ),
            dim=-1,
        )

        final = raw[:, cursor : cursor + 4].reshape(-1, 2, 2)
        cursor += 4
        final = _bounded(final, self.model_config.final_response_residual)
        final_response_residual = _complex_pairs(final)
        if cursor != self.OUTPUTS:
            raise AssertionError("Correction-head output accounting error")

        return Condition(
            gap_mm=gap,
            skin_mm=skin,
            fat_mm=fat,
            epsilon_scales=epsilon_scales,
            conductivity_scales=conductivity_scales,
            interface_residuals=interface_residuals,
            reflection_residual=reflection_residual,
            system_parameters=system_parameters,
            final_response_residual=final_response_residual,
        )


class UWBFatModel(nn.Module):
    def __init__(self, model_config: ModelConfig, bounds: BoundsConfig):
        super().__init__()
        self.encoder = SignalEncoder(model_config.hidden_channels, model_config.latent_dim)
        self.correction_head = CorrectionHead(model_config.latent_dim, model_config, bounds)

    def condition(self, response: torch.Tensor) -> Condition:
        return self.correction_head(self.encoder(response))

    @staticmethod
    def reconstruct_from_condition(
        frequency_hz: torch.Tensor,
        condition: Condition,
        skin_mm: torch.Tensor | None = None,
        fat_mm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        skin_mm = condition.skin_mm if skin_mm is None else skin_mm
        fat_mm = condition.fat_mm if fat_mm is None else fat_mm
        gamma = multilayer_reflection(
            frequency_hz,
            condition.gap_mm / 1000.0,
            skin_mm / 1000.0,
            fat_mm / 1000.0,
            condition.epsilon_scales,
            condition.conductivity_scales,
            condition.interface_residuals,
            condition.reflection_residual,
        )
        pulse = system_response(frequency_hz, condition.system_parameters)
        response = pulse * gamma[:, None, :]
        return response * (1.0 + condition.final_response_residual[..., None])

    def forward(self, response: torch.Tensor, frequency_hz: torch.Tensor) -> ModelOutput:
        condition = self.condition(response)
        reconstructed = self.reconstruct_from_condition(frequency_hz, condition)
        caliper = 2.0 * (condition.skin_mm + condition.fat_mm)
        return ModelOutput(
            reconstructed=reconstructed,
            skin_mm=condition.skin_mm,
            fat_mm=condition.fat_mm,
            caliper_mm=caliper,
            condition=condition,
        )


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
