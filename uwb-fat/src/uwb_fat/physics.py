"""Differentiable four-medium transfer-matrix forward model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

EPSILON_0 = 8.8541878128e-12
ETA_0 = 376.730313668
C_LIGHT = 299_792_458.0


@dataclass(frozen=True)
class TissueProperty:
    relative_permittivity: float
    conductivity_s_m: float


TISSUE_PROPERTIES = {
    "air": TissueProperty(1.0, 0.0),
    "skin": TissueProperty(33.3, 5.69),
    "fat": TissueProperty(4.77, 0.43),
    "muscle": TissueProperty(45.7, 7.63),
}


def complex_permittivity(
    frequency_hz: torch.Tensor,
    relative_permittivity: torch.Tensor | float,
    conductivity_s_m: torch.Tensor | float,
) -> torch.Tensor:
    """Equation 2: relative complex permittivity under the exp(+jwt) convention."""
    frequency_hz = torch.as_tensor(frequency_hz)
    real = torch.as_tensor(
        relative_permittivity, device=frequency_hz.device, dtype=frequency_hz.dtype
    )
    sigma = torch.as_tensor(conductivity_s_m, device=frequency_hz.device, dtype=frequency_hz.dtype)
    return torch.complex(
        real + torch.zeros_like(frequency_hz), -sigma / (2.0 * torch.pi * frequency_hz * EPSILON_0)
    )


def wave_parameters(
    frequency_hz: torch.Tensor, epsilon_complex: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    refractive_index = torch.sqrt(epsilon_complex)
    wavenumber = 2.0 * torch.pi * frequency_hz * refractive_index / C_LIGHT
    impedance = ETA_0 / refractive_index
    return wavenumber, impedance


def fresnel_coefficient(
    impedance_left: torch.Tensor, impedance_right: torch.Tensor
) -> torch.Tensor:
    return (impedance_right - impedance_left) / (impedance_right + impedance_left)


def interface_matrix(
    impedance_left: torch.Tensor,
    impedance_right: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    gamma = fresnel_coefficient(impedance_left, impedance_right)
    if residual is not None:
        gamma = gamma + residual
    transmission = 1.0 + gamma
    row0 = torch.stack((torch.ones_like(gamma), gamma), dim=-1)
    row1 = torch.stack((gamma, torch.ones_like(gamma)), dim=-1)
    return torch.stack((row0, row1), dim=-2) / transmission[..., None, None]


def propagation_matrix(wavenumber: torch.Tensor, thickness_m: torch.Tensor) -> torch.Tensor:
    thickness_m = thickness_m[..., None]
    phase = torch.exp(-1j * wavenumber * thickness_m)
    zeros = torch.zeros_like(phase)
    row0 = torch.stack((phase, zeros), dim=-1)
    row1 = torch.stack((zeros, 1.0 / phase), dim=-1)
    return torch.stack((row0, row1), dim=-2)


def _material(
    frequency_hz: torch.Tensor,
    name: str,
    epsilon_scale: torch.Tensor,
    conductivity_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    nominal = TISSUE_PROPERTIES[name]
    epsilon = complex_permittivity(
        frequency_hz,
        nominal.relative_permittivity * epsilon_scale[..., None],
        nominal.conductivity_s_m * conductivity_scale[..., None],
    )
    return wave_parameters(frequency_hz, epsilon)


def multilayer_reflection(
    frequency_hz: torch.Tensor,
    gap_m: torch.Tensor,
    skin_m: torch.Tensor,
    fat_m: torch.Tensor,
    epsilon_scales: torch.Tensor | None = None,
    conductivity_scales: torch.Tensor | None = None,
    interface_residuals: torch.Tensor | None = None,
    reflection_residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return Gamma=M21/M11 for air-skin-fat-muscle.

    Thickness tensors have shape [batch]. Scale tensors have shape [batch, 3] in
    skin/fat/muscle order. Interface residuals have shape [batch, 3] complex.
    """
    frequency_hz = torch.as_tensor(frequency_hz)
    batch = gap_m.shape[0]
    dtype = frequency_hz.dtype
    device = frequency_hz.device
    if epsilon_scales is None:
        epsilon_scales = torch.ones((batch, 3), dtype=dtype, device=device)
    if conductivity_scales is None:
        conductivity_scales = torch.ones((batch, 3), dtype=dtype, device=device)

    air_epsilon = complex_permittivity(frequency_hz, 1.0, 0.0)
    k_air, eta_air_1d = wave_parameters(frequency_hz, air_epsilon)
    k_air = k_air.expand(batch, -1)
    eta_air = eta_air_1d.expand(batch, -1)
    tissue = []
    for index, name in enumerate(("skin", "fat", "muscle")):
        tissue.append(
            _material(
                frequency_hz,
                name,
                epsilon_scales[:, index],
                conductivity_scales[:, index],
            )
        )
    (k_skin, eta_skin), (k_fat, eta_fat), (_, eta_muscle) = tissue

    if interface_residuals is None:
        interface_residuals = torch.zeros((batch, 3), dtype=eta_air.dtype, device=device)
    p_air = propagation_matrix(k_air, gap_m)
    i_air_skin = interface_matrix(eta_air, eta_skin, interface_residuals[:, 0, None])
    p_skin = propagation_matrix(k_skin, skin_m)
    i_skin_fat = interface_matrix(eta_skin, eta_fat, interface_residuals[:, 1, None])
    p_fat = propagation_matrix(k_fat, fat_m)
    i_fat_muscle = interface_matrix(eta_fat, eta_muscle, interface_residuals[:, 2, None])

    matrix = p_air @ i_air_skin @ p_skin @ i_skin_fat @ p_fat @ i_fat_muscle
    gamma = matrix[..., 1, 0] / matrix[..., 0, 0]
    if reflection_residual is not None:
        gamma = gamma * (1.0 + reflection_residual[..., None])
    return gamma


def system_response(frequency_hz: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    """Gaussian magnitude and quadratic phase for each observation and channel.

    parameters shape: [batch, channels, 5] containing amplitude, bandwidth_GHz,
    phase_rad, delay_ns, and dimensionless chirp.
    """
    center = frequency_hz.mean()
    offset_ghz = (frequency_hz - center) / 1e9
    amplitude, bandwidth, phase0, delay_ns, chirp = parameters.unbind(dim=-1)
    x = offset_ghz[None, None, :]
    magnitude = amplitude[..., None] * torch.exp(-((x / bandwidth[..., None]) ** 2))
    phase = (
        phase0[..., None]
        + 2.0 * torch.pi * delay_ns[..., None] * x
        + torch.pi * chirp[..., None] * x**2
    )
    return magnitude * torch.exp(1j * phase)
