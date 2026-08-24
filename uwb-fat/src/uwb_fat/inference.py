"""Site-agnostic thickness-grid inference."""

from __future__ import annotations

from dataclasses import fields

import torch

from .config import BoundsConfig
from .model import Condition, UWBFatModel


def candidate_grid(bounds: BoundsConfig, device: torch.device | str = "cpu") -> torch.Tensor:
    step = bounds.grid_step_mm
    skin = torch.arange(bounds.skin_mm[0], bounds.skin_mm[1] + step / 2.0, step, device=device)
    fat = torch.arange(bounds.fat_mm[0], bounds.fat_mm[1] + step / 2.0, step, device=device)
    grid = torch.cartesian_prod(skin, fat)
    doubled_fold = 2.0 * grid.sum(dim=1)
    valid = (doubled_fold >= bounds.caliper_mm[0]) & (doubled_fold <= bounds.caliper_mm[1])
    return grid[valid]


def _repeat_one(condition: Condition, index: int, count: int) -> Condition:
    values = {}
    for field in fields(Condition):
        value = getattr(condition, field.name)[index : index + 1]
        values[field.name] = value.expand(count, *value.shape[1:])
    return Condition(**values)


@torch.no_grad()
def grid_search(
    model: UWBFatModel,
    response: torch.Tensor,
    frequency_hz: torch.Tensor,
    bounds: BoundsConfig,
    chunk_size: int = 512,
) -> dict[str, torch.Tensor]:
    """Apply Equation 9 independently to each observation."""
    model.eval()
    condition = model.condition(response)
    grid = candidate_grid(bounds, response.device)
    best_skin: list[torch.Tensor] = []
    best_fat: list[torch.Tensor] = []
    best_error: list[torch.Tensor] = []

    for batch_index in range(response.shape[0]):
        current_error = torch.tensor(float("inf"), device=response.device)
        current_pair = grid[0]
        for start in range(0, grid.shape[0], chunk_size):
            candidates = grid[start : start + chunk_size]
            repeated = _repeat_one(condition, batch_index, candidates.shape[0])
            predicted = model.reconstruct_from_condition(
                frequency_hz,
                repeated,
                skin_mm=candidates[:, 0],
                fat_mm=candidates[:, 1],
            )
            observed = response[batch_index : batch_index + 1].expand_as(predicted)
            errors = torch.mean(torch.abs(predicted - observed) ** 2, dim=(1, 2))
            value, local_index = torch.min(errors, dim=0)
            if value < current_error:
                current_error = value
                current_pair = candidates[local_index]
        best_skin.append(current_pair[0])
        best_fat.append(current_pair[1])
        best_error.append(current_error)

    skin = torch.stack(best_skin)
    fat = torch.stack(best_fat)
    return {
        "skin_mm": skin,
        "fat_mm": fat,
        "caliper_mm": 2.0 * (skin + fat),
        "reconstruction_mse": torch.stack(best_error),
    }
