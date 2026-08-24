"""Regression and Bland-Altman metrics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def regression_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float | int]:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape or reference.ndim != 1 or reference.size == 0:
        raise ValueError(
            "reference and estimate must be non-empty one-dimensional arrays of equal size"
        )
    difference = estimate - reference
    bias = float(np.mean(difference))
    difference_sd = float(np.std(difference, ddof=1)) if len(difference) > 1 else 0.0
    residual_sum = float(np.sum(difference**2))
    total_sum = float(np.sum((reference - reference.mean()) ** 2))
    r_squared = float(1.0 - residual_sum / total_sum) if total_sum > 0 else float("nan")
    return {
        "n": int(reference.size),
        "mae_mm": float(np.mean(np.abs(difference))),
        "rmse_mm": float(np.sqrt(np.mean(difference**2))),
        "r_squared": r_squared,
        "bias_mm": bias,
        "difference_sd_mm": difference_sd,
        "loa_lower_mm": bias - 1.96 * difference_sd,
        "loa_upper_mm": bias + 1.96 * difference_sd,
    }


def aggregate_by_recording(
    reference: np.ndarray,
    estimate: np.ndarray,
    recording_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, recording_id in enumerate(np.asarray(recording_ids).astype(str)):
        groups[recording_id].append(index)
    ordered_ids = np.asarray(sorted(groups), dtype="U")
    aggregated_reference = []
    aggregated_estimate = []
    for recording_id in ordered_ids:
        indices = groups[str(recording_id)]
        references = np.asarray(reference)[indices]
        if not np.allclose(references, references[0]):
            raise ValueError(f"Recording {recording_id} has inconsistent caliper labels")
        aggregated_reference.append(float(references[0]))
        aggregated_estimate.append(float(np.mean(np.asarray(estimate)[indices])))
    return np.asarray(aggregated_reference), np.asarray(aggregated_estimate), ordered_ids
