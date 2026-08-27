"""Cross-day protocol with recording-disjoint enrollment, calibration, and testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class ProtocolFold:
    fold: int
    enrollment: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True, slots=True)
class CrossConditionProtocol:
    folds: tuple[ProtocolFold, ...]
    shared_conditions: tuple[str, ...]
    eligible_postures: tuple[str, ...]


def make_protocol(metadata: pd.DataFrame, config: ExperimentConfig) -> CrossConditionProtocol:
    """Create paired, condition-disjoint folds at recording granularity."""

    config.validate()
    subjects = set(config.subjects)
    selected = metadata[
        (metadata["source_sheet"] == config.sheet)
        & (metadata["collection_date"] == config.collection_date)
        & metadata["subject"].isin(subjects)
    ].copy()
    if selected.empty:
        raise ValueError("The configured collection partition is empty")

    shared_conditions: set[str] | None = None
    for subject in config.subjects:
        conditions = set(selected.loc[selected["subject"] == subject, "condition_id"])
        shared_conditions = conditions if shared_conditions is None else shared_conditions & conditions
    shared_conditions = shared_conditions or set()
    if len(shared_conditions) < config.folds:
        raise ValueError("Not enough paired nuisance conditions for the requested folds")

    eligible_postures: set[str] | None = None
    for subject in config.subjects:
        for condition in shared_conditions:
            postures = set(
                selected.loc[
                    (selected["subject"] == subject) & (selected["condition_id"] == condition),
                    "posture",
                ]
            )
            eligible_postures = (
                postures if eligible_postures is None else eligible_postures & postures
            )
    eligible_postures = eligible_postures or set()
    if len(eligible_postures) < 2:
        raise ValueError("Fewer than two postures are shared across subjects and conditions")

    selected = selected[
        selected["condition_id"].isin(shared_conditions)
        & selected["posture"].isin(eligible_postures)
    ].copy()
    rng = np.random.default_rng(config.random_state)
    shuffled = np.asarray(sorted(shared_conditions), dtype=object)
    rng.shuffle(shuffled)
    condition_groups = [
        tuple(str(value) for value in group) for group in np.array_split(shuffled, config.folds)
    ]
    folds: list[ProtocolFold] = []
    for fold_index in range(config.folds):
        test_conditions = set(condition_groups[fold_index])
        validation_conditions = set(condition_groups[(fold_index + 1) % config.folds])
        enrollment_conditions = shared_conditions - test_conditions - validation_conditions
        if not enrollment_conditions:
            raise ValueError("Each fold requires at least one enrollment condition")

        partitions = []
        for conditions in (enrollment_conditions, validation_conditions, test_conditions):
            partition = selected[selected["condition_id"].isin(conditions)]
            partitions.append(partition.sort_values("experiment").reset_index(drop=True))
        experiment_sets = [set(partition["experiment"]) for partition in partitions]
        if any(
            experiment_sets[left] & experiment_sets[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise AssertionError("Protocol partitions overlap within a fold")
        folds.append(ProtocolFold(fold_index + 1, *partitions))

    return CrossConditionProtocol(
        tuple(folds), tuple(sorted(shared_conditions)), tuple(sorted(eligible_postures))
    )
