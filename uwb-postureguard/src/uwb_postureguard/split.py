"""Reproducible frame-level and recording-level data splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import SplitConfig


@dataclass(slots=True)
class DatasetSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def _can_stratify(labels: pd.Series, holdout_size: float) -> bool:
    counts = labels.value_counts()
    holdout_count = int(np.ceil(len(labels) * holdout_size))
    return bool((counts >= 2).all() and holdout_count >= len(counts))


def make_split(labels: pd.Series, sessions: pd.Series, config: SplitConfig) -> DatasetSplit:
    """Split temporal samples using the manuscript protocol or disjoint recordings."""

    config.validate()
    indices = np.arange(len(labels))
    if config.strategy == "frame_stratified":
        stratify = labels if _can_stratify(labels, config.test_size) else None
        train_validation, test = train_test_split(
            indices,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=stratify,
        )
        remaining_labels = labels.iloc[train_validation]
        stratify_remaining = (
            remaining_labels if _can_stratify(remaining_labels, config.validation_size) else None
        )
        train, validation = train_test_split(
            train_validation,
            test_size=config.validation_size,
            random_state=config.random_state,
            stratify=stratify_remaining,
        )
    else:
        if sessions.nunique() < 3:
            raise ValueError("Group splitting requires at least three independent recordings")
        first = GroupShuffleSplit(
            n_splits=1, test_size=config.test_size, random_state=config.random_state
        )
        train_validation_pos, test_pos = next(first.split(indices, labels, groups=sessions))
        train_validation = indices[train_validation_pos]
        test = indices[test_pos]
        remaining_sessions = sessions.iloc[train_validation]
        if remaining_sessions.nunique() < 2:
            raise ValueError("Not enough training recordings remain for validation")
        second = GroupShuffleSplit(
            n_splits=1,
            test_size=config.validation_size,
            random_state=config.random_state,
        )
        train_pos, validation_pos = next(
            second.split(train_validation, labels.iloc[train_validation], groups=remaining_sessions)
        )
        train = train_validation[train_pos]
        validation = train_validation[validation_pos]

    return DatasetSplit(
        train=np.asarray(train, dtype=int),
        validation=np.asarray(validation, dtype=int),
        test=np.asarray(test, dtype=int),
    )
