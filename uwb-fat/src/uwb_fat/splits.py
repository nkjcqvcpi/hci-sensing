"""Participant-level split logic that prevents window leakage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def participant_loso_split(
    participant_ids: np.ndarray, held_out: str
) -> tuple[np.ndarray, np.ndarray]:
    participant_ids = np.asarray(participant_ids).astype(str)
    test = np.flatnonzero(participant_ids == str(held_out))
    train = np.flatnonzero(participant_ids != str(held_out))
    if len(test) == 0:
        raise ValueError(f"Held-out participant {held_out!r} does not exist")
    if len(np.unique(participant_ids[train])) < 2:
        raise ValueError(
            "At least two outer-training participants are required for inner validation"
        )
    return train, test


def inner_participant_split(
    participant_ids: np.ndarray,
    outer_train: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    participant_ids = np.asarray(participant_ids).astype(str)
    participants = np.unique(participant_ids[outer_train])
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(participants)
    n_validation = max(1, int(round(validation_fraction * len(participants))))
    n_validation = min(n_validation, len(participants) - 1)
    validation_participants = set(shuffled[:n_validation])
    validation_mask = np.asarray(
        [participant_ids[index] in validation_participants for index in outer_train]
    )
    validation = outer_train[validation_mask]
    train = outer_train[~validation_mask]
    return train, validation


def full_loso_split(
    participant_ids: np.ndarray,
    held_out: str,
    validation_fraction: float = 0.2,
    seed: int = 2026,
) -> SplitIndices:
    outer_train, test = participant_loso_split(participant_ids, held_out)
    train, validation = inner_participant_split(
        participant_ids, outer_train, validation_fraction, seed
    )
    return SplitIndices(train=train, validation=validation, test=test)
