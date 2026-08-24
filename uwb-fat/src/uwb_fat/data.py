"""Observation archive and PyTorch dataset utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ObservationArchive:
    REQUIRED = {
        "response",
        "frequencies_hz",
        "participant_id",
        "recording_id",
        "site",
        "caliper_mm",
    }

    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as archive:
            missing = self.REQUIRED.difference(archive.files)
            if missing:
                raise ValueError(f"Observation archive is missing arrays: {sorted(missing)}")
            for name in self.REQUIRED:
                setattr(self, name, np.asarray(archive[name]))
        n = self.response.shape[0]
        for name in ("participant_id", "recording_id", "site", "caliper_mm"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} length does not match response observations")
        if self.response.ndim != 3 or self.response.shape[1] != 2:
            raise ValueError("response must have shape [observation, 2, frequency]")


class ObservationDataset(Dataset):
    def __init__(
        self,
        response: np.ndarray,
        caliper_mm: np.ndarray,
        indices: np.ndarray | None = None,
    ):
        self.response = np.asarray(response)
        self.caliper_mm = np.asarray(caliper_mm)
        self.indices = np.arange(len(response)) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[position])
        return (
            torch.from_numpy(self.response[index].astype(np.complex64, copy=False)),
            torch.tensor(self.caliper_mm[index], dtype=torch.float32),
            index,
        )
