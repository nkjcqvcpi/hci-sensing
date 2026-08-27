"""Typed experiment configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    sheet: str = "labels"
    collection_date: str = "250602"
    subjects: tuple[int, ...] = (2, 3)
    folds: int = 3
    random_state: int = 42
    regularization: float = 1.0

    def validate(self) -> None:
        if len(self.subjects) < 2 or len(set(self.subjects)) != len(self.subjects):
            raise ValueError("At least two unique subjects are required")
        if self.folds < 3:
            raise ValueError("At least three folds are required for disjoint calibration")
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")

    @classmethod
    def from_toml(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).expanduser().open("rb") as stream:
            raw = tomllib.load(stream)
        data = raw.get("data", {})
        protocol = raw.get("protocol", {})
        model = raw.get("model", {})
        config = cls(
            sheet=str(data.get("sheet", "labels")),
            collection_date=str(data.get("collection_date", "250602")),
            subjects=tuple(int(subject) for subject in data.get("subjects", [2, 3])),
            folds=int(protocol.get("folds", 3)),
            random_state=int(protocol.get("random_state", 42)),
            regularization=float(model.get("regularization", 1.0)),
        )
        config.validate()
        return config
