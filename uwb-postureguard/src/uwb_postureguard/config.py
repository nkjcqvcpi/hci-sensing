"""Typed project configuration with TOML loading."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class FeatureConfig:
    window_size: int = 5
    iqr_factor: float = 1.5
    drop_average_features: bool = True
    derive_cir_polar: bool = True

    def validate(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if self.iqr_factor <= 0:
            raise ValueError("iqr_factor must be positive")


@dataclass(slots=True)
class SplitConfig:
    strategy: str = "frame_stratified"
    test_size: float = 0.40
    validation_size: float = 0.20
    random_state: int = 42

    def validate(self) -> None:
        if self.strategy not in {"frame_stratified", "group"}:
            raise ValueError("split strategy must be 'frame_stratified' or 'group'")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")


@dataclass(slots=True)
class ModelConfig:
    num_leaves: int = 64
    learning_rate: float = 0.05
    feature_fraction: float = 0.90
    max_rounds: int = 1000
    early_stopping_rounds: int = 10
    n_jobs: int = -1

    def validate(self) -> None:
        if self.num_leaves < 2:
            raise ValueError("num_leaves must be at least 2")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 < self.feature_fraction <= 1:
            raise ValueError("feature_fraction must be in (0, 1]")
        if self.max_rounds < 1 or self.early_stopping_rounds < 1:
            raise ValueError("training round counts must be positive")


@dataclass(slots=True)
class OODConfig:
    enabled: bool = True
    nu: float = 0.05
    score_quantile: float = 0.05
    svd_components: int = 64
    max_train_samples: int = 5000

    def validate(self) -> None:
        if not 0 < self.nu < 1:
            raise ValueError("nu must be between 0 and 1")
        if not 0 <= self.score_quantile < 1:
            raise ValueError("score_quantile must be in [0, 1)")
        if self.svd_components < 1 or self.max_train_samples < 2:
            raise ValueError("OOD dimensionality and sample limits must be positive")


@dataclass(slots=True)
class TrainingConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ood: OODConfig = field(default_factory=OODConfig)

    def validate(self) -> None:
        self.features.validate()
        self.split.validate()
        self.model.validate()
        self.ood.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_toml(cls, path: str | Path) -> TrainingConfig:
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
        config = cls(
            features=FeatureConfig(**raw.get("features", {})),
            split=SplitConfig(**raw.get("split", {})),
            model=ModelConfig(**raw.get("model", {})),
            ood=OODConfig(**raw.get("ood", {})),
        )
        config.validate()
        return config
