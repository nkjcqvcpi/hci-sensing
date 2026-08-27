"""UWB frame selection, denoising, and temporal features."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import FRAME_COLUMN, LABEL_COLUMN, SESSION_COLUMN

_RANGING_TERMS = ("distance", "azimuth", "elevation", "fom", "pdoa")
_QUALITY_TERMS = (
    "nlos",
    "first_path_idx",
    "1st_path",
    "main_path_idx",
    "snr",
    "rssi",
    "cir_main_pwr",
    "cir_1st_path_pwr",
    "cir_first_path_pwr",
    "noise_variance",
    "cfo",
    "aoa_phase",
)
_CIR_TOKEN = re.compile(r"(?:^|_)(?:re|real|im|imag|mag|magnitude|ang|angle|phase)(?:_|$)")


def _replace_token(name: str, old: str, new: str) -> str | None:
    pattern = re.compile(rf"(^|_){old}(_|$)", re.IGNORECASE)
    if not pattern.search(name):
        return None
    return pattern.sub(lambda match: f"{match.group(1)}{new}{match.group(2)}", name, count=1)


def derive_cir_polar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive magnitude and phase when matching real/imaginary CIR columns exist."""

    output = frame.copy()
    lookup = {column.lower(): column for column in output.columns}
    for real_column in list(output.columns):
        lower = real_column.lower()
        if "cir" not in lower:
            continue
        imag_name = _replace_token(real_column, "re", "im") or _replace_token(
            real_column, "real", "imag"
        )
        if imag_name is None or imag_name.lower() not in lookup:
            continue
        imag_column = lookup[imag_name.lower()]
        real = pd.to_numeric(output[real_column], errors="coerce")
        imag = pd.to_numeric(output[imag_column], errors="coerce")

        mag_name = _replace_token(real_column, "re", "mag") or _replace_token(
            real_column, "real", "mag"
        )
        phase_name = _replace_token(real_column, "re", "ang") or _replace_token(
            real_column, "real", "ang"
        )
        if mag_name and mag_name.lower() not in lookup:
            output[mag_name] = np.hypot(real, imag)
            lookup[mag_name.lower()] = mag_name
        if phase_name and phase_name.lower() not in lookup:
            output[phase_name] = np.arctan2(imag, real)
            lookup[phase_name.lower()] = phase_name
    return output


def feature_group(name: str) -> str | None:
    lower = name.lower()
    if lower in {SESSION_COLUMN, FRAME_COLUMN, LABEL_COLUMN}:
        return None
    if "cir" in lower and _CIR_TOKEN.search(lower):
        return "cir"
    if any(term in lower for term in _QUALITY_TERMS):
        return "signal_quality"
    if any(term in lower for term in _RANGING_TERMS):
        return "ranging"
    return None


def select_frame_features(
    frame: pd.DataFrame,
    *,
    drop_average_features: bool = True,
    derive_cir_polar: bool = True,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Select the ranging, signal-quality, and CIR feature families."""

    working = derive_cir_polar_features(frame) if derive_cir_polar else frame.copy()
    groups: dict[str, list[str]] = {"ranging": [], "signal_quality": [], "cir": []}
    selected: dict[str, pd.Series] = {}

    for column in working.columns:
        if drop_average_features and "avg_" in column.lower():
            continue
        group = feature_group(column)
        if group is None:
            continue
        numeric = pd.to_numeric(working[column], errors="coerce")
        if numeric.notna().any():
            selected[column] = numeric.astype(float)
            groups[group].append(column)

    if not selected:
        raise ValueError(
            "No UWB features were recognized. Expected ranging, signal-quality, or CIR columns."
        )
    output = pd.DataFrame(selected, index=working.index)
    for reserved in (SESSION_COLUMN, FRAME_COLUMN, LABEL_COLUMN):
        if reserved in working:
            output[reserved] = working[reserved].to_numpy()
    return output, groups


@dataclass(slots=True)
class TemporalResult:
    X: pd.DataFrame
    labels: pd.Series | None
    sessions: pd.Series
    frames: pd.Series


class TemporalFeatureBuilder:
    """Create current, lagged, and rolling-mean features without crossing recordings."""

    def __init__(self, window_size: int = 5):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = window_size

    def transform(self, frame: pd.DataFrame, feature_columns: Iterable[str]) -> TemporalResult:
        features = list(feature_columns)
        missing = [column for column in features if column not in frame]
        if missing:
            raise ValueError(f"Missing frame features: {missing[:8]}")
        if SESSION_COLUMN not in frame or FRAME_COLUMN not in frame:
            raise ValueError("Canonical session and frame columns are required")

        feature_blocks: list[pd.DataFrame] = []
        label_blocks: list[pd.Series] = []
        session_blocks: list[pd.Series] = []
        frame_blocks: list[pd.Series] = []

        for _, recording in frame.groupby(SESSION_COLUMN, sort=False, dropna=False):
            recording = recording.sort_values(FRAME_COLUMN, kind="stable").reset_index(drop=True)
            base = recording[features]
            parts = [base.add_suffix("__t0")]
            for lag in range(1, self.window_size):
                parts.append(base.shift(lag).add_suffix(f"__lag{lag}"))
            rolling = base.rolling(window=self.window_size, min_periods=self.window_size).mean()
            parts.append(rolling.add_suffix(f"__roll_mean_w{self.window_size}"))
            temporal = pd.concat(parts, axis=1)

            valid = np.arange(len(recording)) >= self.window_size - 1
            if not valid.any():
                continue
            feature_blocks.append(temporal.loc[valid].reset_index(drop=True))
            session_blocks.append(recording.loc[valid, SESSION_COLUMN].reset_index(drop=True))
            frame_blocks.append(recording.loc[valid, FRAME_COLUMN].reset_index(drop=True))
            if LABEL_COLUMN in recording:
                label_blocks.append(recording.loc[valid, LABEL_COLUMN].reset_index(drop=True))

        if not feature_blocks:
            raise ValueError(
                f"No recording contains the required {self.window_size} consecutive frames"
            )
        labels = pd.concat(label_blocks, ignore_index=True) if label_blocks else None
        return TemporalResult(
            X=pd.concat(feature_blocks, ignore_index=True),
            labels=labels,
            sessions=pd.concat(session_blocks, ignore_index=True),
            frames=pd.concat(frame_blocks, ignore_index=True),
        )


class IQRClipper:
    """Training-fitted IQR winsorization that preserves temporal frame alignment."""

    def __init__(self, factor: float = 1.5):
        if factor <= 0:
            raise ValueError("factor must be positive")
        self.factor = factor
        self.columns_: list[str] | None = None
        self.lower_: pd.Series | None = None
        self.upper_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> IQRClipper:
        self.columns_ = list(X.columns)
        q1 = X.quantile(0.25, numeric_only=True)
        q3 = X.quantile(0.75, numeric_only=True)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None or self.lower_ is None or self.upper_ is None:
            raise RuntimeError("IQRClipper must be fitted before transform")
        missing = [column for column in self.columns_ if column not in X]
        if missing:
            raise ValueError(f"Missing temporal features: {missing[:8]}")
        ordered = X[self.columns_].copy()
        return ordered.clip(lower=self.lower_, upper=self.upper_, axis="columns")

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


class MedianImputer:
    """Deterministic median imputation with zero fallback for all-missing columns."""

    def __init__(self):
        self.columns_: list[str] | None = None
        self.medians_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> MedianImputer:
        self.columns_ = list(X.columns)
        self.medians_ = X.median(numeric_only=True).reindex(self.columns_).fillna(0.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None or self.medians_ is None:
            raise RuntimeError("MedianImputer must be fitted before transform")
        missing = [column for column in self.columns_ if column not in X]
        if missing:
            raise ValueError(f"Missing temporal features: {missing[:8]}")
        ordered = X[self.columns_].replace([np.inf, -np.inf], np.nan)
        return ordered.fillna(self.medians_)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
