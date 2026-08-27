"""Recording-level UWB features derived from initiator and responder ranging streams."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SIGNAL_COLUMNS = (
    "nlos",
    "distance",
    "azimuth",
    "azimuth_fom",
    "elevation",
    "elevation_fom",
    "rssi",
    "pdoa1",
    "pdoa2",
)
STATISTICS = ("median", "iqr", "mad", "std", "q10", "q90", "mean_abs_delta")


def _summarize(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return {statistic: float("nan") for statistic in STATISTICS}
    array = numeric.to_numpy(dtype=float)
    median = float(np.median(array))
    differences = np.abs(np.diff(array))
    return {
        "median": median,
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
        "mad": float(np.median(np.abs(array - median))),
        "std": float(np.std(array, ddof=0)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
        "mean_abs_delta": float(differences.mean()) if len(differences) else 0.0,
    }


def summarize_range_csv(path: str | Path, role: str) -> dict[str, float]:
    """Summarize a frame stream without using collector-provided moving averages."""

    frame = pd.read_csv(Path(path))
    missing = [column for column in SIGNAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if "seq_cnt" in frame:
        frame = frame.sort_values("seq_cnt", kind="stable")
    features: dict[str, float] = {}
    for column in SIGNAL_COLUMNS:
        for statistic, value in _summarize(frame[column]).items():
            features[f"{role}__{column}__{statistic}"] = value
    return features


def extract_recording(data_root: str | Path, experiment: str) -> dict[str, float]:
    recording = Path(data_root).expanduser().resolve() / experiment
    features: dict[str, float] = {}
    for role in ("i", "r"):
        path = recording / f"range_{role}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        features.update(summarize_range_csv(path, role))
    return features


def extract_feature_table(metadata: pd.DataFrame, data_root: str | Path) -> pd.DataFrame:
    """Create one independent feature vector per labeled recording."""

    rows: list[dict[str, object]] = []
    for record in metadata.itertuples(index=False):
        row: dict[str, object] = {
            "experiment": record.experiment,
            "subject": int(record.subject),
            "posture": record.posture,
            "collection_date": record.collection_date,
            "source_sheet": record.source_sheet,
        }
        row.update(extract_recording(data_root, str(record.experiment)))
        rows.append(row)
    if not rows:
        raise ValueError("No recordings were selected for feature extraction")
    return pd.DataFrame(rows)


def feature_columns(table: pd.DataFrame, roles: tuple[str, ...] = ("i", "r")) -> list[str]:
    prefixes = tuple(f"{role}__" for role in roles)
    columns = sorted(column for column in table.columns if column.startswith(prefixes))
    if not columns:
        raise ValueError(f"No features were found for roles {roles}")
    return columns
