"""Input loading for preprocessed UWB frame CSV files."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .taxonomy import normalize_posture

SESSION_COLUMN = "__session__"
FRAME_COLUMN = "__frame__"
LABEL_COLUMN = "__label__"

_LABEL_ALIASES = ("posture", "label", "target", "class", "y")
_FRAME_ALIASES = ("seq_cnt", "frame", "frame_index", "frame_id", "timestamp")


def _resolve_column(
    columns: Iterable[str], requested: str | None, aliases: Iterable[str]
) -> str | None:
    by_lower = {str(column).lower(): str(column) for column in columns}
    if requested:
        if requested in columns:
            return requested
        match = by_lower.get(requested.lower())
        if match:
            return match
        raise ValueError(f"Column {requested!r} was not found")
    for alias in aliases:
        if alias in by_lower:
            return by_lower[alias]
    return None


def _csv_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    paths = sorted(path for path in input_path.rglob("*.csv") if path.is_file())
    if not paths:
        raise ValueError(f"No CSV files found under {input_path}")
    return paths


def load_recordings(
    input_path: str | Path,
    *,
    label_column: str | None = None,
    session_column: str | None = None,
    frame_column: str | None = None,
    require_labels: bool = True,
) -> pd.DataFrame:
    """Load one CSV or a directory of CSV recordings into a canonical frame table.

    Every file is treated as an independent recording. If a session column is supplied,
    its values are nested under the file identity so windows never cross file boundaries.
    """

    root = Path(input_path).expanduser().resolve()
    paths = _csv_paths(root)
    base = root if root.is_dir() else root.parent
    recordings: list[pd.DataFrame] = []

    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame.columns = [str(column).strip() for column in frame.columns]

        found_label = _resolve_column(frame.columns, label_column, _LABEL_ALIASES)
        if require_labels and found_label is None:
            raise ValueError(f"No posture label column found in {path}")

        found_session = (
            _resolve_column(frame.columns, session_column, ()) if session_column else None
        )
        found_frame = _resolve_column(frame.columns, frame_column, _FRAME_ALIASES)
        file_id = str(path.relative_to(base).with_suffix(""))

        canonical = frame.copy()
        if found_session:
            canonical[SESSION_COLUMN] = file_id + "::" + canonical[found_session].astype(str)
        else:
            canonical[SESSION_COLUMN] = file_id
        if found_frame:
            canonical[FRAME_COLUMN] = pd.to_numeric(canonical[found_frame], errors="coerce")
            fallback = pd.Series(np.arange(len(canonical)), index=canonical.index)
            canonical[FRAME_COLUMN] = canonical[FRAME_COLUMN].fillna(fallback)
        else:
            canonical[FRAME_COLUMN] = np.arange(len(canonical))
        if found_label:
            try:
                canonical[LABEL_COLUMN] = canonical[found_label].map(normalize_posture)
            except ValueError as error:
                raise ValueError(f"Invalid posture label in {path}: {error}") from error
        recordings.append(canonical)

    if not recordings:
        raise ValueError(f"All CSV files under {root} were empty")
    return pd.concat(recordings, ignore_index=True, sort=False)


def class_counts(frame: pd.DataFrame) -> dict[int, int]:
    if LABEL_COLUMN not in frame:
        return {}
    counts = frame[LABEL_COLUMN].value_counts().sort_index()
    return {int(label): int(count) for label, count in counts.items()}
