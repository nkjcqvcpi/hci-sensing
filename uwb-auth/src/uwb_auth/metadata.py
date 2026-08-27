"""Load only anonymized experiment labels needed by UWBAuth."""

from __future__ import annotations

import re
from hashlib import sha256
from json import dumps
from pathlib import Path

import numpy as np
import pandas as pd

_EXPERIMENT_PATTERN = re.compile(r"^\d{6}-\d{2}_\d{2}_\d{2}$")


def normalize_metadata_sheet(raw: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Normalize the first three columns of one project label sheet.

    Names and demographic sheets in the workbook are intentionally never loaded.
    """

    if raw.shape[1] < 18:
        raise ValueError(f"Sheet {sheet_name!r} must contain columns A through R")
    table = raw.iloc[2:, :3].copy()
    table.columns = ["experiment", "subject", "posture"]
    table["experiment"] = table["experiment"].astype("string").str.strip()
    table["subject"] = pd.to_numeric(table["subject"], errors="coerce")
    table["posture"] = table["posture"].astype("string").str.strip().str.lower()
    valid = (
        table["experiment"].map(lambda value: bool(_EXPERIMENT_PATTERN.fullmatch(str(value))))
        & table["subject"].notna()
        & table["posture"].notna()
    )
    table = table.loc[valid].copy()
    table["subject"] = table["subject"].astype(int)
    table["collection_date"] = table["experiment"].str[:6]
    table["source_sheet"] = sheet_name
    conditions = raw.iloc[2:, 3:18].loc[valid]
    table["condition_id"] = [
        sha256(dumps([_condition_value(value) for value in row]).encode()).hexdigest()[:12]
        for row in conditions.itertuples(index=False, name=None)
    ]
    return table.reset_index(drop=True)


def _condition_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return int(value) if float(value).is_integer() else float(value)
    return str(value).strip().lower()


def load_metadata(path: str | Path, sheet: str) -> pd.DataFrame:
    """Load anonymized labels and nuisance conditions from one experiment sheet."""

    workbook = Path(path).expanduser().resolve()
    raw = pd.read_excel(workbook, sheet_name=sheet, header=None, usecols="A:R")
    metadata = normalize_metadata_sheet(raw, sheet)
    duplicates = metadata["experiment"].duplicated(keep=False)
    if duplicates.any():
        names = metadata.loc[duplicates, "experiment"].unique().tolist()
        raise ValueError(f"Experiment identifiers are not unique: {names[:8]}")
    return metadata
