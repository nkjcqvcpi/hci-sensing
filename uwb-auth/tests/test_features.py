from __future__ import annotations

import pandas as pd
import pytest

from uwb_auth.features import SIGNAL_COLUMNS, extract_recording, summarize_range_csv


def _frame() -> pd.DataFrame:
    rows = []
    for index in range(5):
        rows.append({"seq_cnt": index, **{column: float(index + 1) for column in SIGNAL_COLUMNS}})
    return pd.DataFrame(rows)


def test_summary_is_recording_level_and_excludes_collector_averages(tmp_path):
    path = tmp_path / "range_i.csv"
    frame = _frame()
    frame["avg_distance"] = 9999.0
    frame.to_csv(path, index=False)

    features = summarize_range_csv(path, "i")

    assert features["i__distance__median"] == 3.0
    assert features["i__distance__iqr"] == 2.0
    assert features["i__distance__mean_abs_delta"] == 1.0
    assert all("avg_" not in name for name in features)


def test_extract_recording_requires_both_links(tmp_path):
    recording = tmp_path / "recording-1"
    recording.mkdir()
    _frame().to_csv(recording / "range_i.csv", index=False)

    with pytest.raises(FileNotFoundError):
        extract_recording(tmp_path, "recording-1")
