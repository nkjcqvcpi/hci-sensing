from __future__ import annotations

import pandas as pd

from uwb_auth.metadata import normalize_metadata_sheet


def test_metadata_loader_uses_only_first_three_anonymized_columns():
    raw = pd.DataFrame(
        [
            ["ExpName", "SubjectNo", "Posture", *([None] * 15)],
            [None] * 18,
            ["250602-10_00_00", 2, " Upright ", "TWS", *([None] * 14)],
            [None, 2, "walk", "TWS", *([None] * 14)],
        ]
    )

    table = normalize_metadata_sheet(raw, "labels")

    assert list(table.columns) == [
        "experiment",
        "subject",
        "posture",
        "collection_date",
        "source_sheet",
        "condition_id",
    ]
    assert table.loc[0, "experiment"] == "250602-10_00_00"
    assert table.loc[0, "subject"] == 2
    assert table.loc[0, "posture"] == "upright"
    assert len(table.loc[0, "condition_id"]) == 12
