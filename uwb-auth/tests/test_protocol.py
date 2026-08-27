from __future__ import annotations

import pandas as pd

from uwb_auth.config import ExperimentConfig
from uwb_auth.protocol import make_protocol


def _metadata() -> pd.DataFrame:
    rows = []
    for subject in (2, 3):
        for condition_index in range(6):
            for posture_index, posture in enumerate(("upright", "walk", "hunched")):
                rows.append(
                    {
                        "experiment": f"250602-{subject}{condition_index}_{posture_index:02d}_00",
                        "subject": subject,
                        "posture": posture,
                        "collection_date": "250602",
                        "source_sheet": "labels",
                        "condition_id": f"condition-{condition_index}",
                    }
                )
    return pd.DataFrame(rows)


def test_protocol_is_recording_and_condition_disjoint():
    protocol = make_protocol(_metadata(), ExperimentConfig(folds=3))

    tested = set()
    for fold in protocol.folds:
        enrollment = set(fold.enrollment["experiment"])
        validation = set(fold.validation["experiment"])
        test = set(fold.test["experiment"])
        assert not enrollment & validation
        assert not enrollment & test
        assert not validation & test
        condition_sets = [
            set(partition["condition_id"])
            for partition in (fold.enrollment, fold.validation, fold.test)
        ]
        assert not condition_sets[0] & condition_sets[1]
        assert not condition_sets[0] & condition_sets[2]
        assert not condition_sets[1] & condition_sets[2]
        tested.update(test)
    assert tested == set(_metadata()["experiment"])
    assert protocol.eligible_postures == ("hunched", "upright", "walk")
