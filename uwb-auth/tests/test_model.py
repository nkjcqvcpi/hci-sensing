from __future__ import annotations

import numpy as np
import pandas as pd

from uwb_auth.model import run_verification


def _partition(seed: int, samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for subject, center in ((2, -2.0), (3, 2.0)):
        for index in range(samples):
            rows.append(
                {
                    "subject": subject,
                    "i__signal__median": center + rng.normal(0, 0.15),
                    "r__signal__median": center + rng.normal(0, 0.15),
                    "experiment": f"{seed}-{subject}-{index}",
                }
            )
    return pd.DataFrame(rows)


def test_calibration_and_test_are_separate():
    report = run_verification(
        _partition(1, 20),
        _partition(2, 10),
        _partition(3, 12),
        ["i__signal__median", "r__signal__median"],
        (2, 3),
        regularization=1.0,
        random_state=42,
    )

    assert report["test"]["overall"]["identification_accuracy"] == 1.0
    assert report["test"]["overall"]["macro_balanced_accuracy"] == 1.0
    assert set(report["calibration"]) == {"subject-02", "subject-03"}
