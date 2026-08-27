"""Calibrated one-claim-at-a-time verification over recording-level features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def _threshold_at_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    false_positive, true_positive, thresholds = roc_curve(labels, scores)
    false_negative = 1.0 - true_positive
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise ValueError("No finite verification threshold is available")
    positions = np.flatnonzero(finite)
    position = int(positions[np.argmin(np.abs(false_positive[finite] - false_negative[finite]))])
    eer = (false_positive[position] + false_negative[position]) / 2.0
    return float(thresholds[position]), float(eer)


def _claim_metrics(
    actual_subjects: np.ndarray,
    scores: np.ndarray,
    claimant: int,
    threshold: float,
) -> dict[str, float | int]:
    genuine = actual_subjects == claimant
    accepted = scores >= threshold
    false_accepts = int(accepted[~genuine].sum())
    false_rejects = int((~accepted[genuine]).sum())
    false_acceptance_rate = false_accepts / int((~genuine).sum())
    false_rejection_rate = false_rejects / int(genuine.sum())
    balanced = float(balanced_accuracy_score(genuine, accepted))
    auc = float(roc_auc_score(genuine, scores))
    _, test_eer = _threshold_at_eer(genuine.astype(int), scores)
    return {
        "genuine_trials": int(genuine.sum()),
        "impostor_trials": int((~genuine).sum()),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "false_acceptance_rate": false_acceptance_rate,
        "false_rejection_rate": false_rejection_rate,
        "balanced_accuracy": balanced,
        "roc_auc": auc,
        "descriptive_eer": test_eer,
    }


def _macro(per_claim: dict[str, dict[str, float | int]]) -> dict[str, float]:
    metrics = (
        "false_acceptance_rate",
        "false_rejection_rate",
        "balanced_accuracy",
        "roc_auc",
        "descriptive_eer",
    )
    return {
        f"macro_{metric}": float(np.mean([claim[metric] for claim in per_claim.values()]))
        for metric in metrics
    }


def run_verification(
    enrollment: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    subjects: tuple[int, ...],
    *,
    regularization: float,
    random_state: int,
) -> dict[str, Any]:
    """Fit identities on enrollment, calibrate on validation, and score test once."""

    classifier = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", RobustScaler(quantile_range=(10.0, 90.0))),
            (
                "classifier",
                LogisticRegression(
                    C=regularization,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    classifier.fit(enrollment[columns], enrollment["subject"])
    classes = [int(value) for value in classifier.named_steps["classifier"].classes_]
    if set(classes) != set(subjects):
        raise ValueError("Enrollment does not contain every configured subject")

    validation_probabilities = classifier.predict_proba(validation[columns])
    test_probabilities = classifier.predict_proba(test[columns])
    validation_actual = validation["subject"].to_numpy(dtype=int)
    test_actual = test["subject"].to_numpy(dtype=int)
    thresholds: dict[int, float] = {}
    calibration: dict[str, dict[str, float | int]] = {}
    test_scores: dict[int, np.ndarray] = {}

    for claimant in subjects:
        column = classes.index(claimant)
        validation_scores = validation_probabilities[:, column]
        threshold, validation_eer = _threshold_at_eer(
            (validation_actual == claimant).astype(int), validation_scores
        )
        thresholds[claimant] = threshold
        calibration[f"subject-{claimant:02d}"] = {
            "threshold": threshold,
            "validation_eer": validation_eer,
            "genuine_trials": int((validation_actual == claimant).sum()),
            "impostor_trials": int((validation_actual != claimant).sum()),
        }
        test_scores[claimant] = test_probabilities[:, column]

    per_claim = {
        f"subject-{claimant:02d}": _claim_metrics(
            test_actual, test_scores[claimant], claimant, thresholds[claimant]
        )
        for claimant in subjects
    }
    predicted = np.asarray(classes)[np.argmax(test_probabilities, axis=1)]
    overall: dict[str, float | int | dict[str, list[float]]] = _macro(per_claim)
    overall["identification_accuracy"] = float(accuracy_score(test_actual, predicted))
    overall["identification_correct"] = int((test_actual == predicted).sum())
    overall["test_recordings"] = len(test)
    overall["verification_decisions"] = len(test) * len(subjects)
    return {
        "feature_count": len(columns),
        "calibration": calibration,
        "test": {"per_claim": per_claim, "overall": overall},
    }
