"""End-to-end UWBAuth experiment orchestration."""

from __future__ import annotations

import json
import math
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from .config import ExperimentConfig
from .features import extract_feature_table, feature_columns
from .metadata import load_metadata
from .model import run_verification
from .protocol import CrossConditionProtocol, ProtocolFold, make_protocol


def _attach_features(partition: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    feature_only = features.drop(
        columns=[
            "subject",
            "posture",
            "collection_date",
            "source_sheet",
            "condition_id",
        ],
        errors="ignore",
    )
    merged = partition.merge(feature_only, on="experiment", how="left", validate="one_to_one")
    if merged.filter(regex=r"^[ir]__").isna().all(axis=1).any():
        raise ValueError("At least one protocol recording has no extracted features")
    return merged


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        f"subject-{int(subject):02d}": int(count)
        for subject, count in frame["subject"].value_counts().sort_index().items()
    }


def _protocol_report(
    protocol: CrossConditionProtocol, config: ExperimentConfig
) -> dict[str, Any]:
    return {
        "threat_model": "claimed-identity verification against zero-effort impostors",
        "unit_of_analysis": "recording",
        "collection_date": config.collection_date,
        "subjects": [f"subject-{subject:02d}" for subject in config.subjects],
        "folds": config.folds,
        "paired_nuisance_conditions": len(protocol.shared_conditions),
        "eligible_postures": list(protocol.eligible_postures),
        "partition_policy": (
            "paired nuisance-condition groups are disjoint among enrollment, validation, and test"
        ),
        "threshold_policy": "per-claimant threshold chosen at validation equal-error point",
        "test_coverage": "each selected recording appears in the test partition exactly once",
    }


def _fold_counts(fold: ProtocolFold) -> dict[str, dict[str, int]]:
    return {
        "enrollment": _counts(fold.enrollment),
        "validation": _counts(fold.validation),
        "test": _counts(fold.test),
    }


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> list[float]:
    if trials == 0:
        return [float("nan"), float("nan")]
    rate = successes / trials
    denominator = 1 + z**2 / trials
    center = (rate + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / trials + z**2 / (4 * trials**2)) / denominator
    return [center - margin, center + margin]


def _aggregate_folds(folds: list[dict[str, Any]], subjects: tuple[int, ...]) -> dict[str, Any]:
    per_claim: dict[str, dict[str, float | int | list[float]]] = {}
    for subject in subjects:
        key = f"subject-{subject:02d}"
        claims = [fold["test"]["per_claim"][key] for fold in folds]
        genuine = sum(int(claim["genuine_trials"]) for claim in claims)
        impostor = sum(int(claim["impostor_trials"]) for claim in claims)
        false_accepts = sum(int(claim["false_accepts"]) for claim in claims)
        false_rejects = sum(int(claim["false_rejects"]) for claim in claims)
        far = false_accepts / impostor
        frr = false_rejects / genuine
        per_claim[key] = {
            "genuine_trials": genuine,
            "impostor_trials": impostor,
            "false_accepts": false_accepts,
            "false_rejects": false_rejects,
            "false_acceptance_rate": far,
            "false_rejection_rate": frr,
            "balanced_accuracy": 1.0 - (far + frr) / 2.0,
            "mean_fold_roc_auc": float(np.mean([claim["roc_auc"] for claim in claims])),
            "mean_fold_descriptive_eer": float(
                np.mean([claim["descriptive_eer"] for claim in claims])
            ),
            "false_acceptance_95_percent_ci": _wilson_interval(false_accepts, impostor),
            "false_rejection_95_percent_ci": _wilson_interval(false_rejects, genuine),
        }

    fars = [float(claim["false_acceptance_rate"]) for claim in per_claim.values()]
    frrs = [float(claim["false_rejection_rate"]) for claim in per_claim.values()]
    correct = sum(int(fold["test"]["overall"]["identification_correct"]) for fold in folds)
    recordings = sum(int(fold["test"]["overall"]["test_recordings"]) for fold in folds)
    return {
        "per_claim": per_claim,
        "overall": {
            "macro_false_acceptance_rate": float(np.mean(fars)),
            "macro_false_rejection_rate": float(np.mean(frrs)),
            "macro_balanced_accuracy": float(1.0 - (np.mean(fars) + np.mean(frrs)) / 2.0),
            "mean_fold_macro_roc_auc": float(
                np.mean([fold["test"]["overall"]["macro_roc_auc"] for fold in folds])
            ),
            "mean_fold_macro_descriptive_eer": float(
                np.mean([fold["test"]["overall"]["macro_descriptive_eer"] for fold in folds])
            ),
            "identification_accuracy": correct / recordings,
            "test_recordings": recordings,
            "verification_decisions": recordings * len(subjects),
        },
    }


def run_experiment(
    *,
    data_root: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
    config: ExperimentConfig | None = None,
) -> dict[str, Any]:
    config = config or ExperimentConfig()
    config.validate()
    metadata = load_metadata(labels_path, config.sheet)
    protocol = make_protocol(metadata, config)
    selected = (
        pd.concat(
            [
                partition
                for fold in protocol.folds
                for partition in (fold.enrollment, fold.validation, fold.test)
            ],
            ignore_index=True,
        )
        .drop_duplicates("experiment")
        .reset_index(drop=True)
    )
    extracted = extract_feature_table(selected, data_root)
    attached_folds = [
        ProtocolFold(
            fold.fold,
            _attach_features(fold.enrollment, extracted),
            _attach_features(fold.validation, extracted),
            _attach_features(fold.test, extracted),
        )
        for fold in protocol.folds
    ]

    variants: dict[str, Any] = {}
    for name, roles in {
        "initiator_only": ("i",),
        "responder_only": ("r",),
        "dual_link": ("i", "r"),
    }.items():
        columns = feature_columns(extracted, roles)
        fold_results = [
            run_verification(
                fold.enrollment,
                fold.validation,
                fold.test,
                columns,
                config.subjects,
                regularization=config.regularization,
                random_state=config.random_state + fold.fold,
            )
            for fold in attached_folds
        ]
        variants[name] = {
            "feature_count": len(columns),
            "folds": fold_results,
            "aggregate_test": _aggregate_folds(fold_results, config.subjects),
        }

    report: dict[str, Any] = {
        "project": "UWBAuth",
        "status": "preliminary two-subject within-day condition-disjoint feasibility study",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": _protocol_report(protocol, config),
        "fold_recording_counts": [_fold_counts(fold) for fold in protocol.folds],
        "features": {
            "source": "range_i.csv and range_r.csv",
            "signals": [
                "nlos",
                "distance",
                "azimuth",
                "azimuth_fom",
                "elevation",
                "elevation_fom",
                "rssi",
                "pdoa1",
                "pdoa2",
            ],
            "statistics": ["median", "iqr", "mad", "std", "q10", "q90", "mean_abs_delta"],
            "collector_moving_averages_used": False,
        },
        "model": {
            "classifier": "L2-regularized logistic regression",
            "scaling": "enrollment-fitted robust scaling using 10th and 90th percentiles",
            "regularization": config.regularization,
            "random_state": config.random_state,
        },
        "variants": variants,
        "limitations": [
            "Only two subjects share enough matched nuisance conditions for this protocol.",
            "No second-day ranging data are present, so long-term persistence is not evaluated.",
            "The study evaluates zero-effort impostors, not replay, relay, or active spoofing.",
            "Collection environment and body geometry may confound identity.",
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": sys.platform,
        },
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
