"""End-to-end training, serialization, evaluation, and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from .config import TrainingConfig
from .data import FRAME_COLUMN, LABEL_COLUMN, SESSION_COLUMN, class_counts, load_recordings
from .features import (
    IQRClipper,
    MedianImputer,
    TemporalFeatureBuilder,
    TemporalResult,
    select_frame_features,
)
from .model import PoseGBDTClassifier
from .ood import LeafOODDetector
from .split import make_split
from .taxonomy import POSTURES


def _metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    labels = sorted(set(true.tolist()) | set(predicted.tolist()))
    names = [POSTURES[label] for label in labels]
    report = classification_report(
        true,
        predicted,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0.0,
    )
    report["accuracy"] = float(accuracy_score(true, predicted))
    return report


@dataclass
class PoseGuardBundle:
    config: TrainingConfig
    base_feature_columns: list[str]
    feature_groups: dict[str, list[str]]
    temporal_builder: TemporalFeatureBuilder
    clipper: IQRClipper
    imputer: MedianImputer
    classifier: PoseGBDTClassifier
    ood_detector: LeafOODDetector | None
    training_summary: dict[str, Any]

    def _temporal_from_frame(self, frame: pd.DataFrame) -> TemporalResult:
        selected, _ = select_frame_features(
            frame,
            drop_average_features=self.config.features.drop_average_features,
            derive_cir_polar=self.config.features.derive_cir_polar,
        )
        missing = [column for column in self.base_feature_columns if column not in selected]
        if missing:
            raise ValueError(
                "Input is missing features used during training: " + ", ".join(missing[:12])
            )
        columns = self.base_feature_columns + [SESSION_COLUMN, FRAME_COLUMN]
        if LABEL_COLUMN in selected:
            columns.append(LABEL_COLUMN)
        return self.temporal_builder.transform(selected[columns], self.base_feature_columns)

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, TemporalResult]:
        temporal = self._temporal_from_frame(frame)
        clipped = self.clipper.transform(temporal.X)
        return self.imputer.transform(clipped), temporal

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        X, temporal = self.transform(frame)
        predicted = self.classifier.predict(X).astype(int)
        probabilities = self.classifier.predict_proba(X)
        confidence = probabilities.max(axis=1)
        class_to_position = {
            int(label): position for position, label in enumerate(self.classifier.classes_)
        }

        if self.ood_detector:
            ood_score = self.ood_detector.score_samples(self.classifier, X)
            is_ood = ood_score < float(self.ood_detector.threshold_)
        else:
            ood_score = np.full(len(X), np.nan)
            is_ood = np.zeros(len(X), dtype=bool)

        output = pd.DataFrame(
            {
                "session": temporal.sessions,
                "frame": temporal.frames,
                "predicted_id": predicted,
                "predicted_posture": [POSTURES[label] for label in predicted],
                "confidence": confidence,
                "is_ood": is_ood,
                "ood_score": ood_score,
            }
        )
        for label, position in class_to_position.items():
            output[f"probability_{label}_{POSTURES[label].replace(' ', '_')}"] = probabilities[
                :, position
            ]
        if temporal.labels is not None:
            truth = temporal.labels.astype(int).to_numpy()
            output["true_id"] = truth
            output["true_posture"] = [POSTURES[label] for label in truth]
            output["correct"] = predicted == truth
        return output

    def predict_path(
        self,
        input_path: str | Path,
        *,
        label_column: str | None = None,
        session_column: str | None = None,
        frame_column: str | None = None,
    ) -> pd.DataFrame:
        frame = load_recordings(
            input_path,
            label_column=label_column,
            session_column=session_column,
            frame_column=frame_column,
            require_labels=False,
        )
        return self.predict_frame(frame)

    def save(self, path: str | Path) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> PoseGuardBundle:
        bundle = joblib.load(Path(path).expanduser().resolve())
        if not isinstance(bundle, cls):
            raise TypeError("Artifact is not a PoseGuardBundle")
        return bundle


def train_from_frame(frame: pd.DataFrame, config: TrainingConfig | None = None) -> PoseGuardBundle:
    config = config or TrainingConfig()
    config.validate()
    selected, feature_groups = select_frame_features(
        frame,
        drop_average_features=config.features.drop_average_features,
        derive_cir_polar=config.features.derive_cir_polar,
    )
    base_feature_columns = [
        column
        for column in selected.columns
        if column not in {SESSION_COLUMN, FRAME_COLUMN, LABEL_COLUMN}
    ]
    temporal_builder = TemporalFeatureBuilder(config.features.window_size)
    temporal = temporal_builder.transform(selected, base_feature_columns)
    if temporal.labels is None:
        raise ValueError("Training data requires posture labels")
    if temporal.labels.nunique() < 2:
        raise ValueError("Training requires at least two posture classes")

    split = make_split(temporal.labels, temporal.sessions, config.split)
    X_train_raw = temporal.X.iloc[split.train]
    X_validation_raw = temporal.X.iloc[split.validation]
    X_test_raw = temporal.X.iloc[split.test]
    y_train = temporal.labels.iloc[split.train]
    y_validation = temporal.labels.iloc[split.validation]
    y_test = temporal.labels.iloc[split.test]

    clipper = IQRClipper(config.features.iqr_factor)
    X_train_clipped = clipper.fit_transform(X_train_raw)
    X_validation_clipped = clipper.transform(X_validation_raw)
    X_test_clipped = clipper.transform(X_test_raw)

    imputer = MedianImputer()
    X_train = imputer.fit_transform(X_train_clipped)
    X_validation = imputer.transform(X_validation_clipped)
    X_test = imputer.transform(X_test_clipped)

    classifier = PoseGBDTClassifier(config.model, config.split.random_state)
    classifier.fit(X_train, y_train, X_validation, y_validation)
    test_prediction = classifier.predict(X_test).astype(int)

    ood_detector = None
    known_test_ood_rate = None
    if config.ood.enabled:
        ood_detector = LeafOODDetector(config.ood, config.split.random_state)
        ood_detector.fit(classifier, X_train)
        known_test_ood_rate = float(ood_detector.is_ood(classifier, X_test).mean())

    summary: dict[str, Any] = {
        "protocol": {
            "window_size": config.features.window_size,
            "split_strategy": config.split.strategy,
            "test_size": config.split.test_size,
            "validation_size_within_training_partition": config.split.validation_size,
            "iqr_policy": "training-fitted winsorization",
        },
        "frames": {
            "raw": len(frame),
            "temporal": len(temporal.X),
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "class_counts": class_counts(frame),
        "base_feature_count": len(base_feature_columns),
        "temporal_feature_count": int(temporal.X.shape[1]),
        "feature_groups": {name: len(columns) for name, columns in feature_groups.items()},
        "best_iteration": classifier.best_iteration_,
        "known_test_ood_rate": known_test_ood_rate,
        "test_metrics": _metrics(y_test, test_prediction),
    }
    return PoseGuardBundle(
        config=config,
        base_feature_columns=base_feature_columns,
        feature_groups=feature_groups,
        temporal_builder=temporal_builder,
        clipper=clipper,
        imputer=imputer,
        classifier=classifier,
        ood_detector=ood_detector,
        training_summary=summary,
    )


def train_from_path(
    input_path: str | Path,
    config: TrainingConfig | None = None,
    *,
    label_column: str | None = None,
    session_column: str | None = None,
    frame_column: str | None = None,
) -> PoseGuardBundle:
    frame = load_recordings(
        input_path,
        label_column=label_column,
        session_column=session_column,
        frame_column=frame_column,
        require_labels=True,
    )
    return train_from_frame(frame, config)
