"""Run posture training, prediction, validation, and synthetic-data tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from uwb_postureguard.config import TrainingConfig
from uwb_postureguard.data import class_counts, load_recordings
from uwb_postureguard.features import TemporalFeatureBuilder, select_frame_features
from uwb_postureguard.pipeline import PoseGuardBundle, train_from_path
from uwb_postureguard.synthetic import write_synthetic_csvs


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _add_input_arguments(parser: argparse.ArgumentParser, labels: bool = True) -> None:
    parser.add_argument("--input", required=True, help="CSV file or directory of recording CSVs")
    if labels:
        parser.add_argument("--label-column", help="Posture label column; inferred by default")
    parser.add_argument("--session-column", help="Optional recording/session column")
    parser.add_argument("--frame-column", help="Optional frame-order column")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train PoseGBDT and leaf-embedding OOD detector")
    _add_input_arguments(train)
    train.add_argument("--config", help="TOML configuration; built-in defaults otherwise")
    train.add_argument("--artifact", required=True, help="Output .joblib model bundle")
    train.add_argument("--report", help="Output JSON report; defaults beside the artifact")

    predict = subparsers.add_parser("predict", help="Run temporal posture and OOD inference")
    _add_input_arguments(predict)
    predict.add_argument("--model", required=True, help="Trained .joblib model bundle")
    predict.add_argument("--output", required=True, help="Prediction CSV")

    validate = subparsers.add_parser("validate-input", help="Validate schema and feature coverage")
    _add_input_arguments(validate)
    validate.add_argument("--config", help="TOML configuration; built-in defaults otherwise")

    synthetic = subparsers.add_parser(
        "make-synthetic", help="Create optional non-human smoke-test recordings"
    )
    synthetic.add_argument("--output", required=True, help="Output directory")
    synthetic.add_argument("--classes", type=int, default=19)
    synthetic.add_argument("--sessions-per-class", type=int, default=3)
    synthetic.add_argument("--frames-per-session", type=int, default=20)
    synthetic.add_argument("--random-state", type=int, default=7)
    return parser


def _config(path: str | None) -> TrainingConfig:
    return TrainingConfig.from_toml(path) if path else TrainingConfig()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        config = _config(args.config)
        bundle = train_from_path(
            args.input,
            config,
            label_column=args.label_column,
            session_column=args.session_column,
            frame_column=args.frame_column,
        )
        bundle.save(args.artifact)
        report = args.report or str(Path(args.artifact).with_suffix(".report.json"))
        _write_json(report, bundle.training_summary)
        print(json.dumps({"artifact": args.artifact, "report": report}, indent=2))
        return 0

    if args.command == "predict":
        bundle = PoseGuardBundle.load(args.model)
        predictions = bundle.predict_path(
            args.input,
            label_column=args.label_column,
            session_column=args.session_column,
            frame_column=args.frame_column,
        )
        destination = Path(args.output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(destination, index=False)
        print(json.dumps({"predictions": str(destination), "rows": len(predictions)}, indent=2))
        return 0

    if args.command == "validate-input":
        config = _config(args.config)
        frame = load_recordings(
            args.input,
            label_column=args.label_column,
            session_column=args.session_column,
            frame_column=args.frame_column,
            require_labels=True,
        )
        selected, groups = select_frame_features(
            frame,
            drop_average_features=config.features.drop_average_features,
            derive_cir_polar=config.features.derive_cir_polar,
        )
        feature_columns = [column for columns in groups.values() for column in columns]
        temporal = TemporalFeatureBuilder(config.features.window_size).transform(
            selected, feature_columns
        )
        print(
            json.dumps(
                {
                    "raw_frames": len(frame),
                    "temporal_frames": len(temporal.X),
                    "recordings": int(temporal.sessions.nunique()),
                    "class_counts": class_counts(frame),
                    "feature_groups": {name: len(columns) for name, columns in groups.items()},
                    "temporal_features": temporal.X.shape[1],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    paths = write_synthetic_csvs(
        args.output,
        class_count=args.classes,
        sessions_per_class=args.sessions_per_class,
        frames_per_session=args.frames_per_session,
        random_state=args.random_state,
    )
    print(json.dumps({"directory": args.output, "recordings": len(paths)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
