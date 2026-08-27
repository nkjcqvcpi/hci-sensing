"""Run UWB-Fat preprocessing, training, evaluation, and conversion tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from uwb_fat.bodyfat import body_fat_percent, jackson_pollock_density
from uwb_fat.config import load_config
from uwb_fat.io import build_observation_archive, read_manifest
from uwb_fat.metrics import aggregate_by_recording, regression_metrics
from uwb_fat.training import train_fold


def _preprocess(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    build_observation_archive(args.manifest, args.output, config, args.dataset, args.layout)
    print(json.dumps({"observation_archive": str(Path(args.output).resolve())}, indent=2))


def _validate_manifest(args: argparse.Namespace) -> None:
    rows = read_manifest(args.manifest)
    participants = sorted({row.participant_id for row in rows})
    sites = sorted({row.site for row in rows})
    missing = [row.recording_path for row in rows if not Path(row.recording_path).is_file()]
    result = {
        "recordings": len(rows),
        "participants": len(participants),
        "sites": sites,
        "missing_recordings": missing,
    }
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit(2)


def _train_fold(args: argparse.Namespace) -> None:
    report = train_fold(
        args.dataset,
        args.held_out,
        args.output,
        load_config(args.config),
        args.device,
    )
    print(json.dumps(report, indent=2))


def _metrics(args: argparse.Namespace) -> None:
    references: list[float] = []
    predictions: list[float] = []
    recording_ids: list[str] = []
    with Path(args.predictions).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            references.append(float(row["reference_mm"]))
            predictions.append(float(row["prediction_mm"]))
            recording_ids.append(row["recording_id"])
    reference = np.asarray(references)
    prediction = np.asarray(predictions)
    rec_ref, rec_pred, _ = aggregate_by_recording(reference, prediction, np.asarray(recording_ids))
    print(
        json.dumps(
            {
                "window_level": regression_metrics(reference, prediction),
                "recording_level": regression_metrics(rec_ref, rec_pred),
            },
            indent=2,
        )
    )


def _bodyfat(args: argparse.Namespace) -> None:
    if args.sex == "male":
        sites = {"chest": args.chest, "abdomen": args.abdomen, "thigh": args.thigh}
    else:
        sites = {
            "triceps": args.triceps,
            "suprailiac": args.suprailiac,
            "thigh": args.thigh,
        }
    if any(value is None for value in sites.values()):
        raise SystemExit(f"Missing required site value(s) for {args.sex}: {sorted(sites)}")
    density = jackson_pollock_density(args.sex, args.age, sites)
    print(
        json.dumps(
            {
                "body_density": density,
                "body_fat_percent": body_fat_percent(args.sex, args.age, sites),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-manifest")
    validate.add_argument("--manifest", required=True)
    validate.set_defaults(func=_validate_manifest)

    preprocess = commands.add_parser("preprocess")
    preprocess.add_argument("--manifest", required=True)
    preprocess.add_argument("--output", required=True)
    preprocess.add_argument("--config", required=True)
    preprocess.add_argument("--dataset")
    preprocess.add_argument(
        "--layout",
        default="auto",
        choices=["auto", "frames_channels_bins", "channels_bins_frames", "bins_channels_frames"],
    )
    preprocess.set_defaults(func=_preprocess)

    train = commands.add_parser("train-fold")
    train.add_argument("--dataset", required=True)
    train.add_argument("--held-out", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--device", default="cpu")
    train.set_defaults(func=_train_fold)

    metrics = commands.add_parser("metrics")
    metrics.add_argument("--predictions", required=True)
    metrics.set_defaults(func=_metrics)

    bodyfat = commands.add_parser("bodyfat")
    bodyfat.add_argument("--sex", required=True, choices=["male", "female"])
    bodyfat.add_argument("--age", required=True, type=float)
    bodyfat.add_argument("--chest", type=float)
    bodyfat.add_argument("--triceps", type=float)
    bodyfat.add_argument("--abdomen", type=float)
    bodyfat.add_argument("--suprailiac", type=float)
    bodyfat.add_argument("--thigh", required=True, type=float)
    bodyfat.set_defaults(func=_bodyfat)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
