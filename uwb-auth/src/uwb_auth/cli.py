"""Command-line interface for UWBAuth."""

from __future__ import annotations

import argparse
import json

from .config import ExperimentConfig
from .experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uwb-auth")
    subparsers = parser.add_subparsers(dest="command", required=True)
    experiment = subparsers.add_parser(
        "experiment", help="Run condition-disjoint verification cross-validation"
    )
    experiment.add_argument("--data-root", required=True, help="Path to UWB_raw_data/ranging")
    experiment.add_argument("--labels", required=True, help="Path to labels.xlsx")
    experiment.add_argument("--config", help="Experiment TOML configuration")
    experiment.add_argument("--output", required=True, help="Aggregate JSON result path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig.from_toml(args.config) if args.config else ExperimentConfig()
    report = run_experiment(
        data_root=args.data_root,
        labels_path=args.labels,
        output_path=args.output,
        config=config,
    )
    dual = report["variants"]["dual_link"]["aggregate_test"]["overall"]
    print(
        json.dumps(
            {
                "output": args.output,
                "macro_balanced_accuracy": dual["macro_balanced_accuracy"],
                "macro_false_acceptance_rate": dual["macro_false_acceptance_rate"],
                "macro_false_rejection_rate": dual["macro_false_rejection_rate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
