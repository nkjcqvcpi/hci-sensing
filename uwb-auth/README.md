# UWBAuth

UWBAuth is a research pipeline for claimed-identity verification from UWB ranging streams. It converts each recording into robust temporal summaries, trains an identity model on enrollment conditions, calibrates a claimant-specific threshold on separate conditions, and evaluates on nuisance conditions unseen during either step.

## Role in HCI Sensing

Within [HCI Sensing](../README.md), UWBAuth provides identity-sensitive evaluation. It reuses UWB streams from behavioral sensing to measure claimed-identity verification under held-out nuisance conditions and quantify the identity information carried by those streams.

## Approach

The dual-link model uses both initiator and responder streams. It summarizes nine original per-frame signals from `range_i.csv` and `range_r.csv` with the median, interquartile range, median absolute deviation, standard deviation, 10th and 90th percentiles, and mean absolute frame-to-frame change. An enrollment-fitted robust scaler and L2-regularized logistic regression produce identity scores.

The evaluated threat model is zero-effort impostor verification.

## Evaluation protocol

The available archive supports three-fold, within-day, condition-disjoint cross-validation for two anonymized subjects with nine matched nuisance conditions. Each fold assigns complete condition groups to enrollment, threshold validation, or test. The partitions are disjoint at both recording and frame levels. Thirteen postures are shared across every selected subject-condition cell.

Across the three folds, 234 distinct recordings appear in test exactly once, producing 468 claimed-identity decisions. The table reports the dual-link primary analysis and two single-link ablations.

| Input | Balanced accuracy | False-accept rate | False-reject rate | Mean fold ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| Initiator only | 76.1% | 23.5% | 24.4% | 82.1% |
| Responder only | 82.7% | 15.8% | 18.8% | 84.7% |
| **Dual link, primary** | **78.6%** | **23.9%** | **18.8%** | **80.1%** |

The checked-in [aggregate report](reports/condition_cv.json) records fold-level thresholds, counts, metrics, software versions, confidence intervals, and aggregate results.

## Reproduce

Requirements: uv, Python 3.11 to 3.13, and the private UWB-Posture data archive.

```bash
uv sync --group dev

uv run python scripts/run.py experiment \
  --data-root /path/to/UWB_raw_data/ranging \
  --labels /path/to/UWB_raw_data/labels.xlsx \
  --config configs/condition_cv.toml \
  --output reports/condition_cv.json
```

The loader selects the experiment, anonymized subject number, posture, and nuisance-condition columns from the configured label sheet. The authorized UWB-Posture archive manages source recordings and labels.

Run the checks with:

```bash
uv run pytest
uv run ruff check .
```

## License

MIT. See the repository-level `LICENSE` file.
