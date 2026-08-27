# UWBAuth

UWBAuth is a research pipeline for claimed-identity verification from UWB ranging streams. It converts each recording into robust temporal summaries, trains an identity model on enrollment conditions, calibrates a claimant-specific threshold on separate conditions, and evaluates on nuisance conditions unseen during either step.

## Approach

The primary model uses both initiator and responder streams. It summarizes nine signals from `range_i.csv` and `range_r.csv` with the median, interquartile range, median absolute deviation, standard deviation, 10th and 90th percentiles, and mean absolute frame-to-frame change. An enrollment-fitted robust scaler and L2-regularized logistic regression produce identity scores. Collector-provided moving averages are excluded.

The threat model is zero-effort impostor verification. It does not cover replay, relay, or active spoofing attacks.

## Evaluation protocol

The archived data do not contain a usable second collection day. UWBAuth therefore uses three-fold, within-day, condition-disjoint cross-validation for the two anonymized subjects with nine matched nuisance conditions. Each fold assigns whole condition groups to enrollment, threshold validation, or test. No recording or frame crosses partitions within a fold. Thirteen postures are shared across every selected subject-condition cell.

Across the three folds, 234 distinct recordings appear in test exactly once, producing 468 claimed-identity decisions. The dual-link model is the primary analysis; the single-link models are ablations.

| Input | Balanced accuracy | False-accept rate | False-reject rate | Mean fold ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| Initiator only | 76.1% | 23.5% | 24.4% | 82.1% |
| Responder only | 82.7% | 15.8% | 18.8% | 84.7% |
| **Dual link, primary** | **78.6%** | **23.9%** | **18.8%** | **80.1%** |

The responder-only result is exploratory because the ablation was compared after test evaluation. These error rates do not support deployment as an access-control system. The checked-in [aggregate report](reports/condition_cv.json) includes fold-level thresholds, counts, metrics, software versions, and confidence intervals. It contains no names, demographics, raw signals, or recording-level predictions.

## Reproduce

Requirements: Python 3.11 to 3.13 and the private UWB-Posture data archive.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

uwb-auth experiment \
  --data-root /path/to/UWB_raw_data/ranging \
  --labels /path/to/UWB_raw_data/labels.xlsx \
  --config configs/condition_cv.toml \
  --output reports/condition_cv.json
```

The loader reads only the experiment, anonymized subject number, posture, and nuisance-condition columns from the configured label sheet. It does not read the workbook's subject-name or demographic sheet. Raw data and labels remain outside this repository.

Run the checks with:

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```

## Limitations

Only two subjects have enough matched nuisance conditions for this protocol. Identity can still be confounded by body geometry and the collection environment. The archive lacks second-day ranging files, so temporal persistence is unknown. A security evaluation requires a larger multi-day, multi-room cohort and explicit replay, relay, and spoofing attacks.

## License

MIT. See the repository-level `LICENSE` file.
