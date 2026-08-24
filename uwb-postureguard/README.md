# UWB-PostureGuard

UWB-PostureGuard is a Python pipeline for temporal posture classification from UWB-derived features. It combines LightGBM classification with a leaf-embedding out-of-distribution detector.

## Requirements

- Python 3.11 to 3.13
- NumPy, pandas, scikit-learn, LightGBM, and joblib

## Install and smoke test

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
uwb-postureguard make-synthetic --output synthetic-data
uwb-postureguard validate-input --input synthetic-data
```

The interface also supports model training and prediction. Default research parameters are in `configs/paper.toml`.

## Data

Human-subject recordings, videos, and labels are excluded. The synthetic-data command creates non-human inputs for pipeline checks only.

## License

MIT. See the repository-level `LICENSE` file.

