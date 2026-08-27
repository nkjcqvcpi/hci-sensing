# UWB-PostureGuard

UWB-PostureGuard performs temporal posture classification from UWB-derived features. It combines LightGBM classification with a leaf-embedding out-of-distribution detector.

## Role in HCI Sensing

Within the [HCI Sensing portfolio](../README.md), UWB-PostureGuard provides behavioral inference with an explicit distribution-shift signal. The pipeline reports a posture prediction and an out-of-distribution score derived from the classifier's leaf embedding, separating task accuracy from representation familiarity.

## Workflow

The pipeline selects ranging, signal-quality, and channel-impulse-response features; removes outliers; builds temporal windows; trains the posture classifier; fits the out-of-distribution detector; and exports posture predictions with familiarity scores.

## Requirements

- Python 3.11 to 3.13
- NumPy, pandas, scikit-learn, LightGBM, and joblib

## Install and smoke test

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/run.py make-synthetic --output synthetic-data
python scripts/run.py validate-input --input synthetic-data
```

Use `python scripts/run.py --help` to inspect task arguments. Default parameters are in `configs/default.toml`.

## Data and evaluation

Study recordings reside in authorized research storage. The `make-synthetic` task generates labeled signals for pipeline and schema checks.

The default configuration uses a frame-stratified split, and the recording-disjoint strategy supports independent-session evaluation. The out-of-distribution score measures familiarity in the classifier's leaf-embedding space. Participant, session, room, device, and placement splits support targeted generalization studies.

## License

MIT. See the repository-level `LICENSE` file.
