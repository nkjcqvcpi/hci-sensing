# UWB-Fat

UWB-Fat processes ultra-wideband channel impulse responses and estimates subcutaneous-fat thickness with a physics-informed neural model.

## Role in HCI Sensing

Within the [HCI Sensing portfolio](../README.md), UWB-Fat provides the physical-estimation layer. It connects a declared UWB acquisition configuration to signal preprocessing, tissue-thickness estimation, recording-level evaluation, and formula-based body-fat conversion.

## Workflow

The pipeline validates a recording manifest, converts raw channel data into observation archives, trains one leave-one-subject-out fold, evaluates window-level and recording-level predictions, and converts anatomical-site estimates to body-fat percentage.

## Requirements

- Python 3.11 to 3.13
- NumPy, h5py, and PyTorch

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/run.py --help
```

Use `python scripts/run.py --help` to inspect task arguments. Default parameters are in `configs/default.toml`.

## Data and evaluation

Participant recordings enter the pipeline through an explicit manifest that points to authorized local files. The repository provides default model parameters in `configs/default.toml` and X7F202 acquisition settings in `configs/x7f202.json`.

The included configuration specifies one acquisition setup. Training produces study-specific weights from the authorized recordings, and evaluation reports both window-level and recording-level regression metrics. The `bodyfat` task applies the implemented equations to site measurements.

## License

MIT. See the repository-level `LICENSE` file.
