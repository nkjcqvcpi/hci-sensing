# UWB-Fat

UWB-Fat is a Python research implementation for processing ultra-wideband channel impulse responses and estimating subcutaneous-fat thickness with a physics-informed neural model.

## Requirements

- Python 3.11 to 3.13
- NumPy, h5py, and PyTorch

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
uwb-fat --help
```

The command-line interface supports manifest validation, signal preprocessing, leave-one-subject-out training, regression metrics, and body-fat conversion. Paper-aligned parameters are in `configs/paper.toml`.

## Data

No participant recordings are included. Preprocessing requires an explicit manifest that points to locally authorized recordings.

## License

MIT. See the repository-level `LICENSE` file.

