# Tooth Acoustic Toolbox

The Tooth Acoustic Toolbox processes short contact-acoustic events in MATLAB. It provides waveform conditioning, active-region extraction, spectral descriptors, gesture-conditioned templates, score fusion, and threshold analysis.

## Role in HCI Sensing

Within the [HCI Sensing portfolio](../README.md), this toolbox provides body-coupled acoustic authentication. It follows the shared measurement-to-representation-to-verification pipeline through acoustic event localization, spectral description, gesture conditioning, and multi-event score fusion.

## Requirements

- MATLAB R2021b or newer
- Core MATLAB functions only

## Run

```matlab
startup
out = run_analysis;
run_tests
```

Custom audio supplied to `toothaudio.condition` should be a real-valued vector sampled at the rate set in the configuration.

## Data and evaluation

The default analysis generates acoustic events and reports an equal-error estimate for a complete computational check. Human-subject studies extend this evaluation to authentication accuracy, longitudinal stability, liveness, replay resistance, and imitation resistance under approved consent and privacy controls.

## Main package

The `+toothaudio` package contains signal conditioning, event description, template fitting, matching, score fusion, and assessment functions.

## License

MIT. See the repository-level `LICENSE` file.
