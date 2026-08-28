# Earable Signal Toolbox

The Earable Signal Toolbox processes short multichannel inertial recordings in MATLAB. It covers timestamp repair, drift removal, robust normalization, windowed features, template enrollment, and verification scoring.

## Role in HEARTH

Within [HEARTH](../README.md), this toolbox extends identity verification from ambient radio to body-worn inertial signals. Its modality-specific preprocessing feeds a common sequence of enrollment, scoring, and false-accept versus false-reject evaluation.

## Requirements

- MATLAB R2021b or newer
- Core MATLAB functions only

## Run

```matlab
startup
result = run_pipeline;
run_tests
```

Recorded sessions can be passed to `earable.preprocess` with `time` and `samples` fields.

## Data and evaluation

The default pipeline generates inertial signals in memory and produces an equal-error estimate for a complete computational check. Human-subject studies extend this evaluation to biometric accuracy, longitudinal persistence, imitation resistance, and sensor spoofing.

## Main package

The `+earable` package contains preprocessing, feature extraction, enrollment, scoring, and FAR/FRR evaluation functions.

## License

MIT. See the repository-level `LICENSE` file.
