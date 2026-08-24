# Earable Signal Toolbox

MATLAB utilities for short multichannel inertial recordings. The toolbox covers timestamp repair, drift removal, robust normalization, windowed features, template enrollment, and verification scoring.

## Requirements

- MATLAB R2021b or newer
- Core MATLAB functions only

## Run

```matlab
startup
result = run_pipeline;
run_tests
```

The default pipeline generates non-human signals in memory. Recorded sessions can be passed to `earable.preprocess` with `time` and `samples` fields.

## Main package

The `+earable` package contains preprocessing, feature extraction, enrollment, scoring, and FAR/FRR evaluation functions.

## License

MIT. See the repository-level `LICENSE` file.

