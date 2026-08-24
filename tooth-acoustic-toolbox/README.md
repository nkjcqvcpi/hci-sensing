# Tooth Acoustic Toolbox

MATLAB utilities for short contact-acoustic events. The toolbox provides waveform conditioning, active-region extraction, spectral descriptors, gesture-conditioned templates, score fusion, and threshold analysis.

## Requirements

- MATLAB R2021b or newer
- Core MATLAB functions only

## Run

```matlab
startup
out = run_analysis;
run_tests
```

The default analysis uses procedurally generated waveforms. Custom audio supplied to `toothaudio.condition` should be a real-valued vector sampled at the rate set in the configuration.

## Main package

The `+toothaudio` package contains signal conditioning, event description, template fitting, matching, score fusion, and assessment functions.

## License

MIT. See the repository-level `LICENSE` file.

