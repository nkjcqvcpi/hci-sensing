# HCI Sensing

## Introduction

Human-centered sensing connects physical measurements, computational inference, and responsible data practice. Effective systems document the measurement process, support reproducible signal processing, evaluate realistic variation, and characterize the identity information carried by sensed signals.

HCI Sensing develops this system view through five radio, inertial, and acoustic projects. UWB-Fat estimates subcutaneous-fat thickness from UWB channel impulse responses with a physics-informed model. UWB-PostureGuard combines temporal posture recognition with out-of-distribution detection. UWBAuth evaluates claimed-identity verification under held-out nuisance conditions. The Earable Signal and Tooth Acoustic toolboxes provide reusable preprocessing, enrollment, template matching, and verification methods for inertial and body-coupled acoustic sensing.

Together, these projects form an experimental foundation for HEARTH, an interoperable ecosystem for privacy-preserving in-home sensing. HCI Sensing contributes acquisition-aware processing, realistic evaluation protocols, identity analysis, and reusable research artifacts. HEARTH will connect these methods through machine-readable measurement metadata, computable consent, and reproducible privacy and security audits.

Project website: [HCI Sensing](https://sites.google.com/view/hci-sensing)

| Project | Research role | Modality | Language |
| --- | --- | --- | --- |
| [UWB-Fat](uwb-fat/) | Physics-informed estimation from UWB channel impulse responses | UWB radio | Python |
| [UWB-PostureGuard](uwb-postureguard/) | Temporal inference and out-of-distribution detection | UWB radio | Python |
| [UWBAuth](uwb-auth/) | Claimed-identity verification under unseen nuisance conditions | UWB radio | Python |
| [Earable Signal Toolbox](earable-signal-toolbox/) | Reusable preprocessing, enrollment, and verification | Earable inertial | MATLAB |
| [Tooth Acoustic Toolbox](tooth-acoustic-toolbox/) | Event representation, template matching, and score fusion | Contact acoustic | MATLAB |

## Repository contents

The repository releases source code, non-sensitive configuration files, synthetic examples, tests, and aggregate results. Access-controlled research storage manages study recordings, labels, trained models, and media. Each project README documents its methods, requirements, and run instructions.

## Use

Clone the repo and enter the project you want to run:

```bash
git clone https://github.com/nkjcqvcpi/hci-sensing.git
cd hci-sensing
```

The three Python projects use `uv` for dependency resolution, virtual-environment management, and command execution. Enter a Python project and run `uv sync --group dev` to create its environment from `pyproject.toml` and `uv.lock`. The MATLAB projects run with core MATLAB functions and include lightweight synthetic examples and tests.

## Research governance

Each project follows ethics approval, informed consent, privacy protection, and environment-specific validation. Clinical and security studies add domain-specific evaluation for medical, safety, and access-control settings.

## License

Released under the [MIT License](LICENSE).
