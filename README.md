# HCI Sensing

Open-source research software for privacy-aware sensing, behavioral inference, and wearable interaction. This repository brings four projects into one reproducible portfolio.

Project website: [HCI Sensing](https://sites.google.com/view/hci-sensing)

| Project | Focus | Language |
| --- | --- | --- |
| [UWB-Fat](uwb-fat/) | UWB signal processing and physics-informed estimation of subcutaneous-fat thickness | Python |
| [UWB-PostureGuard](uwb-postureguard/) | Temporal posture recognition and out-of-distribution detection from UWB features | Python |
| [Earable Signal Toolbox](earable-signal-toolbox/) | Multichannel inertial preprocessing, enrollment, and verification | MATLAB |
| [Tooth Acoustic Toolbox](tooth-acoustic-toolbox/) | Contact-acoustic event analysis, template matching, and score fusion | MATLAB |

## Repository scope

The repository contains source code and non-sensitive configuration files. Human-subject recordings, labels, videos, trained models, manuscripts, and presentation assets are excluded. Each project README documents its requirements and entry point.

## Use

Clone the portfolio and enter the project you want to run:

```bash
git clone https://github.com/nkjcqvcpi/hci-sensing-portfolio.git
cd hci-sensing-portfolio
```

The two Python projects use isolated `pyproject.toml` environments. The MATLAB projects run with core MATLAB functions and include lightweight synthetic examples and tests.

## Responsible research use

These projects are research prototypes, not medical devices or safety systems. Users are responsible for ethics approval, informed consent, privacy protection, and validation in their intended setting.

## License

Released under the [MIT License](LICENSE).
