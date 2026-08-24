"""UWB-Fat: UWB skinfold estimation."""

from .bodyfat import body_fat_percent, jackson_pollock_density, siri_percent
from .config import ProjectConfig, load_config
from .metrics import regression_metrics

__all__ = [
    "ProjectConfig",
    "body_fat_percent",
    "jackson_pollock_density",
    "load_config",
    "regression_metrics",
    "siri_percent",
]

__version__ = "0.1.0"
