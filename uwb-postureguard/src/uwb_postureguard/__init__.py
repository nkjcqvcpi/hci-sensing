"""UWB-PostureGuard research implementation."""

from .config import TrainingConfig
from .pipeline import PoseGuardBundle, train_from_path
from .taxonomy import POSTURES, Posture

__all__ = ["POSTURES", "PoseGuardBundle", "Posture", "TrainingConfig", "train_from_path"]
__version__ = "0.1.0"
