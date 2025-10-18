"""Interfaces (ports) for domain layer"""

from .repositories import IPlayerRepository, IGameRepository
from .detectors import IFaceDetector, IGestureDetector, ITableDetector
from .audio_processor import IAudioProcessor

__all__ = [
    "IPlayerRepository",
    "IGameRepository",
    "IFaceDetector",
    "IGestureDetector",
    "ITableDetector",
    "IAudioProcessor",
]
