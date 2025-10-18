"""Face detection implementations"""

from .legacy_face_detector import LegacyFaceDetector
from .yolo_face_detector import YOLOv8FaceDetector
from .insightface_recognizer import InsightFaceRecognizer
from .hybrid_face_detector import HybridFaceDetector

__all__ = [
    "LegacyFaceDetector",
    "YOLOv8FaceDetector",
    "InsightFaceRecognizer",
    "HybridFaceDetector",
]
