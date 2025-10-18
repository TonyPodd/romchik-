"""Hybrid face detector - YOLOv8 для детекции + InsightFace для эмбеддингов"""

from typing import List
import numpy as np

from core.interfaces.detectors import IFaceDetector, Face
from .yolo_face_detector import YOLOv8FaceDetector
from .insightface_recognizer import InsightFaceRecognizer


class HybridFaceDetector(IFaceDetector):
    """
    Гибридный детектор: YOLOv8 + InsightFace

    Pipeline:
    1. YOLOv8 детектирует bbox лиц (быстро, точно)
    2. InsightFace извлекает 512D эмбеддинги для каждого bbox (качество SOTA)

    Преимущества:
    - Лучшее из двух миров
    - YOLOv8: быстрая детекция, много лиц, любые углы
    - InsightFace: максимально качественные эмбеддинги для распознавания
    - Оптимизировано для real-time (30+ FPS на CPU)
    """

    def __init__(
        self,
        yolo_model_size: str = "n",  # n, s, m, l
        insightface_model: str = "buffalo_s",  # buffalo_s более стабилен чем buffalo_l
        device: str = "cpu",
        confidence_threshold: float = 0.5,
    ):
        """
        Args:
            yolo_model_size: Размер YOLOv8 модели (n=fastest, l=most accurate)
            insightface_model: Модель InsightFace (buffalo_s для стабильности)
            device: cpu или cuda
            confidence_threshold: Минимальная уверенность детекции
        """
        self.device = device
        self.insightface_available = False

        # YOLOv8 для детекции bbox (всегда работает)
        print(f"[HybridFaceDetector] Initializing YOLOv8...")
        self.yolo = YOLOv8FaceDetector(
            model_size=yolo_model_size,
            confidence_threshold=confidence_threshold,
            device=device
        )

        # InsightFace для эмбеддингов (опционально)
        print(f"[HybridFaceDetector] Initializing InsightFace...")
        try:
            self.insightface = InsightFaceRecognizer(
                model_name=insightface_model,
                device=device
            )
            self.insightface_available = self.insightface._initialized

            if self.insightface_available:
                print(f"[HybridFaceDetector] ✅ initialized (YOLO{yolo_model_size} + {insightface_model})")
            else:
                print(f"[HybridFaceDetector] ⚠️  initialized (YOLO{yolo_model_size} only, InsightFace failed)")

        except Exception as e:
            print(f"[HybridFaceDetector] ⚠️  InsightFace initialization failed: {e}")
            print(f"[HybridFaceDetector] ✅ initialized (YOLO{yolo_model_size} only)")
            self.insightface = None
            self.insightface_available = False

    async def detect(self, frame: np.ndarray) -> List[Face]:
        """
        Обнаружить лица и получить эмбеддинги

        Args:
            frame: BGR изображение

        Returns:
            Список лиц с bbox и 512D эмбеддингами (если InsightFace доступен)
        """
        # Шаг 1: Детектируем bbox через YOLOv8
        faces_with_bbox = await self.yolo.detect(frame)

        if len(faces_with_bbox) == 0:
            return []

        # Шаг 2: Добавляем эмбеддинги через InsightFace (если доступен)
        if self.insightface_available and self.insightface is not None:
            try:
                faces_with_embeddings = self.insightface.add_embeddings_to_faces(
                    frame, faces_with_bbox
                )
                return faces_with_embeddings
            except Exception as e:
                print(f"[HybridFaceDetector] ⚠️  InsightFace error, falling back to YOLO only: {e}")
                return faces_with_bbox
        else:
            # Возвращаем только bbox от YOLO
            return faces_with_bbox

    def get_embedding_dim(self) -> int:
        """Размерность эмбеддинга InsightFace"""
        return 512
