"""InsightFace recognizer - получение 512D эмбеддингов лиц"""

from typing import List, Optional
import numpy as np
from pathlib import Path
import cv2

from core.interfaces.detectors import Face


class InsightFaceRecognizer:
    """
    InsightFace ArcFace распознавание лиц

    Преимущества:
    - SOTA качество эмбеддингов (512D)
    - Быстрая работа на CPU/GPU
    - Робастность к углам, освещению, частичным окклюзиям
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",  # buffalo_l, buffalo_s, antelopev2
        device: str = "cpu",
    ):
        """
        Args:
            model_name: Модель InsightFace (buffalo_l - лучшее качество)
            device: Устройство (cpu или cuda)
        """
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface not installed. Install with: pip install insightface onnxruntime"
            )

        self.device = device
        ctx_id = 0 if device == "cuda" else -1  # -1 для CPU

        # Инициализируем FaceAnalysis
        self.app = FaceAnalysis(
            name=model_name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
        )

        # Prepare с определенным размером
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        print(f"[InsightFace] initialized with model: {model_name}, device: {device}")

    def get_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Получить эмбеддинг для одного лица

        Args:
            face_img: Crop лица (BGR format)

        Returns:
            512D эмбеддинг или None
        """
        if face_img is None or face_img.size == 0:
            return None

        try:
            # InsightFace analyze
            faces = self.app.get(face_img)

            if len(faces) == 0:
                return None

            # Берем первое (самое большое) лицо
            face = faces[0]

            # Эмбеддинг уже нормализован
            embedding = face.embedding.astype(np.float32)

            return embedding

        except Exception as e:
            print(f"[InsightFace] embedding error: {e}")
            return None

    def add_embeddings_to_faces(self, frame: np.ndarray, faces: List[Face]) -> List[Face]:
        """
        Добавить эмбеддинги к уже детектированным лицам

        Args:
            frame: Исходный кадр (BGR)
            faces: Список лиц с bbox

        Returns:
            Список лиц с эмбеддингами
        """
        enriched_faces = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox

            # Расширяем bbox для лучшего качества эмбеддинга
            h, w = frame.shape[:2]
            padding = int(0.1 * (x2 - x1))
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)

            # Crop лица
            face_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

            # Получаем эмбеддинг
            embedding = self.get_embedding(face_crop)

            if embedding is not None:
                # Создаем новый Face с эмбеддингом
                enriched_face = Face(
                    bbox=face.bbox,
                    embedding=embedding,
                    confidence=face.confidence,
                    landmarks=face.landmarks
                )
                enriched_faces.append(enriched_face)

        return enriched_faces
