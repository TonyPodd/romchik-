"""Legacy face detector adapter - обертка над текущим кодом"""

from typing import List
import numpy as np

from core.interfaces.detectors import IFaceDetector, Face
from config.settings import settings

# Импортируем текущую реализацию
from video.stream_worker import _FaceBackendONNX, _FaceBackendLandmarks, _make_face_backend_initial


class LegacyFaceDetector(IFaceDetector):
    """
    Адаптер для текущего face detector
    Оборачивает _FaceBackendONNX/_FaceBackendLandmarks в новый интерфейс
    """

    def __init__(self):
        # Используем текущую логику выбора бэкенда
        self._backend = _make_face_backend_initial()
        print(f"[LegacyFaceDetector] initialized with backend: {type(self._backend).__name__}")

    async def detect(self, frame: np.ndarray) -> List[Face]:
        """
        Обнаружить лица на кадре

        Args:
            frame: BGR изображение

        Returns:
            Список обнаруженных лиц с эмбеддингами
        """
        # Вызываем текущую реализацию
        results = self._backend.analyze(frame)

        # Конвертируем в новый формат
        faces = []
        for r in results:
            face = Face(
                bbox=r["bbox"],
                embedding=r["embedding"],
                confidence=r.get("score", 1.0),
                landmarks=None  # Текущая реализация не возвращает landmarks
            )
            faces.append(face)

        return faces

    def get_embedding_dim(self) -> int:
        """Размерность эмбеддинга"""
        # ONNX возвращает 512D, Landmarks возвращает переменный размер
        # Определяем по типу бэкенда
        if isinstance(self._backend, _FaceBackendONNX):
            return 512
        else:
            # Landmarks: 468 точек * 2 координаты = 936
            return 936
