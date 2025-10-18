"""Detector interfaces - абстракции для детекторов"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Face:
    """Обнаруженное лицо"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    embedding: np.ndarray             # Вектор эмбеддинга (512D)
    confidence: float                 # Уверенность детекции
    landmarks: Optional[np.ndarray] = None  # Ключевые точки лица


@dataclass
class Hand:
    """Обнаруженная рука"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]           # Центр руки
    confidence: float                 # Уверенность детекции
    landmarks: Optional[np.ndarray] = None  # 21 точка MediaPipe


@dataclass
class Gesture:
    """Распознанный жест"""
    gesture_type: str                 # "finger_on_table", "fist", "open_palm", etc
    confidence: float                 # Уверенность распознавания
    hand: Hand                        # Рука с которой связан жест
    finger_count: Optional[int] = None  # Количество пальцев (если применимо)


class IFaceDetector(ABC):
    """
    Интерфейс детектора лиц
    Можно заменить реализацию без изменения domain layer
    """

    @abstractmethod
    async def detect(self, frame: np.ndarray) -> List[Face]:
        """
        Обнаружить лица на кадре

        Args:
            frame: BGR изображение (numpy array)

        Returns:
            Список обнаруженных лиц с эмбеддингами
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Размерность эмбеддинга"""
        pass


class IGestureDetector(ABC):
    """
    Интерфейс детектора жестов
    """

    @abstractmethod
    async def detect_hands(self, frame: np.ndarray) -> List[Hand]:
        """
        Обнаружить руки на кадре

        Args:
            frame: BGR изображение

        Returns:
            Список обнаруженных рук с landmarks
        """
        pass

    @abstractmethod
    async def recognize_gestures(self, frame: np.ndarray, hands: List[Hand]) -> List[Gesture]:
        """
        Распознать жесты по обнаруженным рукам

        Args:
            frame: BGR изображение
            hands: Список обнаруженных рук

        Returns:
            Список распознанных жестов
        """
        pass

    @abstractmethod
    async def is_finger_on_table(
        self,
        hand: Hand,
        table_polygon: Optional[np.ndarray],
        table_y_threshold: float = 0.8
    ) -> bool:
        """
        Проверить находится ли палец на столе

        Args:
            hand: Обнаруженная рука
            table_polygon: Полигон стола (normalized coordinates)
            table_y_threshold: Y-порог для простой проверки

        Returns:
            True если палец на столе
        """
        pass


class ITableDetector(ABC):
    """
    Интерфейс детектора стола
    """

    @abstractmethod
    async def detect_table(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Автоматически обнаружить контур стола

        Args:
            frame: BGR изображение

        Returns:
            Полигон стола (normalized coordinates 0-1) или None
        """
        pass

    @abstractmethod
    async def validate_table_polygon(self, polygon: np.ndarray, frame_shape: Tuple[int, int]) -> bool:
        """
        Проверить валидность полигона стола

        Args:
            polygon: Полигон стола
            frame_shape: (height, width)

        Returns:
            True если полигон валиден
        """
        pass
