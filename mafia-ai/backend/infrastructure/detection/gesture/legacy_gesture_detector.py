"""Legacy gesture detector adapter"""

from typing import List, Optional, Tuple
import numpy as np

from core.interfaces.detectors import IGestureDetector, Hand, Gesture
from config.settings import settings

# Импортируем текущую реализацию
from video.gestures import GestureDetector


class LegacyGestureDetector(IGestureDetector):
    """
    Адаптер для текущего gesture detector
    Оборачивает GestureDetector в новый интерфейс
    """

    def __init__(self, table_y_ratio: float = 0.8):
        self._detector = GestureDetector(table_y_ratio=table_y_ratio)
        self._table_y_ratio = table_y_ratio
        print(f"[LegacyGestureDetector] initialized")

    async def detect_hands(self, frame: np.ndarray) -> List[Hand]:
        """
        Обнаружить руки на кадре

        Args:
            frame: BGR изображение

        Returns:
            Список обнаруженных рук
        """
        # Используем текущую реализацию
        result = self._detector.process_frame(frame)

        hands = []
        for hand_data in result.hands:
            hand = Hand(
                bbox=hand_data.bbox,
                center=hand_data.center,
                confidence=1.0,  # Текущая реализация не возвращает confidence
                landmarks=None  # TODO: добавить landmarks если нужно
            )
            hands.append(hand)

        return hands

    async def recognize_gestures(self, frame: np.ndarray, hands: List[Hand]) -> List[Gesture]:
        """
        Распознать жесты по обнаруженным рукам

        Args:
            frame: BGR изображение
            hands: Список обнаруженных рук

        Returns:
            Список распознанных жестов
        """
        # Текущая реализация уже включает детекцию + классификацию
        result = self._detector.process_frame(frame)

        gestures = []
        for i, hand_data in enumerate(result.hands):
            # Определяем тип жеста
            finger_count = hand_data.count if hasattr(hand_data, 'count') else 0

            gesture_type = "unknown"
            if finger_count == 0:
                gesture_type = "fist"
            elif finger_count == 1:
                gesture_type = "one_finger"
            elif finger_count == 2:
                gesture_type = "two_fingers"
            elif finger_count == 5:
                gesture_type = "open_palm"

            # Создаем Hand объект для этого жеста
            if i < len(hands):
                hand = hands[i]
            else:
                hand = Hand(
                    bbox=hand_data.bbox,
                    center=hand_data.center,
                    confidence=1.0,
                    landmarks=None
                )

            gesture = Gesture(
                gesture_type=gesture_type,
                confidence=1.0,
                hand=hand,
                finger_count=finger_count
            )
            gestures.append(gesture)

        return gestures

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
        cx, cy = hand.center
        h, w = 720, 1280  # TODO: получать из frame

        # Нормализуем координаты
        cy_norm = cy / h

        # Простая проверка: рука ниже порога
        if cy_norm >= table_y_threshold:
            # Если есть полигон, проверяем вхождение
            if table_polygon is not None:
                import cv2
                # Денормализуем полигон
                poly_px = (table_polygon * [w, h]).astype(np.int32)
                result = cv2.pointPolygonTest(poly_px, (cx, cy), False)
                return result >= 0
            return True

        return False
