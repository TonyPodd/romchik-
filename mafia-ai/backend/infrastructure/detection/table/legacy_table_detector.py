"""Legacy table detector adapter"""

from typing import Optional, Tuple
import numpy as np

from core.interfaces.detectors import ITableDetector

# Импортируем функцию auto_detect_table из stream_worker
from video.stream_worker import GestureStream


class LegacyTableDetector(ITableDetector):
    """
    Адаптер для текущего table detector
    Использует auto_detect_table из GestureStream
    """

    def __init__(self):
        print(f"[LegacyTableDetector] initialized")

    async def detect_table(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Автоматически обнаружить контур стола

        Args:
            frame: BGR изображение

        Returns:
            Полигон стола (normalized coordinates 0-1) или None
        """
        # Используем статический метод из GestureStream
        # Создаем временный объект для вызова метода
        temp_stream = object.__new__(GestureStream)
        temp_stream._table_poly_norm = None

        # Вызываем метод автодетекции
        import asyncio
        h, w = frame.shape[:2]

        # Копируем логику из auto_detect_table
        roi = frame[int(h * 0.35):, :]
        y0 = int(h * 0.35)

        try:
            import cv2

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            g = cv2.bilateralFilter(gray, 9, 75, 75)
            edges = cv2.Canny(g, 50, 130)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
            cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if cnts:
                best = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.int32)
                best[:, 1] += y0
                hull = cv2.convexHull(best).reshape(-1, 2)
                peri = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.015 * peri, True).reshape(-1, 2)

                if approx.shape[0] >= 4:
                    # Нормализуем координаты
                    poly_norm = [(float(x) / w, float(y) / h) for (x, y) in approx]
                    return np.array(poly_norm, dtype=np.float32)
        except Exception as e:
            print(f"[table_detect] error: {e}")

        return None

    async def validate_table_polygon(self, polygon: np.ndarray, frame_shape: Tuple[int, int]) -> bool:
        """
        Проверить валидность полигона стола

        Args:
            polygon: Полигон стола (normalized)
            frame_shape: (height, width)

        Returns:
            True если полигон валиден
        """
        if polygon is None or len(polygon) < 3:
            return False

        # Проверяем что все координаты в диапазоне [0, 1]
        if np.any(polygon < 0) or np.any(polygon > 1):
            return False

        # Проверяем минимальную площадь
        h, w = frame_shape
        import cv2
        poly_px = (polygon * [w, h]).astype(np.int32)
        area = cv2.contourArea(poly_px)
        min_area = 0.02 * w * h

        return area >= min_area
