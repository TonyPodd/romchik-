"""YOLOv8-face detector - быстрая и точная детекция лиц"""

from typing import List, Optional
import numpy as np
from pathlib import Path
import urllib.request

from core.interfaces.detectors import IFaceDetector, Face
from config.settings import settings


class YOLOv8FaceDetector(IFaceDetector):
    """
    YOLOv8-face детектор

    Преимущества:
    - Быстрее чем MediaPipe (30+ FPS)
    - Точнее на сложных углах и освещении
    - Нет ограничения на количество лиц
    - Работает на CPU и GPU
    """

    def __init__(
        self,
        model_size: str = "n",  # n (nano), s (small), m (medium), l (large)
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",  # cpu или cuda
    ):
        """
        Args:
            model_size: Размер модели (n, s, m, l). n - самая быстрая, l - самая точная
            confidence_threshold: Минимальная уверенность детекции
            iou_threshold: IoU порог для NMS
            device: Устройство (cpu или cuda)
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Путь к модели
        models_dir = Path(settings.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        # YOLOv8-face модели можно взять отсюда:
        # https://github.com/derronqi/yolov8-face
        model_name = f"yolov8{model_size}-face.pt"
        self.model_path = models_dir / model_name

        # Если модели нет, используем обычный YOLOv8n (он тоже может детектировать лица)
        if not self.model_path.exists():
            print(f"[YOLOv8Face] {model_name} not found, using yolov8n.pt")
            # Можно использовать предобученный YOLOv8 на COCO (класс 0 = person)
            # Или скачать специализированную YOLOv8-face модель
            self.model_path = models_dir / "yolov8n.pt"
            if not self.model_path.exists():
                print("[YOLOv8Face] Downloading yolov8n.pt from Ultralytics...")
                # YOLO сам скачает при первом запуске
                self.model = YOLO("yolov8n.pt")
            else:
                self.model = YOLO(str(self.model_path))
        else:
            self.model = YOLO(str(self.model_path))

        # Переводим на нужное устройство
        if device == "cuda":
            self.model.to("cuda")

        print(f"[YOLOv8Face] initialized with model: {model_name}, device: {device}")

    async def detect(self, frame: np.ndarray) -> List[Face]:
        """
        Обнаружить лица на кадре

        Args:
            frame: BGR изображение (OpenCV format)

        Returns:
            Список обнаруженных лиц
        """
        # YOLOv8 inference
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[0],  # 0 = person в COCO dataset (или face в специализированной модели)
            verbose=False,
            device=self.device,
        )

        faces = []

        if len(results) == 0 or results[0].boxes is None:
            return faces

        boxes = results[0].boxes

        for box in boxes:
            # Получаем bbox
            xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            conf = float(box.conf[0].cpu().numpy())

            x1, y1, x2, y2 = map(int, xyxy)

            # Создаем Face объект (без эмбеддинга пока)
            # Эмбеддинг будет добавлен через InsightFace
            face = Face(
                bbox=(x1, y1, x2, y2),
                embedding=np.zeros(512, dtype=np.float32),  # Placeholder
                confidence=conf,
                landmarks=None
            )
            faces.append(face)

        return faces

    def get_embedding_dim(self) -> int:
        """YOLOv8 не возвращает эмбеддинги, только bbox"""
        return 0  # Эмбеддинги будут от InsightFace
