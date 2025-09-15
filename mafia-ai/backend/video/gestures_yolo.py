# backend/video/gestures_yolo.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import cv2

class YOLOHandDetector:
    """
    Обёртка над Ultralytics YOLOv8 для детекции рук.
    Выдаёт список боксов (x1,y1,x2,y2), score, cls.
    """
    def __init__(self, model_path: str | Path, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640, device: str = "cpu"):
        from ultralytics import YOLO  # требует: pip install ultralytics
        self.model = YOLO(str(model_path))
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        H, W = frame_bgr.shape[:2]
        res = self.model.predict(
            frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )[0]
        out: List[Dict[str, Any]] = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
            conf = float(b.conf[0]) if b.conf is not None else 0.0
            cls = int(b.cls[0]) if b.cls is not None else 0
            if x2 > x1 and y2 > y1:
                out.append({"bbox": (x1, y1, x2, y2), "score": conf, "cls": cls})
        return out
