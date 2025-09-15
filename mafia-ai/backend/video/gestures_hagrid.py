# backend/video/gestures_hagrid.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import urllib.request, shutil

import numpy as np
import cv2

from .gestures_core import GestureLabel

# Кандидаты для скачивания onnx (HaGRID classifiers, 224x224)
MODEL_CANDIDATES = [
    # ResNet50 / MobileNetV3 варианты из hagrid-models (HF mirrors)
    "https://huggingface.co/spaces/hagrid-dataset/hagrid-models/resolve/main/classifiers/resnet50.onnx?download=true",
    "https://huggingface.co/spaces/hagrid-dataset/hagrid-models/resolve/main/classifiers/mobilenetv3_large.onnx?download=true",
]

# Список классов (упрощённо, покрывает базовые жесты)
# Реальные модели HaGRID обычно имеют ~18 классов; названия могут незначительно отличаться.
# При желании отладьте по print(self.class_names).
DEFAULT_CLASSES = [
    "fist","palm","ok","like","dislike","one","two","three","four","five",
    "stop","peace","rock","call","mute","no_gesture"
]

# Маппинг названий модели в ваши GestureLabel
CLASS2LABEL = {
    "fist":    GestureLabel.FIST,
    "palm":    GestureLabel.OPEN_PALM,
    "stop":    GestureLabel.OPEN_PALM,
    "five":    GestureLabel.FIVE,
    "four":    GestureLabel.FOUR,
    "three":   GestureLabel.THREE,
    "two":     GestureLabel.TWO,
    "peace":   GestureLabel.TWO,
    "one":     GestureLabel.ONE,
    "ok":      GestureLabel.OK_SIGN,
    "like":    GestureLabel.THUMB_UP,
    "dislike": GestureLabel.THUMB_DOWN,
    "rock":    GestureLabel.THREE,     # условно
    "call":    GestureLabel.POINT,     # или ввести отдельный жест позже
}

def _download(urls: List[str], dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    last = None
    for u in urls:
        try:
            print("[hagrid] downloading", u)
            req = urllib.request.Request(u, headers={"User-Agent":"MafiaAI/gestures"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            if tmp.stat().st_size < 1_000_000:
                raise RuntimeError("file too small")
            tmp.replace(dst)
            print(f"[hagrid] saved to {dst} ({dst.stat().st_size/1024/1024:.2f} MB)")
            return
        except Exception as e:
            last = e
            try: tmp.unlink(missing_ok=True)
            except: pass
    raise RuntimeError(f"Cannot download HaGRID model: {last}")

class HaGRIDClassifier:
    """
    ONNX-классификатор жестов из HaGRID. Вход: BGR кроп кисти -> 224x224.
    Предобработка: RGB, [0..1], Normalize(mean,std) как в TorchVision.
    Выход: дистрибуция по классам (softmax).
    """
    def __init__(self, model_path: Optional[Path]=None, classes: Optional[List[str]]=None, providers=None):
        import onnxruntime as ort
        self.model_path = model_path or (Path(__file__).resolve().parents[1]/"models"/"hagrid_cls.onnx")
        if not self.model_path.exists():
            _download(MODEL_CANDIDATES, self.model_path)
        if providers is None: providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(self.model_path), providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name
        # Классы: если модель содержит словарь — можно подгрузить отдельно; используем дефолт.
        self.class_names = classes or DEFAULT_CLASSES
        # ImageNet нормализация (типично для TorchVision) — см. torchvision docs
        # mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] :contentReference[oaicite:1]{index=1}
        self.mean = np.array([0.485,0.456,0.406], dtype=np.float32)[None,None,:]
        self.std  = np.array([0.229,0.224,0.225], dtype=np.float32)[None,None,:]

    def preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        img = img / 255.0
        img = (img - self.mean) / (self.std + 1e-6)     # HWC
        img = np.transpose(img, (2,0,1))[None,...].astype(np.float32)  # 1x3x224x224
        return img

    def predict_proba(self, crop_bgr: np.ndarray):
        if not self.available: return None
        x = self.preprocess(crop_bgr)
        logits = self.sess.run([self.out], {self.inp: x})[0][0]  # (C,)
        p = logits - np.max(logits)
        p = np.exp(p); p = p / (p.sum() + 1e-9)
        return p  # np.ndarray, shape (C,)

    def predict(self, crop_bgr: np.ndarray):
        proba = self.predict_proba(crop_bgr)
        if proba is None: return None
        j = int(np.argmax(proba)); conf = float(proba[j])
        name = self.class_names[j] if j < len(self.class_names) else f"class_{j}"
        return name, conf

    def to_label(self, class_name: str) -> GestureLabel:
        return CLASS2LABEL.get(class_name, GestureLabel.UNKNOWN)
