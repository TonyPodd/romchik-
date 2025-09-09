# backend/video/face_id_onnx.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import os, urllib.request, hashlib
import numpy as np
import cv2

import mediapipe as mp
import onnxruntime as ort
from pathlib import Path

MODEL_URL = "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-11.onnx"
MODEL_SHA256 = "b3b4f6b85cfe3f29f01e8e350919edc0c587a837ec0e2b17ad27003f091f13f4"  # контрольная сумма
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "arcface.onnx"

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def _ensure_model():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    # простая проверка целостности (не фатальная, если не совпало — просто предупреждение)
    try:
        digest = _sha256(MODEL_PATH)
        if digest.lower() != MODEL_SHA256.lower():
            print("[arcface] sha256 mismatch (continuing anyway)")
    except Exception:
        pass

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    na = np.linalg.norm(a) + 1e-6
    nb = np.linalg.norm(b) + 1e-6
    return float(np.dot(a, b) / (na * nb))

class FaceONNX:
    """Детекция (MediaPipe) + эмбеддинги (ArcFace ONNX)."""
    def __init__(self, sim_threshold: float = 0.38):
        _ensure_model()
        self.sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

        self.det = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.6
        )
        self.sim_threshold = sim_threshold

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        # ArcFace ожидает 112x112 RGB, нормализацию [-1,1], NCHW
        img = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img / 127.5) - 1.0  # [-1,1]
        img = np.transpose(img, (2, 0, 1))  # C,H,W
        img = np.expand_dims(img, 0)  # N,C,H,W
        return img

    def _embed(self, face_bgr: np.ndarray) -> np.ndarray:
        inp = self._preprocess(face_bgr)
        out = self.sess.run([self.output_name], {self.input_name: inp})[0]
        emb = out[0].astype(np.float32)
        # L2-нормировка
        n = np.linalg.norm(emb) + 1e-6
        return emb / n

    def analyze(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Список лиц: {bbox:(x1,y1,x2,y2), score, embedding:512}"""
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.det.process(rgb)
        out: List[Dict[str,Any]] = []
        if not res.detections:
            return out

        for det in res.detections:
            # bbox в относительных координатах
            r = det.location_data.relative_bounding_box
            x1 = int(max(0, r.xmin) * W)
            y1 = int(max(0, r.ymin) * H)
            x2 = int(min(1.0, r.xmin + r.width) * W)
            y2 = int(min(1.0, r.ymin + r.height) * H)
            # немного расширим область, чтобы захватить контекст
            pad_x = int(0.15 * (x2 - x1))
            pad_y = int(0.20 * (y2 - y1))
            xx1 = max(0, x1 - pad_x); yy1 = max(0, y1 - pad_y)
            xx2 = min(W, x2 + pad_x); yy2 = min(H, y2 + pad_y)
            crop = frame_bgr[yy1:yy2, xx1:xx2]
            if crop.size == 0:
                continue
            emb = self._embed(crop)
            out.append({
                "bbox": (xx1, yy1, xx2, yy2),
                "score": float(det.score[0] if det.score else 0.0),
                "embedding": emb
            })
        return out

    @staticmethod
    def largest_face(faces: List[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
        if not faces: return None
        def area(bb): x1,y1,x2,y2 = bb; return (x2-x1)*(y2-y1)
        return max(faces, key=lambda f: area(f["bbox"]))
