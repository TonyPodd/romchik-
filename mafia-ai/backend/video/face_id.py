from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
import os

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    raise RuntimeError("Install insightface and onnxruntime in your venv") from e

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-6
    nb = np.linalg.norm(b) + 1e-6
    return float(np.dot(a, b) / (na * nb))

class FaceIdentifier:
    def __init__(self, det_size: Tuple[int,int]=(640,640), providers: Optional[list]=None):
        self.app = FaceAnalysis(name="buffalo_l", providers=providers or ["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=det_size)

    def analyze(self, frame_bgr) -> List[Dict[str, Any]]:
        """Returns list of faces: {bbox, kps, det_score, embedding}"""
        faces = self.app.get(frame_bgr)
        out = []
        for f in faces:
            x1,y1,x2,y2 = map(int, f.bbox)
            emb = f.normed_embedding if hasattr(f, "normed_embedding") else f.embedding
            emb = np.array(emb, dtype=np.float32)
            out.append({
                "bbox": (x1,y1,x2,y2),
                "score": float(getattr(f, "det_score", 1.0)),
                "embedding": emb
            })
        return out

    def largest_face(self, faces: List[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
        if not faces: return None
        def area(bb): x1,y1,x2,y2 = bb; return (x2-x1)*(y2-y1)
        return max(faces, key=lambda f: area(f["bbox"]))
