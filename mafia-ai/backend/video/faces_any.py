# backend/video/faces_any.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np, cv2, os

@dataclass
class DetFace:
    bbox: Tuple[int,int,int,int]   # x1,y1,x2,y2
    embedding: np.ndarray          # L2-normalized
    score: float

def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-8
    return x / n

# ---- ArcFace ONNX (optional) ----
_ORT = None; _IN = None; _OUT = None
def _find_arcface_path() -> Optional[str]:
    mdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    for name in ["arcface.onnx","glintr100.onnx","face-recognition-resnet100-arcface-onnx.onnx"]:
        p = os.path.join(mdir, name)
        if os.path.exists(p): return p
    return None

def _init_arcface():
    global _ORT, _IN, _OUT
    if _ORT is not None: return
    try:
        import onnxruntime as ort
        mpth = _find_arcface_path()
        if mpth:
            _ORT = ort.InferenceSession(mpth, providers=["CPUExecutionProvider"])
            _IN = _ORT.get_inputs()[0].name; _OUT = _ORT.get_outputs()[0].name
            print(f"[faces_any] ArcFace ONNX loaded: {mpth}")
        else:
            print("[faces_any] ArcFace model not found — using handcrafted embeddings")
    except Exception as e:
        print(f"[faces_any] ONNX init failed: {e} — using handcrafted")

def _arcface_embed(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    _init_arcface()
    if _ORT is None: return None
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112,112), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2,0,1))[None,...]
    out = _ORT.run([_OUT], {_IN: img})[0][0].astype(np.float32)
    return l2norm(out)

def _handcrafted_embed(crop_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (128,128), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([g],[0],None,[128],[0,256]).flatten().astype(np.float32)
    hist = hist / (hist.sum()+1e-8)
    small = cv2.resize(g,(8,8),interpolation=cv2.INTER_AREA).astype(np.float32).flatten()
    small = (small - small.mean())/(small.std()+1e-8)
    return l2norm(np.concatenate([hist,small]).astype(np.float32))

# ---- Face detector YuNet -> MP fallback ----
_YUNET = None
def _find_yunet_path() -> Optional[str]:
    mdir = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","models"))
    for name in ["face_detection_yunet_2023mar.onnx","yunet.onnx"]:
        p=os.path.join(mdir,name)
        if os.path.exists(p): return p
    return None

def _yunet_create(input_size):
    global _YUNET
    mpth = _find_yunet_path()
    if mpth is None or not hasattr(cv2, "FaceDetectorYN_create"):
        return None
    if _YUNET is None:
        _YUNET = cv2.FaceDetectorYN_create(mpth, "", input_size, 0.9, 0.3, 5000)
        print(f"[faces_any] YuNet loaded: {mpth}")
    else:
        _YUNET.setInputSize(input_size)
    return _YUNET

try:
    import mediapipe as mp
    _FD = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
except Exception:
    _FD = None

class FaceRecognizer:
    def __init__(self): pass

    def _detect_yunet(self, frame_bgr):
        h,w = frame_bgr.shape[:2]
        det = _yunet_create((w,h))
        if det is None: return None
        _, faces = det.detect(frame_bgr)
        out = []
        if faces is None: return out
        for f in faces:
            x,y,wf,hf = f[:4].astype(int)
            x1,y1,x2,y2 = max(0,x),max(0,y),min(w,x+wf),min(h,y+hf)
            out.append(((x1,y1,x2,y2), float(f[-1])))
        return out

    def _detect_mp(self, frame_bgr):
        out=[]
        if _FD is None: return out
        res = _FD.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res or not res.detections: return out
        h,w = frame_bgr.shape[:2]
        for d in res.detections:
            rb = d.location_data.relative_bounding_box
            x1 = int(max(0, rb.xmin) * w); y1 = int(max(0, rb.ymin) * h)
            x2 = int(min(1.0, rb.xmin + rb.width) * w); y2 = int(min(1.0, rb.ymin + rb.height) * h)
            out.append(((x1,y1,x2,y2), float(d.score[0]) if d.score else 0.0))
        return out

    def detect(self, frame_bgr) -> List[DetFace]:
        faces = self._detect_yunet(frame_bgr)
        if faces is None:
            faces = self._detect_mp(frame_bgr) or []
        h,w = frame_bgr.shape[:2]
        out: List[DetFace] = []
        for (x1,y1,x2,y2),score in faces:
            cx, cy = (x1+x2)//2, (y1+y2)//2
            s = int(max(x2-x1, y2-y1)*1.25)
            x1n,x2n = max(0,cx-s//2), min(w,cx+s//2)
            y1n,y2n = max(0,cy-s//2), min(h,cy+s//2)
            crop = frame_bgr[y1n:y2n, x1n:x2n]
            emb = _arcface_embed(crop) or _handcrafted_embed(crop)
            out.append(DetFace(bbox=(x1n,y1n,x2n,y2n), embedding=emb, score=score))
        return out
