# backend/video/face_recog_adv.py
from __future__ import annotations
import os, json, shutil, urllib.request
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
STORAGE_DIR = ROOT / "storage"
FACEBANK_PATH = STORAGE_DIR / "facebank_v2.json"
THUMBS_DIR = STORAGE_DIR / "facebank_v2"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
THUMBS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Align ----------
ARC_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

def align_by_5pts(bgr: np.ndarray, pts5: np.ndarray, out_size=(112,112)) -> np.ndarray:
    assert pts5.shape == (5,2)
    dst = ARC_TEMPLATE.copy()
    if out_size != (112,112):
        sx = out_size[0] / 112.0
        sy = out_size[1] / 112.0
        dst[:,0] *= sx; dst[:,1] *= sy
    M, _ = cv2.estimateAffinePartial2D(pts5.astype(np.float32), dst, method=cv2.LMEDS)
    aligned = cv2.warpAffine(bgr, M, out_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return aligned

# ---------- MediaPipe face detector + 5pts ----------
class MPFaceDetector:
    def __init__(self, det_conf=0.6, mesh_track_conf=0.6):
        import mediapipe as mp
        self.det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=det_conf)
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=12, refine_landmarks=False,
            min_detection_confidence=det_conf, min_tracking_confidence=mesh_track_conf
        )

    @staticmethod
    def _mesh_5pts(lmset, W, H) -> Optional[np.ndarray]:
        ids = [33, 263, 1, 61, 291]  # rightEyeOut, leftEyeOut, nose, mouthL, mouthR
        pts = []
        for i in ids:
            li = lmset.landmark[i]
            x = np.clip(li.x, 0.0, 1.0) * W
            y = np.clip(li.y, 0.0, 1.0) * H
            pts.append((x,y))
        pts = np.array(pts, dtype=np.float32)
        pts[[0,1]] = pts[[1,0]]  # swap → leftEye, rightEye, nose, mouthL, mouthR
        return pts

    def detect_and_landmarks(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        det_res = self.det.process(rgb)
        out = []
        if not det_res.detections:
            return out

        mesh_res = self.mesh.process(rgb)
        mesh_sets = mesh_res.multi_face_landmarks or []

        def bbox_from_mesh(lmset):
            xs = [np.clip(lm.x, 0.0, 1.0)*W for lm in lmset.landmark]
            ys = [np.clip(lm.y, 0.0, 1.0)*H for lm in lmset.landmark]
            return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

        mesh_boxes = [bbox_from_mesh(m) for m in mesh_sets]

        def iou(a,b):
            ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
            ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
            iw,ih=max(0,ix2-ix1),max(0,iy2-iy1)
            inter=iw*ih; ra=(ax2-ax1)*(ay2-ay1); rb=(bx2-bx1)*(by2-by1)
            return inter/max(1.0, ra+rb-inter)

        for det in det_res.detections:
            r = det.location_data.relative_bounding_box
            x1 = int(max(0, r.xmin) * W); y1 = int(max(0, r.ymin) * H)
            x2 = int(min(1.0, r.xmin + r.width) * W); y2 = int(min(1.0, r.ymin + r.height) * H)
            px = int(0.12*(x2-x1)); py = int(0.18*(y2-y1))
            xx1,yy1 = max(0, x1-px), max(0, y1-py)
            xx2,yy2 = min(W, x2+px), min(H, y2+py)
            bb = (xx1,yy1,xx2,yy2)

            best_pts = None
            if mesh_sets:
                ious = [iou(bb, mb) for mb in mesh_boxes]
                j = int(np.argmax(ious))
                if ious[j] > 0.05:
                    best_pts = self._mesh_5pts(mesh_sets[j], W, H)

            out.append({"bbox": bb, "score": float(det.score[0] if det.score else 0.0), "pts5": best_pts})
        return out

# ---------- Embedders ----------
class BaseEmbedder:
    dim: int = 512
    def embed(self, aligned_112x112_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class MobileFaceNetEmbedder(BaseEmbedder):
    dim = 512
    CANDIDATES = [
        "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx?download=true",
        "https://huggingface.co/WePrompt/buffalo_sc/resolve/main/w600k_mbf.onnx?download=true",
    ]
    def __init__(self, model_path: Optional[Path] = None, providers=None):
        import onnxruntime as ort
        self.model_path = model_path or (MODELS_DIR / "w600k_mbf.onnx")
        self._ensure_model()
        if providers is None: providers=["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(self.model_path), providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name
    def _ensure_model(self):
        if self.model_path.exists() and self.model_path.stat().st_size>1_000_000: return
        headers={"User-Agent":"MafiaAI/face-adv"}
        tmp = self.model_path.with_suffix(".part")
        last=None
        for url in self.CANDIDATES:
            try:
                print("[model] downloading", url)
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as r, open(tmp,"wb") as f:
                    shutil.copyfileobj(r,f)
                if tmp.stat().st_size<1_000_000: raise RuntimeError("too small file")
                tmp.replace(self.model_path); return
            except Exception as e:
                last=e
                try: tmp.unlink(missing_ok=True)
                except: pass
        raise RuntimeError(f"Cannot download MobileFaceNet: {last}")
    @staticmethod
    def _prep(img112: np.ndarray)->np.ndarray:
        x = cv2.cvtColor(img112, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = (x/127.5)-1.0
        x = np.transpose(x,(2,0,1))[None,...].astype(np.float32)
        return x
    def embed(self, img112: np.ndarray) -> np.ndarray:
        vec = self.sess.run([self.out], {self.inp: self._prep(img112)})[0][0].astype(np.float32)
        vec /= (np.linalg.norm(vec)+1e-6)
        return vec

class ArcFaceR100Embedder(BaseEmbedder):
    dim = 512
    def __init__(self, model_path: Path, providers=None):
        import onnxruntime as ort
        if not Path(model_path).exists():
            raise FileNotFoundError(f"ArcFace r100 model not found: {model_path}")
        if providers is None: providers=["CPUExecutionProvider"]
        self.model_path = Path(model_path)
        print(f"[embedder] r100 model: {self.model_path}")
        self.sess = ort.InferenceSession(str(self.model_path), providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name
    @staticmethod
    def _prep(img112: np.ndarray)->np.ndarray:
        x = cv2.cvtColor(img112, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = (x/127.5)-1.0
        x = np.transpose(x,(2,0,1))[None,...].astype(np.float32)
        return x
    def embed(self, img112: np.ndarray) -> np.ndarray:
        vec = self.sess.run([self.out], {self.inp: self._prep(img112)})[0][0].astype(np.float32)
        vec /= (np.linalg.norm(vec)+1e-6)
        return vec

# ---------- Auto-discovery ----------
def _find_r100_model_path(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Provided r100 model not found: {explicit}")

    env = os.getenv("FACE_R100_ONNX") or os.getenv("ARC_R100_MODEL")
    if env:
        p = Path(env)
        if p.exists(): return p.resolve()
        print(f"[warn] env model path not found: {p}")

    patterns = [
        "*arcface1*.onnx",
        "*iresnet100*.onnx",
        "*glintr100*.onnx",
        "arcfaceresnet100-11.onnx",
        "*r100*.onnx",
    ]
    search_roots = [MODELS_DIR, ROOT]
    candidates: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pat in patterns:
            candidates += list(root.rglob(pat))
    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        raise FileNotFoundError(
            "r100 .onnx модель не найдена. Укажи путь через аргумент/ENV или помести в backend/models/."
        )
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    best = candidates[0].resolve()
    print(f"[embedder] r100 auto-selected: {best} ({best.stat().st_size/1024/1024:.1f} MB)")
    return best

def make_embedder(kind: str, model_path: Optional[str]) -> BaseEmbedder:
    kind = (kind or "mbf").lower()
    if kind == "r100":
        path = _find_r100_model_path(model_path)
        return ArcFaceR100Embedder(path)
    return MobileFaceNetEmbedder(Path(model_path) if model_path else None)

# ---------- Quality & FaceBank ----------
def laplacian_var(gray: np.ndarray)->float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def face_quality_score(img112: np.ndarray)->float:
    g = cv2.cvtColor(img112, cv2.COLOR_BGR2GRAY)
    blur = laplacian_var(g)
    mean = float(g.mean())
    b = max(0.0, min(1.0, (blur-30.0)/300.0))
    m = max(0.0, min(1.0, (mean-40.0)/60.0))
    return 0.7*b + 0.3*m

class FaceBankV2:
    def __init__(self, path: Path = FACEBANK_PATH):
        self.path = path
        self.items: List[Dict[str, Any]] = []
        self._load()
    def _load(self):
        if self.path.exists():
            try: self.items = json.loads(self.path.read_text(encoding="utf-8"))
            except: self.items = []
    def save(self):
        self.path.write_text(json.dumps(self.items, ensure_ascii=False, indent=2), encoding="utf-8")
    def add_or_update(self, name: str, embs: List[np.ndarray], thumb_bgr: np.ndarray) -> Dict[str, Any]:
        mean = np.mean(np.stack(embs,0), axis=0)
        mean = mean / (np.linalg.norm(mean)+1e-6)
        gallery = [e.astype(float).tolist() for e in embs]
        for it in self.items:
            if it["name"].lower()==name.lower():
                it["mean"] = mean.astype(float).tolist()
                it["gallery"] = gallery
                thumb = THUMBS_DIR / f'{it["id"]}.jpg'
                cv2.imwrite(str(thumb), thumb_bgr)
                self.save(); return it
        new_id = 1+max([it["id"] for it in self.items], default=0)
        thumb = THUMBS_DIR / f"{new_id}.jpg"
        cv2.imwrite(str(thumb), thumb_bgr)
        rec = {"id": new_id, "name": name, "mean": mean.astype(float).tolist(), "gallery": gallery, "thumb": f"facebank_v2/{new_id}.jpg"}
        self.items.append(rec); self.save(); return rec
    def match(self, emb: np.ndarray, threshold_mean=0.55, threshold_gallery=0.55) -> Tuple[Optional[int], Optional[str], float]:
        if not self.items: return None, None, 0.0
        e = emb / (np.linalg.norm(emb)+1e-6)
        best = (None, None, 0.0)
        for it in self.items:
            mean = np.array(it["mean"], dtype=np.float32)
            sim_mean = float(mean @ e)
            sim_gal = sim_mean
            if it.get("gallery"):
                G = np.array(it["gallery"], dtype=np.float32)
                G = G / (np.linalg.norm(G, axis=1, keepdims=True)+1e-6)
                sim_gal = float(np.max(G @ e))
            score = max(sim_mean, sim_gal)
            if score > best[2]:
                best = (it["id"], it["name"], score)
        pid, name, score = best
        thr = max(threshold_mean, threshold_gallery)
        if score >= thr:
            return pid, name, score
        return None, None, score
