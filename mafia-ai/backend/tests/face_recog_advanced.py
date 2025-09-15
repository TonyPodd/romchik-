import os, sys, time, json, math, shutil, urllib.request
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np

# ---------------- Paths & IO ----------------
ROOT = Path(__file__).resolve().parents[1]  # .../backend
MODELS_DIR = ROOT / "models"
STORAGE_DIR = ROOT / "storage"
FACEBANK_PATH = STORAGE_DIR / "facebank_v2.json"
THUMBS_DIR = STORAGE_DIR / "facebank_v2"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
THUMBS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- ArcFace 5-point template (112x112) ----------------
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

# ---------------- MediaPipe: detection + 5-pts from FaceMesh ----------------
class MPFaceDetector:
    def __init__(self, det_conf=0.6, mesh_track_conf=0.6):
        import mediapipe as mp
        self.det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=det_conf)
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=8, refine_landmarks=False,
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
        pts[[0,1]] = pts[[1,0]]  # swap to: leftEye, rightEye, nose, mouthL, mouthR
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

# ---------------- Embedders ----------------
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

# -------- Auto-discovery for r100 --------
def _find_r100_model_path(explicit: Optional[str]) -> Path:
    """
    Order:
      1) explicit path
      2) env FACE_R100_ONNX or ARC_R100_MODEL
      3) search in MODELS_DIR and ROOT recursively by patterns
    Select the largest .onnx among candidates.
    """
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Provided r100 model not found: {explicit}")

    env = os.getenv("FACE_R100_ONNX") or os.getenv("ARC_R100_MODEL")
    if env:
        p = Path(env)
        if p.exists():
            return p.resolve()
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
            "r100 .onnx модель не найдена. Укажи путь через --model или переменную окружения "
            "FACE_R100_ONNX/ARC_R100_MODEL, либо положи файл в backend/models/. "
            f"Искал в: {', '.join(str(r) for r in search_roots)} по маскам: {', '.join(patterns)}"
        )
    # выбрать самый большой файл
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    best = candidates[0].resolve()
    print(f"[embedder] r100 auto-selected: {best} ({best.stat().st_size/1024/1024:.1f} MB)")
    return best

def make_embedder(kind: str, model_path: Optional[str]) -> 'BaseEmbedder':
    kind = (kind or "mbf").lower()
    if kind == "r100":
        path = _find_r100_model_path(model_path)
        return ArcFaceR100Embedder(path)
    return MobileFaceNetEmbedder(Path(model_path) if model_path else None)

# ---------------- Quality & Matching ----------------
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

# ---------------- Main ----------------
def main(camera_index=0, embedder_kind="mbf", model_path=None, n_enroll=12):
    cap=None
    for api in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
        try:
            cap=cv2.VideoCapture(camera_index, api)
            if cap.isOpened():
                print(f"[camera] opened index={camera_index} api={api}"); break
        except: pass
    if cap is None or not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    det = MPFaceDetector()
    emb = make_embedder(embedder_kind, model_path)
    bank = FaceBankV2()

    print("[info] Controls: E — enroll; R — reload bank; Q — quit")
    print("[info] During enroll, поворачивайте голову (влево/вправо/вверх/вниз): собираем разные ракурсы.")
    last_t=time.time(); fps=0.0
    enrolling=False; samples=[]; enroll_name=""; last_capture=0.0; enroll_face_thumb=None

    def try_capture_sample(aligned):
        nonlocal samples, enroll_face_thumb
        q = face_quality_score(aligned)
        if q < 0.35:
            return False
        if len(samples)==0:
            samples.append(emb.embed(aligned))
            enroll_face_thumb = aligned.copy()
            return True
        e = emb.embed(aligned)
        S = np.array(samples, dtype=np.float32)
        S = S / (np.linalg.norm(S, axis=1, keepdims=True)+1e-6)
        en = e/(np.linalg.norm(e)+1e-6)
        sim = float(np.max(S @ en))
        if sim < 0.90:
            samples.append(e); return True
        return False

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01); continue

        now=time.time()
        dt=now-last_t; last_t=now
        fps = 0.9*fps + 0.1*(1.0/dt if dt>0 else fps)

        faces = det.detect_and_landmarks(frame)
        matches=[]
        for f in faces:
            x1,y1,x2,y2 = f["bbox"]
            pts5 = f["pts5"]
            if pts5 is None:
                crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                if crop.size==0: continue
                aligned = cv2.resize(crop,(112,112))
            else:
                aligned = align_by_5pts(frame, pts5, (112,112))

            if enrolling and now-last_capture>0.25:
                if try_capture_sample(aligned):
                    last_capture=now

            matches.append({"bbox":f["bbox"], "aligned":aligned})

        vis = frame.copy()
        for m in matches:
            a = m["aligned"]
            e = emb.embed(a)
            pid,name,score = bank.match(e, threshold_mean=0.55 if embedder_kind=="r100" else 0.50,
                                           threshold_gallery=0.60 if embedder_kind=="r100" else 0.55)
            x1,y1,x2,y2 = m["bbox"]
            color = (102,182,255) if pid else (140,140,140)
            cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
            label = f'#{pid} {name} ({score:.2f})' if pid else f'? ({score:.2f})'
            cv2.putText(vis,label,(x1,max(0,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)

        if enrolling:
            cv2.putText(vis, f"ENROLL {enroll_name}  {len(samples)}/{n_enroll}",
                        (12,28), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2,cv2.LINE_AA)
        else:
            cv2.putText(vis, f"faces:{len(matches)}  bank:{len(bank.items)}  FPS:{fps:.1f}",
                        (12,28), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)

        cv2.imshow("Face Recog ADV (E-enroll, R-reload, Q-quit)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'),ord('Q')): break
        if key in (ord('r'),ord('R')):
            bank._load(); print("[bank] reloaded")
        if key in (ord('e'),ord('E')):
            if not enrolling:
                try:
                    enroll_name = input("Введите имя для энролла: ").strip() or f"player_{int(time.time())}"
                except: enroll_name = f"player_{int(time.time())}"
                enrolling=True; samples=[]; last_capture=0.0; enroll_face_thumb=None
            else:
                if len(samples) >= max(6, int(0.5*n_enroll)):
                    rec = bank.add_or_update(enroll_name, samples, enroll_face_thumb if enroll_face_thumb is not None else vis)
                    print(f'[bank] saved id={rec["id"]} name={rec["name"]} samples={len(samples)}')
                else:
                    print(f"[bank] мало образцов ({len(samples)}/{n_enroll}), повторите энролл.")
                enrolling=False; samples=[]

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--embedder", choices=["mbf","r100"], default="mbf")
    ap.add_argument("--model", type=str, default=None, help="path to r100 onnx (optional with auto-discovery)")
    ap.add_argument("--n_enroll", type=int, default=12)
    args = ap.parse_args()
    main(camera_index=args.cam, embedder_kind=args.embedder, model_path=args.model, n_enroll=args.n_enroll)


# python tests\face_recog_advanced.py --embedder r100
