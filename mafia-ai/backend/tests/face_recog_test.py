import os, sys, time, json, urllib.request, math
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]         # .../backend
MODELS_DIR = ROOT / "models"
STORAGE_DIR = ROOT / "storage"
FACEBANK_PATH = STORAGE_DIR / "facebank.json"
THUMBS_DIR = STORAGE_DIR / "facebank"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
THUMBS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Model: MobileFaceNet (ArcFace) ----------
MODEL_CANDIDATES = [
    # Hugging Face mirrors (raw download)
    "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx?download=true",
    "https://huggingface.co/WePrompt/buffalo_sc/resolve/main/w600k_mbf.onnx?download=true",
]
MODEL_PATH = MODELS_DIR / "w600k_mbf.onnx"

def _download_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return
    import urllib.request, shutil, time
    headers = {"User-Agent": "Mozilla/5.0 (MafiaAI/face-test)"}
    last_err = None
    tmp = MODEL_PATH.with_suffix(".part")
    for url in MODEL_CANDIDATES:
        print(f"[model] downloading {url}")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            size = tmp.stat().st_size
            if size < 1_000_000:
                raise RuntimeError(f"Downloaded too small file: {size} bytes")
            tmp.replace(MODEL_PATH)
            print(f"[model] saved to {MODEL_PATH} ({size/1024/1024:.2f} MB)")
            return
        except Exception as e:
            last_err = e
            print(f"[model] failed: {e}")
            try: tmp.unlink(missing_ok=True)
            except: pass
            time.sleep(1.0)
    raise RuntimeError(f"Could not download model to {MODEL_PATH}. "
                       f"Try manual download and place the file there.")


class FaceEmbedder:
    """
    ONNX MobileFaceNet (512-D) + ArcFace препроцесс.
    Вход: BGR-кроп с лицом. Выход: L2-нормированный вектор (512,).
    """
    def __init__(self, providers=None):
        import onnxruntime as ort
        _download_model()
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(MODEL_PATH), providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name

    @staticmethod
    def preprocess(face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img / 127.5) - 1.0            # [-1,1]
        img = np.transpose(img, (2, 0, 1))   # C,H,W
        img = np.expand_dims(img, 0)         # 1,C,H,W
        return img.astype(np.float32)

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        inp = self.preprocess(face_bgr)
        vec = self.sess.run([self.out], {self.inp: inp})[0][0].astype(np.float32)
        n = np.linalg.norm(vec) + 1e-6
        return vec / n

# ---------- Detector: MediaPipe Face Detection ----------
class MPFaceDetector:
    def __init__(self, min_conf=0.6, model_selection=1):
        import mediapipe as mp
        self.det = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_conf
        )

    def detect(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.det.process(rgb)
        out = []
        if not res.detections:
            return out
        for det in res.detections:
            r = det.location_data.relative_bounding_box
            x1 = int(max(0, r.xmin) * W); y1 = int(max(0, r.ymin) * H)
            x2 = int(min(1.0, r.xmin + r.width) * W); y2 = int(min(1.0, r.ymin + r.height) * H)
            # padding (немного лба/подбородка) — улучшает устойчивость эмбеддингов
            px = int(0.15 * (x2 - x1)); py = int(0.20 * (y2 - y1))
            xx1 = max(0, x1 - px); yy1 = max(0, y1 - py)
            xx2 = min(W, x2 + px); yy2 = min(H, y2 + py)
            out.append({
                "bbox": (xx1, yy1, xx2, yy2),
                "score": float(det.score[0] if det.score else 0.0)
            })
        return out

# ---------- Facebank ----------
class FaceBank:
    def __init__(self, path: Path = FACEBANK_PATH):
        self.path = path
        self.items: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.items = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.items = []
        else:
            self.items = []

    def save(self):
        self.path.write_text(json.dumps(self.items, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, name: str, emb: np.ndarray, thumb_bgr: np.ndarray) -> Dict[str, Any]:
        # если имя уже есть — перезапишем эмбеддинг (упрощённый вариант)
        for it in self.items:
            if it["name"].lower() == name.lower():
                it["embedding"] = emb.astype(float).tolist()
                # обновим превью
                thumb_path = THUMBS_DIR / f'{it["id"]}.jpg'
                cv2.imwrite(str(thumb_path), thumb_bgr)
                self.save()
                return it
        # новый
        new_id = 1 + max([it["id"] for it in self.items], default=0)
        thumb_path = THUMBS_DIR / f"{new_id}.jpg"
        cv2.imwrite(str(thumb_path), thumb_bgr)
        rec = {
            "id": new_id,
            "name": name,
            "embedding": emb.astype(float).tolist(),
            "thumb": f"facebank/{new_id}.jpg"
        }
        self.items.append(rec)
        self.save()
        return rec

    def match(self, emb: np.ndarray, threshold: float = 0.50) -> Tuple[int|None, str|None, float]:
        if not self.items:
            return None, None, 0.0
        # сгруппировать по размерности (на будущее, если смешаем модели)
        E = []
        meta = []
        for it in self.items:
            vec = np.array(it["embedding"], dtype=np.float32)
            if vec.shape[0] != emb.shape[0]:
                continue
            E.append(vec); meta.append(it)
        if not E:
            return None, None, 0.0
        E = np.stack(E, axis=0)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)
        em = emb / (np.linalg.norm(emb) + 1e-6)
        sims = E @ em
        j = int(np.argmax(sims))
        simv = float(sims[j])
        if simv >= threshold:
            return meta[j]["id"], meta[j]["name"], simv
        return None, None, simv

# ---------- Main loop ----------
def main(camera_index: int = 0, threshold: float = 0.50):
    # инициализация
    cap = None
    for api in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
        try:
            cap = cv2.VideoCapture(camera_index, api)
            if cap.isOpened():
                print(f"[camera] opened index={camera_index} api={api}")
                break
        except Exception:
            pass
    if cap is None or not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # при необходимости

    detector = MPFaceDetector(min_conf=0.6, model_selection=1)
    embedder = FaceEmbedder()
    bank = FaceBank()

    print("[info] Controls:  E — enroll largest face;  R — reload bank;  Q — quit")
    last_t = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01); continue

        # FPS
        now = time.time()
        dt = now - last_t
        last_t = now
        fps = fps * 0.9 + (1.0/dt) * 0.1 if dt > 0 else fps

        faces = detector.detect(frame)
        matches = []
        for f in faces:
            x1,y1,x2,y2 = f["bbox"]
            crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            if crop.size == 0: 
                matches.append({"bbox": f["bbox"], "id": None, "name": None, "sim": 0.0}); 
                continue
            emb = embedder.embed(crop)
            pid, name, sim = bank.match(emb, threshold=threshold)
            matches.append({"bbox": f["bbox"], "id": pid, "name": name, "sim": sim, "embedding": emb, "crop": crop})

        # ENROLL: нажми 'E' — возьмём самое большое лицо и спросим имя в консоли
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        if key in (ord('r'), ord('R')):
            bank._load()
            print("[bank] reloaded")
        if key in (ord('e'), ord('E')):
            if matches:
                # выбрать самое крупное
                def area(bb): x1,y1,x2,y2=bb; return max(0,x2-x1)*max(0,y2-y1)
                best = max(matches, key=lambda m: area(m["bbox"]))
                if "embedding" in best:
                    try:
                        name = input("Введите имя для энролла: ").strip()
                    except Exception:
                        name = ""
                    if not name:
                        name = f"player_{int(time.time())}"
                    rec = bank.add(name, best["embedding"], best["crop"])
                    print(f'[bank] added id={rec["id"]} name={rec["name"]}')
                else:
                    print("[bank] нет валидного кропа для энролла")
            else:
                print("[bank] лиц не найдено для энролла")

        # Overlay
        vis = frame.copy()
        for m in matches:
            x1,y1,x2,y2 = m["bbox"]
            color = (102,182,255) if m["id"] else (140,140,140)
            cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
            label = f'#{m["id"]} {m["name"]}  ({m["sim"]:.2f})' if m["id"] else f'? ({m["sim"]:.2f})'
            cv2.putText(vis, label, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.putText(vis, f"faces:{len(matches)}  bank:{len(bank.items)}  FPS:{fps:.1f}",
                    (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Face Recog Test (E-enroll, R-reload, Q-quit)", vis)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam_idx = 0
    thr = 0.50
    # можно задать через аргументы: python face_recog_test.py 1 0.55
    if len(sys.argv) >= 2:
        try: cam_idx = int(sys.argv[1])
        except: pass
    if len(sys.argv) >= 3:
        try: thr = float(sys.argv[2])
        except: pass
    main(camera_index=cam_idx, threshold=thr)
