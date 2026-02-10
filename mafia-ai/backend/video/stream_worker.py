# backend/video/stream_worker.py
from __future__ import annotations

import asyncio
import os
import sys
import time
import platform
import shutil
import urllib.request
from typing import Optional, Callable, Awaitable, Dict, Any, Tuple, List

import cv2
import numpy as np

from video.gestures import GestureDetector
from storage import players as P
from compreface_client import get_compreface_client

EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]

# ---------- Render modes ----------
RENDER_RAW   = 0  # чистое видео
RENDER_TABLE = 1  # только линия/полигон стола
RENDER_FULL  = 2  # лица + руки + стол

# ---------------- Camera helpers ----------------

def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}

def _try_open(index: int, api: Optional[int]) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(index, api) if api is not None else cv2.VideoCapture(index)
    return cap if cap.isOpened() else None


def _open_capture(idx: int) -> cv2.VideoCapture:
    if sys.platform.startswith("win"):
        order_env = (os.getenv("OPENCV_BACKEND") or "AUTO").upper()
        auto = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
        named = {
            "MSMF": [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY],
            "DSHOW": [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY],
            "ANY": [cv2.CAP_ANY],
            "AUTO": auto,
        }
        apis = named.get(order_env, auto)
    else:
        apis = [cv2.CAP_ANY]

    last_exc: Optional[Exception] = None
    for api in apis:
        cap = _try_open(idx, api)
        if cap:
            print(f"[camera] opened index={idx} api={api}")
            try:
                default_w = "960" if _is_apple_silicon() else "1280"
                default_h = "540" if _is_apple_silicon() else "720"
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(os.getenv("CAM_WIDTH", default_w)))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(os.getenv("CAM_HEIGHT", default_h)))
                if os.getenv("FORCE_MJPEG", "0") == "1":
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception as e:
                last_exc = e
            return cap
    raise RuntimeError(f"Cannot open camera index {idx}; tried apis={apis}; last={last_exc}")


def _point_in_poly(px: Tuple[int, int], poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly.astype(np.int32), (int(px[0]), int(px[1])), False) >= 0


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _expand_bbox(
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int, int],
    pad_ratio_x: float = 0.14,
    pad_ratio_y: float = 0.20,
) -> Tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    padx = int(bw * pad_ratio_x)
    pady = int(bh * pad_ratio_y)
    xx1 = max(0, x1 - padx)
    yy1 = max(0, y1 - pady)
    xx2 = min(w, x2 + padx)
    yy2 = min(h, y2 + pady)
    return xx1, yy1, xx2, yy2


def _face_appearance_descriptor(face_bgr: np.ndarray) -> np.ndarray:
    """
    Легкий внешний вид-дескриптор (fallback, когда нет ArcFace).
    Нужен для различения нескольких игроков в LANDMARKS-режиме.
    """
    if face_bgr is None or face_bgr.size == 0:
        return np.zeros(432, dtype=np.float32)

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (20, 20), interpolation=cv2.INTER_LINEAR)
    gray = cv2.equalizeHist(gray)

    pix = gray.astype(np.float32).reshape(-1) / 255.0  # 400
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).reshape(-1).astype(np.float32)  # 32
    hist = hist / (float(hist.sum()) + 1e-6)

    desc = np.concatenate([pix, hist], axis=0).astype(np.float32)
    desc = desc - float(desc.mean())
    desc = desc / (np.linalg.norm(desc) + 1e-6)
    return desc


# ---------------- Face identification backends ----------------

class _FaceBackendBase:
    sim_threshold: float = 0.60  # Higher threshold for better person differentiation
    def analyze(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        raise NotImplementedError


class _FaceBackendONNX(_FaceBackendBase):
    """ArcFace ONNX + MediaPipe FaceDetection для bbox."""
    def __init__(self, sim_threshold: float = 0.60):  # Higher threshold for better accuracy
        import onnxruntime as ort
        import mediapipe as mp
        from pathlib import Path

        self.sim_threshold = sim_threshold
        det_default = "0" if _is_apple_silicon() else "1"
        self.det = mp.solutions.face_detection.FaceDetection(
            model_selection=_safe_int(os.getenv("FACE_DET_MODEL_SELECTION", det_default), default=1),
            min_detection_confidence=float(os.getenv("FACE_DET_MIN_CONFIDENCE", "0.6")),
        )

        MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        env_model_path = os.getenv("FACE_ONNX_MODEL")
        self.model_path = Path(env_model_path) if env_model_path else (MODELS_DIR / "arcface.onnx")
        if not self.model_path.is_absolute():
            self.model_path = MODELS_DIR / self.model_path
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.model_path.exists():
            env_model_url = os.getenv("FACE_ONNX_MODEL_URL")
            urls = [env_model_url] if env_model_url else []
            urls.extend([
                "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx?download=true",
                "https://huggingface.co/WePrompt/buffalo_sc/resolve/main/w600k_mbf.onnx?download=true",
                "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-11.onnx",
            ])
            self._download_model(urls, self.model_path)

        available = list(ort.get_available_providers())
        providers_env = [p.strip() for p in os.getenv("FACE_ONNX_PROVIDERS", "").split(",") if p.strip()]
        if providers_env:
            preferred_providers = providers_env
        elif _is_apple_silicon():
            preferred_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            preferred_providers = ["CPUExecutionProvider"]
        providers = [p for p in preferred_providers if p in available] or ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        default_threads = "2" if _is_apple_silicon() else "0"
        threads = max(0, _safe_int(os.getenv("FACE_ONNX_THREADS", default_threads), default=0))
        if threads > 0:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1

        self.sess = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=providers)
        print(f"[face] ONNX model loaded: {self.model_path} providers={self.sess.get_providers()}")
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    @staticmethod
    def _download_model(urls: List[Optional[str]], dst_path) -> None:
        filtered = [u for u in urls if isinstance(u, str) and u.strip()]
        if not filtered:
            raise RuntimeError("No ONNX model URL configured")

        headers = {"User-Agent": "MafiaAI/face-stream-worker"}
        tmp_path = dst_path.with_suffix(".part")
        last_error: Optional[Exception] = None
        for url in filtered:
            try:
                print(f"[arcface] downloading model from: {url}")
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_path, "wb") as f:
                    shutil.copyfileobj(resp, f)
                if tmp_path.stat().st_size < 1_000_000:
                    raise RuntimeError("downloaded model file is too small")
                tmp_path.replace(dst_path)
                return
            except Exception as e:
                last_error = e
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                print(f"[arcface] download failed from {url}: {e}")

        raise RuntimeError(f"Cannot download face ONNX model: {last_error}")

    @staticmethod
    def _preprocess(face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img / 127.5) - 1.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0).astype(np.float32)
        return img

    def _embed(self, face_bgr: np.ndarray) -> np.ndarray:
        inp = self._preprocess(face_bgr)
        out = self.sess.run([self.output_name], {self.input_name: inp})[0]
        emb = out[0].astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-6)

    def analyze(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.det.process(rgb)
        out: List[Dict[str, Any]] = []
        if not res.detections:
            return out
        for det in res.detections:
            r = det.location_data.relative_bounding_box
            x1 = int(max(0, r.xmin) * W); y1 = int(max(0, r.ymin) * H)
            x2 = int(min(1.0, r.xmin + r.width) * W); y2 = int(min(1.0, r.ymin + r.height) * H)
            px = int(0.12 * (x2 - x1)); py = int(0.18 * (y2 - y1))
            xx1 = max(0, x1 - px); yy1 = max(0, y1 - py)
            xx2 = min(W, x2 + px); yy2 = min(H, y2 + py)
            crop = frame_bgr[yy1:yy2, xx1:xx2]
            if crop.size == 0:
                continue
            emb = self._embed(crop)
            out.append({"bbox": (xx1, yy1, xx2, yy2), "score": float(det.score[0] if det.score else 0.0), "embedding": emb})
        return out


class _FaceBackendLandmarks(_FaceBackendBase):
    """Фолбэк: MediaPipe FaceMesh → эмбеддинг как нормализованные 2D-координаты."""
    def __init__(self, sim_threshold: float = 0.85):
        import mediapipe as mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=8, refine_landmarks=False,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        self.sim_threshold = sim_threshold

    def analyze(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        out: List[Dict[str, Any]] = []
        if not res.multi_face_landmarks:
            return out
        for lmset in res.multi_face_landmarks:
            xs = [lm.x for lm in lmset.landmark]; ys = [lm.y for lm in lmset.landmark]
            minx, maxx = max(0.0, min(xs)), min(1.0, max(xs))
            miny, maxy = max(0.0, min(ys)), min(1.0, max(ys))
            x1, y1 = int(minx * W), int(miny * H); x2, y2 = int(maxx * W), int(maxy * H)
            bw = maxx - minx + 1e-6; bh = maxy - miny + 1e-6
            vec = []
            for lm in lmset.landmark:
                vec.append((lm.x - minx) / bw); vec.append((lm.y - miny) / bh)
            emb = np.array(vec, dtype=np.float32)
            emb = emb - emb.mean()
            emb = emb / (np.linalg.norm(emb) + 1e-6)
            crop = frame_bgr[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            appearance = _face_appearance_descriptor(crop)
            out.append({"bbox": (x1, y1, x2, y2), "score": 1.0, "embedding": emb, "appearance": appearance})
        return out


def _make_face_backend_initial() -> _FaceBackendBase:
    use = (os.getenv("FACE_BACKEND") or "AUTO").upper()
    if use == "LANDMARKS":
        print("[face] using LANDMARKS backend")
        return _FaceBackendLandmarks(sim_threshold=float(os.getenv("FACE_SIM_THRESHOLD", "0.85")))
    print("[face] using ONNX backend (auto-fallback enabled)")
    try:
        return _FaceBackendONNX(sim_threshold=float(os.getenv("FACE_SIM_THRESHOLD", "0.52")))
    except Exception as e:
        if use == "ONNX":
            raise RuntimeError(f"FACE_BACKEND=ONNX but ONNX init failed: {e}") from e
        print(f"[face] ONNX init failed: {e}. Falling back to LANDMARKS.")
        return _FaceBackendLandmarks(sim_threshold=float(os.getenv("FACE_SIM_THRESHOLD", "0.85")))


# ---------------- Main stream ----------------

class GestureStream:
    def __init__(
        self,
        on_event: EventCallback,
        camera_index: int = 0,
        table_y_ratio: float = 0.80,
        fps: int = 30,  # Higher FPS for smooth bounding box rendering
        width: int = 1280,
        height: int = 720,
    ):
        self.on_event = on_event
        self.camera_index = camera_index
        self.table_y_ratio = table_y_ratio
        self.fps = max(5, fps)
        self.width = width
        self.height = height
        self._calibration_only: bool = False  # в этом режиме скрываем лица/жесты и не считаем их

        self._det = GestureDetector(table_y_ratio=table_y_ratio)
        self._face: _FaceBackendBase = _make_face_backend_initial()
        self._face_failed = False
        self._compreface = get_compreface_client()

        if _is_apple_silicon():
            default_every = "4" if self._compreface.is_active() else "3"
            default_max_side = "768"
        else:
            default_every = "2" if self._compreface.is_active() else "1"
            default_max_side = "0"
        self._face_analyze_every = max(1, _safe_int(os.getenv("FACE_ANALYZE_EVERY_N", default_every), default=1))
        self._face_analyze_max_side = max(0, _safe_int(os.getenv("FACE_ANALYZE_MAX_SIDE", default_max_side), default=0))
        self._compreface_match_interval_sec = float(os.getenv("COMPREFACE_MATCH_INTERVAL_SEC", "0.75"))
        self._compreface_cache_ttl_sec = float(os.getenv("COMPREFACE_CACHE_TTL_SEC", "1.25"))
        self._compreface_min_face_size = max(24, _safe_int(os.getenv("COMPREFACE_MIN_FACE_SIZE", "72"), default=72))
        self._face_overlay_light = os.getenv("FACE_OVERLAY_LIGHT", "1" if _is_apple_silicon() else "0").strip().lower() in {"1", "true", "yes"}

        self._cap: Optional[cv2.VideoCapture] = None
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False

        # Режимы
        self.detect_enabled: bool = True
        self.render_mode: int = RENDER_FULL
        self.gestures_enabled: bool = True
        self.face_match_enabled: bool = True

        # Буферы
        self._last_raw_jpeg: Optional[bytes] = None
        self._last_jpeg: Optional[bytes] = None
        self._jpeg_lock = asyncio.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._frame_lock = asyncio.Lock()

        self._table_poly_norm: Optional[List[Tuple[float, float]]] = None
        self._frame_counter: int = 0  # Для анимации
        self._thumb_desc_cache: Dict[int, np.ndarray] = {}
        self._thumb_desc_stamp: Dict[int, float] = {}
        self._last_faces: List[Dict[str, Any]] = []
        self._last_matches: List[Dict[str, Any]] = []
        self._compreface_face_cache: List[Dict[str, Any]] = []

    # --- Control API ---

    def set_detect_enabled(self, flag: bool):
        self.detect_enabled = bool(flag)

    def set_render_mode(self, mode: int):
        self.render_mode = int(mode)

    def set_gestures_enabled(self, flag: bool):
        self.gestures_enabled = bool(flag)

    def set_face_match_enabled(self, flag: bool):
        self.face_match_enabled = bool(flag)
        self._compreface_face_cache.clear()
        self._last_matches = []

    def begin_table_calibration(self):
        """Вызывайте при входе в шаг калибровки стола."""
        self.set_detect_enabled(False)
        self.set_render_mode(RENDER_TABLE)

    def end_table_calibration(self):
        """Вызывайте при завершении калибровки."""
        self.set_render_mode(RENDER_FULL)
        self.set_detect_enabled(True)

    # --- Lifecycle ---

    async def start(self):
        if self._running:
            return
        self._cap = _open_capture(self.camera_index)

        self._running = True
        print(f"[GestureStream] started (camera={self.camera_index}, fps={self.fps})")

        # warmup JPEG
        for _ in range(30):
            ok, frame = await asyncio.to_thread(self._cap.read)
            if not ok or frame is None:
                await asyncio.sleep(0.02)
                continue
            if frame.size and float(frame.mean()) > 1.0:
                async with self._frame_lock:
                    self._last_frame = frame.copy()
                ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ok2:
                    async with self._jpeg_lock:
                        self._last_jpeg = buf.tobytes()
                        self._last_raw_jpeg = self._last_jpeg
                break
            await asyncio.sleep(0.01)

        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._running = False
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            finally:
                self._task = None
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        print("[GestureStream] stopped")

    # --- Table polygon helpers ---

    def _poly_px(self, w: int, h: int) -> Optional[np.ndarray]:
        if not self._table_poly_norm:
            return None
        pts = np.array([[p[0] * w, p[1] * h] for p in self._table_poly_norm], dtype=np.float32)
        return pts

    def set_table_polygon_norm(self, poly_norm: List[Tuple[float, float]]):
        if len(poly_norm) < 3:
            raise ValueError("table polygon must have at least 3 points")
        self._table_poly_norm = poly_norm

    def clear_table_polygon(self):
        self._table_poly_norm = None

    # --- Improved autodetect ---

    @staticmethod
    def _poly_clockwise(pts: np.ndarray) -> np.ndarray:
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        order = np.argsort(ang)
        return pts[order]

    async def auto_detect_table(self) -> Optional[List[Tuple[float, float]]]:
        async with self._frame_lock:
            frame = None if self._last_frame is None else self._last_frame.copy()
        if frame is None:
            return None

        h, w = frame.shape[:2]
        roi = frame[int(h * 0.35):, :]
        y0 = int(h * 0.35)

        cand_polys: List[np.ndarray] = []

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
                cand_polys.append(approx)

        edges2 = cv2.Canny(g, 80, 160)
        lines = cv2.HoughLinesP(edges2, 1, np.pi / 180, threshold=120,
                                minLineLength=int(min(w, h) * 0.25), maxLineGap=14)
        if lines is not None:
            mask = np.zeros_like(edges2)
            for l in lines[:200]:
                x1, y1, x2, y2 = l[0]
                cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), 1)
            cnts2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts2:
                big = max(cnts2, key=cv2.contourArea).reshape(-1, 2).astype(np.int32)
                big[:, 1] += y0
                hull2 = cv2.convexHull(big).reshape(-1, 2)
                peri2 = cv2.arcLength(hull2, True)
                approx2 = cv2.approxPolyDP(hull2, 0.02 * peri2, True).reshape(-1, 2)
                if approx2.shape[0] >= 4:
                    cand_polys.append(approx2)

        try:
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
            X = lab.reshape(-1, 3).astype(np.float32)
            K = 3
            _, labels, _ = cv2.kmeans(
                X, K, None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                2, cv2.KMEANS_PP_CENTERS
            )
            labels = labels.reshape(lab.shape[:2])
            half = labels[labels.shape[0] // 2:, :]
            vals, counts = np.unique(half, return_counts=True)
            kdom = int(vals[np.argmax(counts)])
            m = (labels == kdom).astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=1)
            cnts3, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts3:
                big3 = max(cnts3, key=cv2.contourArea).reshape(-1, 2).astype(np.int32)
                big3[:, 1] += y0
                hull3 = cv2.convexHull(big3).reshape(-1, 2)
                peri3 = cv2.arcLength(hull3, True)
                approx3 = cv2.approxPolyDP(hull3, 0.02 * peri3, True).reshape(-1, 2)
                if approx3.shape[0] >= 4:
                    cand_polys.append(approx3)
        except Exception:
            pass

        def score(poly: np.ndarray) -> float:
            area = float(cv2.contourArea(poly))
            if area < 0.02 * w * h:
                return 0.0
            rect = cv2.minAreaRect(poly.astype(np.float32))
            box = cv2.boxPoints(rect)
            rect_area = cv2.contourArea(box.astype(np.float32))
            comp = float(area / max(rect_area, 1.0))
            return area * comp

        if not cand_polys:
            return None

        cand_polys = [self._poly_clockwise(p.astype(np.float32)) for p in cand_polys]
        best = max(cand_polys, key=score)
        peri = cv2.arcLength(best, True)
        best = cv2.approxPolyDP(best, 0.015 * peri, True).reshape(-1, 2)

        poly_norm = [(float(x) / w, float(y) / h) for (x, y) in best]
        self._table_poly_norm = poly_norm
        return poly_norm

    # --- Face backend failover ---

    def _fallback_face_backend(self):
        if not self._face_failed and not isinstance(self._face, _FaceBackendLandmarks):
            print("[face] runtime error → switching to LANDMARKS backend")
            self._face = _FaceBackendLandmarks(sim_threshold=float(os.getenv("FACE_SIM_THRESHOLD", "0.85")))
            self._face_failed = True

    def _safe_face_analyze(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        analyze_frame = frame
        scale = 1.0
        if self._face_analyze_max_side > 0:
            h, w = frame.shape[:2]
            max_side = max(h, w)
            if max_side > self._face_analyze_max_side:
                scale = float(self._face_analyze_max_side) / float(max_side)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                analyze_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        try:
            faces = self._face.analyze(analyze_frame)
        except Exception:
            self._fallback_face_backend()
            try:
                faces = self._face.analyze(analyze_frame)
            except Exception as e2:
                print(f"[face] analyze failed even in fallback: {e2}")
                return []

        if scale < 0.999 and faces:
            inv = 1.0 / scale
            h0, w0 = frame.shape[:2]
            for f in faces:
                bb = f.get("bbox")
                if not isinstance(bb, (tuple, list)) or len(bb) != 4:
                    continue
                x1, y1, x2, y2 = bb
                sx1 = max(0, min(w0, int(round(float(x1) * inv))))
                sy1 = max(0, min(h0, int(round(float(y1) * inv))))
                sx2 = max(0, min(w0, int(round(float(x2) * inv))))
                sy2 = max(0, min(h0, int(round(float(y2) * inv))))
                f["bbox"] = (sx1, sy1, sx2, sy2)
        return faces

    # --- Matching ---

    def _player_thumb_descriptor(self, player: Dict[str, Any]) -> Optional[np.ndarray]:
        pid = int(player.get("id", -1))
        thumb_rel = player.get("thumb")
        if not isinstance(thumb_rel, str) or pid <= 0:
            return None

        path = os.path.join("storage", thumb_rel)
        try:
            stamp = float(os.path.getmtime(path))
        except OSError:
            return None

        if pid in self._thumb_desc_cache and self._thumb_desc_stamp.get(pid) == stamp:
            return self._thumb_desc_cache[pid]

        img = cv2.imread(path)
        if img is None or img.size == 0:
            return None

        desc = _face_appearance_descriptor(img)
        self._thumb_desc_cache[pid] = desc
        self._thumb_desc_stamp[pid] = stamp
        return desc

    @staticmethod
    def _encode_crop_jpeg(crop: np.ndarray) -> bytes:
        if crop is None or crop.size == 0:
            return b""
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return buf.tobytes() if ok else b""

    def _subject_from_player(self, player: Dict[str, Any]) -> Optional[str]:
        pid = int(player.get("id", -1))
        if pid <= 0:
            return None
        subject = player.get("face_subject")
        if isinstance(subject, str) and subject.strip():
            return subject.strip()
        return self._compreface.subject_for_player(pid)

    def _player_from_subject(
        self,
        subject: str,
        by_subject: Dict[str, Dict[str, Any]],
        by_id: Dict[int, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not subject:
            return None
        direct = by_subject.get(subject)
        if direct is not None:
            return direct
        prefix = str(getattr(self._compreface, "subject_prefix", "") or "")
        if prefix and subject.startswith(prefix):
            tail = subject[len(prefix):].strip()
            if tail.isdigit():
                return by_id.get(int(tail))
        return None

    @staticmethod
    def _merge_candidates(
        primary: List[Tuple[str, float]],
        secondary: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        merged: Dict[str, float] = {}
        for subject, sim in primary + secondary:
            if not subject:
                continue
            prev = merged.get(subject)
            if prev is None or sim > prev:
                merged[subject] = float(sim)
        return sorted(merged.items(), key=lambda item: item[1], reverse=True)

    def _match_faces_compreface(self, frame: np.ndarray, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        reg = P.list_players()
        if not faces:
            return []
        if not reg:
            return [{"bbox": f["bbox"], "id": None, "name": None, "sim": 0.0} for f in faces]

        by_subject: Dict[str, Dict[str, Any]] = {}
        by_id: Dict[int, Dict[str, Any]] = {}
        for p in reg:
            pid = _safe_int(p.get("id"), 0)
            if pid > 0:
                by_id[pid] = p
            subject = self._subject_from_player(p)
            if subject:
                by_subject[subject] = p

        now = time.time()
        self._compreface_face_cache = [
            row for row in self._compreface_face_cache
            if now - float(row.get("last_seen", 0.0)) <= self._compreface_cache_ttl_sec
        ]

        indexed_faces: List[Tuple[int, Dict[str, Any]]] = list(enumerate(faces))
        indexed_faces.sort(
            key=lambda row: max(1, (row[1]["bbox"][2] - row[1]["bbox"][0]) * (row[1]["bbox"][3] - row[1]["bbox"][1])),
            reverse=True,
        )

        out_by_idx: Dict[int, Dict[str, Any]] = {}
        used_pids: set[int] = set()
        used_cache_indices: set[int] = set()
        margin_high = float(os.getenv("COMPREFACE_SECOND_BEST_MARGIN", "0.055"))
        margin_low = float(os.getenv("COMPREFACE_SECOND_BEST_MARGIN_LOW_CONF", "0.035"))
        recognize_predictions = max(2, _safe_int(os.getenv("COMPREFACE_RECOGNIZE_PREDICTIONS", "3"), default=3))
        recognize_det_prob = float(
            os.getenv("COMPREFACE_RECOGNIZE_DET_PROB", str(self._compreface.det_prob_threshold))
        )

        for idx, f in indexed_faces:
            x1, y1, x2, y2 = f["bbox"]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            diag = float((bw * bw + bh * bh) ** 0.5)
            cx = int((x1 + x2) * 0.5)
            cy = int((y1 + y2) * 0.5)

            cache_row: Optional[Dict[str, Any]] = None
            cache_index: Optional[int] = None
            best_dist2: Optional[float] = None
            max_dist = max(42.0, diag * 0.55)
            max_dist2 = max_dist * max_dist
            for row_idx, row in enumerate(self._compreface_face_cache):
                if row_idx in used_cache_indices:
                    continue
                dx = float(cx - _safe_int(row.get("cx"), 0))
                dy = float(cy - _safe_int(row.get("cy"), 0))
                dist2 = dx * dx + dy * dy
                if dist2 > max_dist2:
                    continue
                if best_dist2 is None or dist2 < best_dist2:
                    best_dist2 = dist2
                    cache_row = row
                    cache_index = row_idx

            if cache_row is not None:
                cache_row["cx"] = cx
                cache_row["cy"] = cy
                cache_row["last_seen"] = now
                cached_pid = _safe_int(cache_row.get("pid"), 0)
                cached_pid_opt = cached_pid if cached_pid > 0 else None
                if (
                    now - float(cache_row.get("recognized_at", 0.0)) < self._compreface_match_interval_sec
                    and (cached_pid_opt is None or cached_pid_opt not in used_pids)
                ):
                    if cached_pid_opt is not None:
                        used_pids.add(cached_pid_opt)
                    if cache_index is not None:
                        used_cache_indices.add(cache_index)
                    out_by_idx[idx] = {
                        "bbox": f["bbox"],
                        "id": cached_pid_opt,
                        "name": cache_row.get("name"),
                        "sim": float(cache_row.get("sim", 0.0)),
                    }
                    continue

            if min(bw, bh) < self._compreface_min_face_size:
                if cache_row is not None:
                    cached_pid = _safe_int(cache_row.get("pid"), 0)
                    cached_pid_opt = cached_pid if cached_pid > 0 and cached_pid not in used_pids else None
                    if cached_pid_opt is not None:
                        used_pids.add(cached_pid_opt)
                    out_by_idx[idx] = {
                        "bbox": f["bbox"],
                        "id": cached_pid_opt,
                        "name": cache_row.get("name") if cached_pid_opt is not None else None,
                        "sim": float(cache_row.get("sim", 0.0)),
                    }
                else:
                    out_by_idx[idx] = {"bbox": f["bbox"], "id": None, "name": None, "sim": 0.0}
                continue

            adaptive_threshold = float(self._compreface.similarity_threshold)
            min_face_side = min(bw, bh)
            if min_face_side < 100:
                adaptive_threshold -= 0.06
            elif min_face_side < 140:
                adaptive_threshold -= 0.03
            adaptive_threshold = max(0.48, adaptive_threshold)

            xx1, yy1, xx2, yy2 = _expand_bbox((x1, y1, x2, y2), frame.shape, pad_ratio_x=0.28, pad_ratio_y=0.36)
            crop = frame[yy1:yy2, xx1:xx2]
            jpg = self._encode_crop_jpeg(crop)
            candidates_primary = self._compreface.recognize_candidates(
                jpg,
                det_prob_threshold=recognize_det_prob,
                prediction_count=recognize_predictions,
            )

            # Wide crop is expensive (2nd network request); use it only when primary match is weak/empty.
            top_primary_sim = float(candidates_primary[0][1]) if candidates_primary else 0.0
            should_try_wide = (
                not candidates_primary
                or top_primary_sim < (adaptive_threshold + 0.02)
                or min_face_side < 120
            )
            candidates_wide: List[Tuple[str, float]] = []
            if should_try_wide:
                xx1b, yy1b, xx2b, yy2b = _expand_bbox((x1, y1, x2, y2), frame.shape, pad_ratio_x=0.40, pad_ratio_y=0.52)
                crop2 = frame[yy1b:yy2b, xx1b:xx2b]
                jpg2 = self._encode_crop_jpeg(crop2)
                candidates_wide = self._compreface.recognize_candidates(
                    jpg2,
                    det_prob_threshold=recognize_det_prob,
                    prediction_count=recognize_predictions,
                )
            candidates = self._merge_candidates(candidates_primary, candidates_wide)

            pid: Optional[int] = None
            pname: Optional[str] = None
            simv = 0.0

            mapped: List[Tuple[int, str, float]] = []
            for subject, similarity in candidates:
                player = self._player_from_subject(subject, by_subject, by_id)
                if player is None:
                    continue
                cand_pid = _safe_int(player.get("id"), 0)
                if cand_pid <= 0 or cand_pid in used_pids:
                    continue
                cand_name = str(player.get("name") or f"Player {cand_pid}")
                mapped.append((cand_pid, cand_name, float(similarity)))

            if mapped:
                top_pid, top_name, top_sim = mapped[0]
                second_sim = mapped[1][2] if len(mapped) > 1 else 0.0
                margin = margin_high if top_sim >= (adaptive_threshold + 0.08) else margin_low
                if top_sim >= adaptive_threshold and (len(mapped) == 1 or (top_sim - second_sim) >= margin):
                    pid = top_pid
                    pname = top_name
                    simv = top_sim
                    used_pids.add(pid)
                else:
                    simv = top_sim
            elif candidates:
                simv = float(candidates[0][1])

            if cache_row is None:
                cache_row = {}
                self._compreface_face_cache.append(cache_row)
                cache_index = len(self._compreface_face_cache) - 1
            cache_row.update(
                {
                    "cx": cx,
                    "cy": cy,
                    "pid": pid,
                    "name": pname,
                    "sim": float(simv),
                    "last_seen": now,
                    "recognized_at": now,
                }
            )
            if cache_index is not None:
                used_cache_indices.add(cache_index)
            out_by_idx[idx] = {"bbox": f["bbox"], "id": pid, "name": pname, "sim": float(simv)}

        out: List[Dict[str, Any]] = []
        for i, face in enumerate(faces):
            out.append(out_by_idx.get(i, {"bbox": face["bbox"], "id": None, "name": None, "sim": 0.0}))
        return out

    def _match_faces_legacy(self, frame: np.ndarray, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        reg = P.list_players()
        out: List[Dict[str, Any]] = []
        if not faces:
            return out
        if not reg:
            return [{"bbox": f["bbox"], "id": None, "name": None, "sim": 0.0} for f in faces]

        buckets: Dict[int, Dict[str, Any]] = {}
        id_to_name: Dict[int, str] = {}  # Map player ID to name
        thumb_desc_by_id: Dict[int, np.ndarray] = {}
        for p in reg:
            emb = np.array(p["embedding"], dtype=np.float32)
            d = int(emb.shape[0])
            if d not in buckets:
                buckets[d] = {"ids": [], "embs": []}
            buckets[d]["ids"].append(p["id"])
            buckets[d]["embs"].append(emb)
            id_to_name[p["id"]] = p.get("name", f"Player {p['id']}")  # Store name
            td = self._player_thumb_descriptor(p)
            if td is not None:
                thumb_desc_by_id[int(p["id"])] = td
        for d in list(buckets.keys()):
            E = np.stack(buckets[d]["embs"], axis=0)
            buckets[d]["norm"] = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)

        # Минимальный отрыв лучшего кандидата от второго (при близких по схожести игроках).
        # Слишком большой отрыв убивает распознавание, когда в базе >1 игрок.
        confidence_margin = float(os.getenv("FACE_MATCH_MARGIN", "0.03"))

        for f in faces:
            emb = f["embedding"].astype(np.float32)
            d = int(emb.shape[0])
            pid, pname, simv = None, None, 0.0
            active_threshold = self._face.sim_threshold
            active_margin = confidence_margin
            if d in buckets:
                embn = emb / (np.linalg.norm(emb) + 1e-6)
                sims = buckets[d]["norm"] @ embn

                # LANDMARKS эмбеддинг (936) слаб для multi-person.
                # Усиливаем матч по внешнему виду лица с миниатюрами из БД.
                if d == 936:
                    face_app = f.get("appearance")
                    if face_app is None:
                        x1, y1, x2, y2 = f["bbox"]
                        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        face_app = _face_appearance_descriptor(crop)

                    ids = buckets[d]["ids"]
                    app_sims = np.array(
                        [
                            float(face_app @ thumb_desc_by_id[i]) if i in thumb_desc_by_id else np.nan
                            for i in ids
                        ],
                        dtype=np.float32,
                    )
                    valid = ~np.isnan(app_sims)
                    if np.any(valid):
                        combined = sims.copy()
                        combined[valid] = 0.85 * app_sims[valid] + 0.15 * sims[valid]
                        sims = combined
                        active_threshold = float(os.getenv("FACE_LANDMARKS_SIM_THRESHOLD", "0.62"))
                        active_margin = float(os.getenv("FACE_LANDMARKS_MATCH_MARGIN", "0.015"))

                # Find best match with confidence check
                sorted_indices = np.argsort(sims)[::-1]  # Sort descending
                best_sim = float(sims[sorted_indices[0]])

                # Check if match is confident (significantly better than 2nd best)
                is_confident = True
                if len(sorted_indices) > 1:
                    second_best_sim = float(sims[sorted_indices[1]])
                    margin = best_sim - second_best_sim
                    # Жесткий margin применяем только в пограничных случаях.
                    if best_sim < (active_threshold + 0.10):
                        is_confident = margin >= active_margin

                simv = best_sim
                # Only assign ID if similarity is above threshold AND match is confident
                if simv >= active_threshold and is_confident:
                    pid = buckets[d]["ids"][sorted_indices[0]]
                    pname = id_to_name.get(pid)  # Get name from map
            out.append({"bbox": f["bbox"], "id": pid, "name": pname, "sim": simv})
        return out

    def _match_faces(self, frame: np.ndarray, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self._compreface.is_active():
            try:
                return self._match_faces_compreface(frame, faces)
            except Exception as e:
                print(f"[face] CompreFace match failed, fallback to legacy matcher: {e}")
        return self._match_faces_legacy(frame, faces)

    # --- Overlay ---

    def _draw_table_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Только линия/полигон стола без рук и лиц."""
        h, w = frame.shape[:2]
        out = frame.copy()
        table_y = int(self.table_y_ratio * h)
        cv2.line(out, (0, table_y), (w, table_y), (110, 110, 110), 1)
        poly_px = self._poly_px(w, h)
        if poly_px is not None:
            cv2.polylines(out, [poly_px.astype(np.int32)], True, (255, 210, 70), 2, cv2.LINE_AA)
            overlay = out.copy()
            cv2.fillPoly(overlay, [poly_px.astype(np.int32)], (50, 190, 255))
            out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)
        return out

    def _draw_overlay(self, frame: np.ndarray, payload: Dict[str, Any], face_matches: List[Dict[str, Any]]) -> np.ndarray:
        h, w = frame.shape[:2]
        out = frame.copy()

        table_y = int(self.table_y_ratio * h)
        cv2.line(out, (0, table_y), (w, table_y), (110, 110, 110), 1)

        for hand in payload.get("hands", []):
            x, y, ww, hh = hand["bbox"]
            cx, cy = hand["center"]
            cv2.rectangle(out, (x, y), (x + ww, y + hh), (0, 255, 170), 2)
            cv2.circle(out, (cx, cy), 5, (0, 255, 255), -1)
            lab = hand.get("label", "")
            if lab:
                cv2.putText(out, lab, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        poly_px = self._poly_px(w, h)
        if poly_px is not None:
            cv2.polylines(out, [poly_px.astype(np.int32)], True, (255, 210, 70), 2, cv2.LINE_AA)
            overlay = out.copy()
            cv2.fillPoly(overlay, [poly_px.astype(np.int32)], (50, 190, 255))
            out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)

        for m in face_matches:
            x1, y1, x2, y2 = m["bbox"]
            pid = m["id"]
            pname = m.get("name")

            # Анимация для распознанных лиц
            if pid:
                if self._face_overlay_light:
                    # Lighter overlay for lower CPU usage on weaker devices.
                    color = (255, 184, 103)
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                else:
                    # Плавная пульсация (30 frames цикл)
                    pulse = np.sin(self._frame_counter * 0.1) * 0.3 + 0.7  # от 0.4 до 1.0
                    color_r = int(103 * pulse)
                    color_g = int(184 * pulse)
                    color_b = int(255 * pulse)
                    color = (color_b, color_g, color_r)
                    thickness = 3 if pulse > 0.85 else 2

                    # Рисуем прямоугольник с анимацией
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

                    # Добавляем угловые акценты
                    corner_len = 20
                    corner_color = (color_b, color_g, color_r)
                    # Верхний левый
                    cv2.line(out, (x1, y1), (x1 + corner_len, y1), corner_color, 4)
                    cv2.line(out, (x1, y1), (x1, y1 + corner_len), corner_color, 4)
                    # Верхний правый
                    cv2.line(out, (x2, y1), (x2 - corner_len, y1), corner_color, 4)
                    cv2.line(out, (x2, y1), (x2, y1 + corner_len), corner_color, 4)
                    # Нижний левый
                    cv2.line(out, (x1, y2), (x1 + corner_len, y2), corner_color, 4)
                    cv2.line(out, (x1, y2), (x1, y2 - corner_len), corner_color, 4)
                    # Нижний правый
                    cv2.line(out, (x2, y2), (x2 - corner_len, y2), corner_color, 4)
                    cv2.line(out, (x2, y2), (x2, y2 - corner_len), corner_color, 4)
            else:
                # Простой серый прямоугольник для нераспознанных лиц
                color = (120, 120, 120)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = pname if pname else "?"
            cv2.putText(out, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Don't show hands/faces count during enrollment
        # text = f"hands:{len(payload.get('hands', []))} faces:{len(face_matches)}"
        # cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    # --- JPEG ---

    async def _encode_jpeg(self, frame: np.ndarray) -> bytes:
        # Higher quality for better visual accuracy (75 for good balance)
        ok, buf = await asyncio.to_thread(cv2.imencode, ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return buf.tobytes() if ok else b""

    # --- Main loop ---

    async def _run(self):
        period = 1.0 / self.fps
        last_evt = 0.0
        black_count = 0

        while self._running:
            ok, frame = await asyncio.to_thread(self._cap.read)
            if not ok or frame is None:
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    break
                continue

            if frame.size == 0 or float(frame.mean()) < 0.5:
                black_count += 1
                if black_count > 20:
                    print("[camera] too many black frames; attempting soft reopen…")
                    try: self._cap.release()
                    except Exception: pass
                    await asyncio.sleep(0.2)
                    self._cap = _open_capture(self.camera_index)
                    black_count = 0
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    break
                continue
            else:
                black_count = 0

            try:
                # сохранить последний кадр
                async with self._frame_lock:
                    self._last_frame = frame

                # RAW JPEG всегда держим актуальным
                raw_jpeg = await self._encode_jpeg(frame)
                async with self._jpeg_lock:
                    self._last_raw_jpeg = raw_jpeg

                # Выбор ветки рендера
                if not self.detect_enabled or self.render_mode == RENDER_RAW:
                    # чистое видео, без событий
                    jpeg = raw_jpeg

                elif self.render_mode == RENDER_TABLE:
                    # только стол (без вычисления лиц/жестов!)
                    table_only = self._draw_table_overlay(frame)
                    jpeg = await self._encode_jpeg(table_only)

                else:  # RENDER_FULL
                    # лица всегда считаем; жесты можно отключить для registration flow
                    hands_out: List[Dict[str, Any]] = []
                    fist_on_table = False
                    digit = -1
                    res = None
                    if self.gestures_enabled:
                        res = await asyncio.to_thread(self._det.process_frame, frame)
                        digit = _safe_int(getattr(res, "digit", None), default=-1)

                    need_face_refresh = (
                        not self._last_faces
                        or not self._last_matches
                        or self._face_analyze_every <= 1
                        or (self._frame_counter % self._face_analyze_every == 0)
                    )
                    if need_face_refresh:
                        faces = await asyncio.to_thread(self._safe_face_analyze, frame)
                        if self.face_match_enabled:
                            matches = await asyncio.to_thread(self._match_faces, frame, faces)
                        else:
                            matches = [{"bbox": f["bbox"], "id": None, "name": None, "sim": 0.0} for f in faces]
                        self._last_faces = faces
                        self._last_matches = matches
                    else:
                        faces = self._last_faces
                        matches = self._last_matches

                    if self.gestures_enabled and res is not None:
                        h, w = frame.shape[:2]
                        poly_px = self._poly_px(w, h)
                        fist_on_table = res.fist_on_table
                        if poly_px is not None:
                            fist_on_table = any(
                                (_safe_int(getattr(hnd, "count", None), default=0) == 0 and _point_in_poly(hnd.center, poly_px))
                                for hnd in res.hands
                            )

                        def center_face(bb):
                            x1, y1, x2, y2 = bb
                            return ((x1 + x2) // 2, (y1 + y2) // 2)

                        face_centers = [(m["id"], center_face(m["bbox"])) for m in matches]
                        face_bbox_by_id: Dict[int, Tuple[int, int, int, int]] = {}
                        for m in matches:
                            mid = m.get("id")
                            bb = m.get("bbox")
                            if isinstance(mid, int) and mid > 0 and isinstance(bb, (tuple, list)) and len(bb) == 4:
                                face_bbox_by_id[mid] = tuple(bb)

                        def _label_for_hand(h) -> Tuple[str, int]:
                            cnt_raw = getattr(h, "count", None)
                            if isinstance(cnt_raw, (int, float)):
                                cnt = _safe_int(cnt_raw, default=0)
                            else:
                                ext = getattr(h, "extended", None)
                                if isinstance(ext, dict):
                                    cnt = int(sum(1 for v in ext.values() if bool(v)))
                                elif isinstance(ext, (list, tuple)):
                                    cnt = int(sum(1 for v in ext if bool(v)))
                                else:
                                    cnt = 0

                            if cnt < 0:
                                cnt = 0
                            if cnt > 10:
                                cnt = 10

                            gesture = str(getattr(h, "gesture", "") or "").strip().lower()
                            if gesture:
                                return gesture, cnt

                            fallback = {0: "fist", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}.get(cnt, "unknown")
                            return fallback, cnt

                        def _contextual_label(base_label: str, owner_id: Optional[int], center: Tuple[int, int]) -> str:
                            if not isinstance(owner_id, int) or owner_id <= 0:
                                return base_label
                            bb = face_bbox_by_id.get(owner_id)
                            if bb is None:
                                return base_label
                            x1, y1, x2, y2 = bb
                            fw = max(1.0, float(x2 - x1))
                            fh = max(1.0, float(y2 - y1))
                            cx, cy = center

                            near_head = (
                                (x1 - 0.40 * fw) <= cx <= (x2 + 0.40 * fw)
                                and (y1 - 0.45 * fh) <= cy <= (y1 + 0.25 * fh)
                            )
                            near_chest = (
                                (x1 - 0.65 * fw) <= cx <= (x2 + 0.65 * fw)
                                and (y2 - 0.05 * fh) <= cy <= (y2 + 1.30 * fh)
                            )

                            if base_label == "1":
                                if near_head:
                                    return "think"
                                if near_chest:
                                    return "self"
                            if base_label == "fist" and near_head:
                                return "think"
                            return base_label

                        for hnd in res.hands:
                            owner = None
                            if face_centers:
                                cx, cy = hnd.center
                                dists = [((cx - fc[1][0]) ** 2 + (cy - fc[1][1]) ** 2, fc[0]) for fc in face_centers]
                                dists.sort(key=lambda t: t[0])
                                owner = dists[0][1]
                            label, fingers = _label_for_hand(hnd)
                            label = _contextual_label(label, owner, hnd.center)
                            hands_out.append({
                                "bbox": hnd.bbox,
                                "center": hnd.center,
                                "count": _safe_int(getattr(hnd, "count", None), default=0),
                                "extended": hnd.extended,
                                "handedness": getattr(hnd, "handedness", ""),
                                "track_id": getattr(hnd, "track_id", None),
                                "owner_id": owner,
                                "label": label,
                                "gesture": label,
                                "fingers": fingers,
                            })

                        # If two hands belong to the same recognized person and both are numeric
                        # gestures, show the same total on both hands (e.g., 4 + 3 => "7" on each).
                        hands_by_owner: Dict[int, List[int]] = {}
                        for idx, hand in enumerate(hands_out):
                            owner_id = hand.get("owner_id")
                            if isinstance(owner_id, int) and owner_id > 0:
                                hands_by_owner.setdefault(owner_id, []).append(idx)

                        for owner_id, idxs in hands_by_owner.items():
                            if len(idxs) != 2:
                                continue
                            vals: List[int] = []
                            valid_pair = True
                            for idx in idxs:
                                raw_gesture = str(hands_out[idx].get("gesture", "")).strip()
                                if raw_gesture.isdigit():
                                    v = int(raw_gesture)
                                else:
                                    valid_pair = False
                                    break
                                if v < 0 or v > 5:
                                    valid_pair = False
                                    break
                                vals.append(v)
                            if not valid_pair:
                                continue

                            total = int(sum(vals))
                            total_label = str(total)
                            for idx in idxs:
                                hands_out[idx]["label"] = total_label
                                hands_out[idx]["gesture"] = total_label
                                hands_out[idx]["fingers_total"] = total
                                hands_out[idx]["fingers_total_owner_id"] = owner_id

                    now = time.time()
                    if now - last_evt >= 0.2:
                        last_evt = now
                        payload = {
                            "type": "gesture",
                            "digit": digit,
                            "fist_on_table": bool(fist_on_table),
                            "hands": hands_out,
                            "faces": [{"bbox": m["bbox"], "id": m["id"], "sim": m["sim"]} for m in matches],
                            "gestures_enabled": self.gestures_enabled,
                        }
                        try:
                            await self.on_event(payload)
                        except Exception:
                            pass

                    overlay = self._draw_overlay(
                        frame,
                        {"hands": hands_out, "fist_on_table": fist_on_table, "digit": digit},
                        matches,
                    )
                    jpeg = await self._encode_jpeg(overlay)

                async with self._jpeg_lock:
                    self._last_jpeg = jpeg

            except Exception as e:
                print(f"[stream] iteration error: {e}")
                try:
                    jpeg = await self._encode_jpeg(frame)
                    async with self._jpeg_lock:
                        self._last_jpeg = jpeg
                except Exception:
                    pass

            # Increment frame counter for animation
            self._frame_counter = (self._frame_counter + 1) % 1000  # Reset every 1000 frames

            try:
                await asyncio.sleep(period)
            except asyncio.CancelledError:
                break

    async def get_last_jpeg(self, raw: bool = False) -> Optional[bytes]:
        async with self._jpeg_lock:
            return self._last_raw_jpeg if raw else self._last_jpeg
