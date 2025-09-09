# backend/video/stream_worker.py
from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Optional, Callable, Awaitable, Dict, Any, Tuple, List

import cv2
import numpy as np

from video.gestures import GestureDetector
from storage import players as P

EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]


# ---------------- Camera helpers ----------------

def _try_open(index: int, api: Optional[int]) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(index, api) if api is not None else cv2.VideoCapture(index)
    return cap if cap.isOpened() else None


def _open_capture(idx: int) -> cv2.VideoCapture:
    """
    Windows: пробуем MSMF → DSHOW → ANY (или порядок из env OPENCV_BACKEND=MSMF|DSHOW|ANY|AUTO)
    Non-Windows: CAP_ANY
    По флагу FORCE_MJPEG=1 форсим MJPG fourcc.
    """
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
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(os.getenv("CAM_WIDTH", "1280")))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(os.getenv("CAM_HEIGHT", "720")))
                if os.getenv("FORCE_MJPEG", "0") == "1":
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception as e:
                last_exc = e
            return cap
    raise RuntimeError(f"Cannot open camera index {idx}; tried apis={apis}; last={last_exc}")


def _point_in_poly(px: Tuple[int, int], poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly.astype(np.int32), (int(px[0]), int(px[1])), False) >= 0


# ---------------- Face identification backends ----------------

class _FaceBackendBase:
    sim_threshold: float = 0.38

    def analyze(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        raise NotImplementedError


class _FaceBackendONNX(_FaceBackendBase):
    """ArcFace ONNX + MediaPipe FaceDetection для bbox."""
    def __init__(self, sim_threshold: float = 0.38):
        import onnxruntime as ort
        import mediapipe as mp
        from pathlib import Path
        import urllib.request

        self.sim_threshold = sim_threshold
        self.det = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.6
        )

        MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model_path = MODELS_DIR / "arcface.onnx"
        if not self.model_path.exists():
            url = "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-11.onnx"
            print("[arcface] downloading model…")
            urllib.request.urlretrieve(url, self.model_path)

        self.sess = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    @staticmethod
    def _preprocess(face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img / 127.5) - 1.0  # [-1,1]
        img = np.transpose(img, (2, 0, 1))  # C,H,W
        img = np.expand_dims(img, 0).astype(np.float32)  # 1,C,H,W
        return img

    def _embed(self, face_bgr: np.ndarray) -> np.ndarray:
        inp = self._preprocess(face_bgr)
        out = self.sess.run([self.output_name], {self.input_name: inp})[0]  # (1,512)
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
            out.append({
                "bbox": (xx1, yy1, xx2, yy2),
                "score": float(det.score[0] if det.score else 0.0),
                "embedding": emb
            })
        return out


class _FaceBackendLandmarks(_FaceBackendBase):
    """
    Фолбэк: MediaPipe FaceMesh → эмбеддинг как нормализованные 2D-координаты.
    Стабильно и без ONNX, достаточно для 10–12 игроков.
    """
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
                vec.append((lm.x - minx) / bw)
                vec.append((lm.y - miny) / bh)
            emb = np.array(vec, dtype=np.float32)
            emb = emb - emb.mean()
            emb = emb / (np.linalg.norm(emb) + 1e-6)
            out.append({"bbox": (x1, y1, x2, y2), "score": 1.0, "embedding": emb})
        return out


def _make_face_backend_initial() -> _FaceBackendBase:
    use = (os.getenv("FACE_BACKEND") or "AUTO").upper()
    if use == "LANDMARKS":
        print("[face] using LANDMARKS backend")
        return _FaceBackendLandmarks(sim_threshold=float(os.getenv("FACE_SIM_THRESHOLD", "0.85")))
    # AUTO / ONNX
    print("[face] using ONNX backend (auto-fallback enabled)")
    try:
        return _FaceBackendONNX(sim_threshold=float(os.getenv("FACE_SIM_THRESHOLD", "0.38")))
    except Exception as e:
        print(f"[face] ONNX init failed: {e}. Falling back to LANDMARKS.")
        return _FaceBackendLandmarks(sim_threshold=float(os.getenv("FACE_SIM_THRESHOLD", "0.85")))


# ---------------- Main stream ----------------

class GestureStream:
    def __init__(
        self,
        on_event: EventCallback,
        camera_index: int = 0,
        table_y_ratio: float = 0.80,
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
    ):
        self.on_event = on_event
        self.camera_index = camera_index
        self.table_y_ratio = table_y_ratio
        self.fps = max(5, fps)
        self.width = width
        self.height = height

        self._det = GestureDetector(table_y_ratio=table_y_ratio)
        self._face: _FaceBackendBase = _make_face_backend_initial()
        self._face_failed = False  # уже переключались на фолбэк?

        self._cap: Optional[cv2.VideoCapture] = None
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False

        self._last_jpeg: Optional[bytes] = None
        self._jpeg_lock = asyncio.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._frame_lock = asyncio.Lock()

        self._table_poly_norm: Optional[List[Tuple[float, float]]] = None

    async def start(self):
        if self._running:
            return
        self._cap = _open_capture(self.camera_index)

        self._running = True
        print(f"[GestureStream] started (camera={self.camera_index}, fps={self.fps})")

        # Прогрев — подготовим первый JPEG
        warm_ok = False
        for _ in range(30):
            ok, frame = await asyncio.to_thread(self._cap.read)
            if not ok or frame is None:
                await asyncio.sleep(0.02)
                continue
            if frame.size and float(frame.mean()) > 1.0:
                async with self._frame_lock:
                    self._last_frame = frame.copy()
                ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok2:
                    async with self._jpeg_lock:
                        self._last_jpeg = buf.tobytes()
                warm_ok = True
                break
            await asyncio.sleep(0.01)
        if not warm_ok:
            print("[camera] warmup got black frames; try OPENCV_BACKEND=MSMF/DSHOW or FORCE_MJPEG=1")

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

    async def auto_detect_table(self) -> Optional[List[Tuple[float, float]]]:
        async with self._frame_lock:
            frame = None if self._last_frame is None else self._last_frame.copy()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.40):, :]
        roi_y0 = int(h * 0.40)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        best = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best) < (w * h) * 0.02:
            return None
        best = best.reshape(-1, 2).astype(np.int32)
        best[:, 1] += roi_y0
        hull = cv2.convexHull(best).reshape(-1, 2)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.01 * peri, True).reshape(-1, 2)
        poly_norm = [(float(x) / w, float(y) / h) for (x, y) in approx]
        self._table_poly_norm = poly_norm
        return poly_norm

    # ----- Face backend failover -----

    def _fallback_face_backend(self):
        if not self._face_failed and not isinstance(self._face, _FaceBackendLandmarks):
            print("[face] runtime error → switching to LANDMARKS backend")
            self._face = _FaceBackendLandmarks(sim_threshold=float(os.getenv("FACE_SIM_THRESHOLD", "0.85")))
            self._face_failed = True

    def _safe_face_analyze(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        try:
            return self._face.analyze(frame)
        except Exception as e:
            # Любая ошибка в ONNX — переключаемся на фолбэк и пробуем ещё раз
            self._fallback_face_backend()
            try:
                return self._face.analyze(frame)
            except Exception as e2:
                print(f"[face] analyze failed even in fallback: {e2}")
                return []

    # ----- Matching -----

    def _match_faces(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        reg = P.list_players()
        out: List[Dict[str, Any]] = []
        if not faces:
            return out
        if not reg:
            return [{"bbox": f["bbox"], "id": None, "sim": 0.0} for f in faces]

        # Сгруппируем реестр по размерности эмбеддинга
        buckets: Dict[int, Dict[str, Any]] = {}
        for p in reg:
            emb = np.array(p["embedding"], dtype=np.float32)
            d = int(emb.shape[0])
            if d not in buckets:
                buckets[d] = {"ids": [], "embs": []}
            buckets[d]["ids"].append(p["id"])
            buckets[d]["embs"].append(emb)
        for d in list(buckets.keys()):
            E = np.stack(buckets[d]["embs"], axis=0)
            buckets[d]["norm"] = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)

        for f in faces:
            emb = f["embedding"].astype(np.float32)
            d = int(emb.shape[0])
            pid, simv = None, 0.0
            if d in buckets:
                embn = emb / (np.linalg.norm(emb) + 1e-6)
                sims = buckets[d]["norm"] @ embn
                j = int(np.argmax(sims))
                simv = float(sims[j])
                if simv >= self._face.sim_threshold:
                    pid = buckets[d]["ids"][j]
            out.append({"bbox": f["bbox"], "id": pid, "sim": simv})
        return out

    # ----- Overlay -----

    def _draw_overlay(self, frame: np.ndarray, payload: Dict[str, Any], face_matches: List[Dict[str, Any]]) -> np.ndarray:
        h, w = frame.shape[:2]
        out = frame.copy()

        # линия-порог (на случай отсутствия полигона)
        table_y = int(self.table_y_ratio * h)
        cv2.line(out, (0, table_y), (w, table_y), (110, 110, 110), 1)

        # руки
        for hand in payload.get("hands", []):
            x, y, ww, hh = hand["bbox"]
            cx, cy = hand["center"]
            cv2.rectangle(out, (x, y), (x + ww, y + hh), (0, 255, 170), 2)
            cv2.circle(out, (cx, cy), 5, (0, 255, 255), -1)
            cv2.putText(out, hand.get("label",""), (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)


        # стол-полигон
        poly_px = self._poly_px(w, h)
        if poly_px is not None:
            cv2.polylines(out, [poly_px.astype(np.int32)], True, (255, 210, 70), 2, cv2.LINE_AA)
            overlay = out.copy()
            cv2.fillPoly(overlay, [poly_px.astype(np.int32)], (50, 190, 255))
            out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)

        # лица + номера
        for m in face_matches:
            x1, y1, x2, y2 = m["bbox"]
            pid = m["id"]
            color = (103, 184, 255) if pid else (120, 120, 120)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"#{pid}" if pid else "?"
            cv2.putText(out, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # статус (видно, что поток живой)
        text = f"hands:{len(payload.get('hands', []))} faces:{len(face_matches)}"
        cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    # ----- JPEG -----

    async def _encode_jpeg(self, frame: np.ndarray) -> bytes:
        ok, buf = await asyncio.to_thread(cv2.imencode, ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return buf.tobytes() if ok else b""

    # ----- Main loop -----

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

            # защита от чёрных кадров
            if frame.size == 0 or float(frame.mean()) < 0.5:
                black_count += 1
                if black_count > 20:
                    print("[camera] too many black frames; attempting soft reopen…")
                    try:
                        self._cap.release()
                    except Exception:
                        pass
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
                # сохраняем последний кадр
                async with self._frame_lock:
                    self._last_frame = frame

                # жесты + лица (лица — через безопасный вызов с фолбэком)
                res = await asyncio.to_thread(self._det.process_frame, frame)
                faces = await asyncio.to_thread(self._safe_face_analyze, frame)
                matches = self._match_faces(faces)

                # fist_on_table с учётом полигона
                h, w = frame.shape[:2]
                poly_px = self._poly_px(w, h)
                fist_on_table = res.fist_on_table
                if poly_px is not None:
                    fist_on_table = any(
                        (hnd.count == 0 and _point_in_poly(hnd.center, poly_px)) for hnd in res.hands
                    )

                # Привязка руки к ближайшему лицу
                                # привязка руки к ближайшему лицу + классификация жеста
                def center_face(bb):
                    x1, y1, x2, y2 = bb
                    return ((x1 + x2) // 2, (y1 + y2) // 2)

                face_centers = [(m["id"], center_face(m["bbox"])) for m in matches]

                def _label_for_hand(h) -> tuple[str, int]:
                    # ориентируемся на число выпрямленных пальцев
                    if hasattr(h, "extended") and h.extended is not None:
                        cnt = int(sum(1 for v in h.extended if v))
                    else:
                        cnt = int(getattr(h, "count", 0))
                    # простая словарная классификация
                    name = {
                        0: "fist",
                        1: "one",
                        2: "two",
                        3: "three",
                        4: "four",
                        5: "open",
                    }.get(cnt, f"{cnt}-fingers")
                    return name, cnt

                hands_out = []
                for hnd in res.hands:
                    owner = None
                    if face_centers:
                        cx, cy = hnd.center
                        dists = [((cx - fc[1][0]) ** 2 + (cy - fc[1][1]) ** 2, fc[0]) for fc in face_centers]
                        dists.sort(key=lambda t: t[0])
                        owner = dists[0][1]  # id или None
                    label, fingers = _label_for_hand(hnd)
                    hands_out.append(
                        {
                            "bbox": hnd.bbox,
                            "center": hnd.center,
                            "count": int(hnd.count),
                            "extended": hnd.extended,
                            "owner_id": owner,
                            "label": label,
                            "fingers": fingers,
                        }
                    )


                # WS (5 Гц)
                now = time.time()
                if now - last_evt >= 0.2:
                    last_evt = now
                    payload = {
                        "type": "gesture",
                        "digit": res.digit,
                        "fist_on_table": bool(fist_on_table),
                        "hands": hands_out,
                        "faces": [{"bbox": m["bbox"], "id": m["id"], "sim": m["sim"]} for m in matches],
                    }
                    try:
                        await self.on_event(payload)
                    except Exception:
                        pass

                # Рендер → JPEG
                overlay = self._draw_overlay(
                    frame,
                    {"hands": hands_out, "fist_on_table": fist_on_table, "digit": res.digit},
                    matches,
                )
                jpeg = await self._encode_jpeg(overlay)
                async with self._jpeg_lock:
                    self._last_jpeg = jpeg

            except Exception as e:
                # не валим цикл: отдадим raw-кадр и продолжим
                print(f"[stream] iteration error: {e}")
                try:
                    jpeg = await self._encode_jpeg(frame)
                    async with self._jpeg_lock:
                        self._last_jpeg = jpeg
                except Exception:
                    pass

            try:
                await asyncio.sleep(period)
            except asyncio.CancelledError:
                break

    async def get_last_jpeg(self) -> Optional[bytes]:
        async with self._jpeg_lock:
            return self._last_jpeg
