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

# ---------- Render modes ----------
RENDER_RAW   = 0  # чистое видео
RENDER_TABLE = 1  # только линия/полигон стола
RENDER_FULL  = 2  # лица + руки + стол

# ---------------- Camera helpers ----------------

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
    sim_threshold: float = 0.40  # Moderate threshold - tracking provides stability
    def analyze(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        raise NotImplementedError


class _FaceBackendONNX(_FaceBackendBase):
    """ArcFace ONNX + MediaPipe FaceDetection для bbox."""
    def __init__(self, sim_threshold: float = 0.40):  # Moderate threshold - tracking provides stability
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
            out.append({"bbox": (x1, y1, x2, y2), "score": 1.0, "embedding": emb})
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
        print(f"[face] ONNX init failed: {e}. Falling back to LANDMARKS.")
        return _FaceBackendLandmarks(sim_threshold=float(os.getenv("FACE_SIM_THRESHOLD", "0.85")))


# ---------------- Face Tracking ----------------

class FaceTrack:
    """Represents a tracked face across multiple frames"""
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], embedding: np.ndarray):
        self.track_id = track_id
        self.bbox = bbox
        self.embedding = embedding
        self.player_id: Optional[int] = None
        self.player_name: Optional[str] = None
        self.age = 0  # frames since last detection
        self.match_history: List[Tuple[Optional[int], float]] = []  # (player_id, similarity)
        self.stable_frames = 0  # consecutive frames with same player_id

    def update(self, bbox: Tuple[int, int, int, int], embedding: np.ndarray):
        """Update track with new detection"""
        self.bbox = bbox
        self.embedding = embedding
        self.age = 0


class FaceTracker:
    """Tracks faces across frames for stable identification"""
    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3):
        self.tracks: List[FaceTrack] = []
        self.next_track_id = 0
        self.max_age = max_age  # frames before track is deleted
        self.iou_threshold = iou_threshold

    @staticmethod
    def _iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / max(union, 1.0)

    def update(self, detections: List[Dict[str, Any]]) -> List[FaceTrack]:
        """
        Update tracks with new detections.
        Returns list of active tracks.
        """
        # Age all tracks
        for track in self.tracks:
            track.age += 1

        # Match detections to tracks using IoU
        matched_tracks = set()
        matched_detections = set()

        # Create cost matrix (negative IoU for Hungarian algorithm)
        if self.tracks and detections:
            for det_idx, det in enumerate(detections):
                det_bbox = det["bbox"]
                best_iou = 0.0
                best_track_idx = None

                for track_idx, track in enumerate(self.tracks):
                    if track_idx in matched_tracks:
                        continue

                    iou = self._iou(det_bbox, track.bbox)
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou = iou
                        best_track_idx = track_idx

                # Greedy assignment: match detection to best track
                if best_track_idx is not None:
                    track = self.tracks[best_track_idx]
                    track.update(det_bbox, det["embedding"])
                    matched_tracks.add(best_track_idx)
                    matched_detections.add(det_idx)

        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detections:
                new_track = FaceTrack(self.next_track_id, det["bbox"], det["embedding"])
                self.tracks.append(new_track)
                self.next_track_id += 1

        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.age < self.max_age]

        return self.tracks


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
        self._face_tracker = FaceTracker(max_age=30, iou_threshold=0.3)

        self._cap: Optional[cv2.VideoCapture] = None
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False

        # Режимы
        self.detect_enabled: bool = True
        self.render_mode: int = RENDER_FULL

        # Буферы
        self._last_raw_jpeg: Optional[bytes] = None
        self._last_jpeg: Optional[bytes] = None
        self._jpeg_lock = asyncio.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._frame_lock = asyncio.Lock()

        self._table_poly_norm: Optional[List[Tuple[float, float]]] = None
        self._frame_counter: int = 0  # Для анимации

    # --- Control API ---

    def set_detect_enabled(self, flag: bool):
        self.detect_enabled = bool(flag)

    def set_render_mode(self, mode: int):
        self.render_mode = int(mode)

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
        try:
            return self._face.analyze(frame)
        except Exception:
            self._fallback_face_backend()
            try:
                return self._face.analyze(frame)
            except Exception as e2:
                print(f"[face] analyze failed even in fallback: {e2}")
                return []

    # --- Matching with Tracking ---

    def _match_faces(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Match detected faces to registered players using temporal tracking.
        Returns list of matched faces with stable player assignments.
        """
        reg = P.list_players()

        # Update tracker with new detections
        tracks = self._face_tracker.update(faces)

        if not reg:
            # No registered players - return unidentified faces
            return [{"bbox": t.bbox, "id": None, "name": None, "sim": 0.0} for t in tracks]

        # Prepare player embeddings
        player_ids = []
        player_embeddings = []
        id_to_name: Dict[int, str] = {}

        for p in reg:
            emb = np.array(p["embedding"], dtype=np.float32)
            emb_norm = emb / (np.linalg.norm(emb) + 1e-6)
            player_ids.append(p["id"])
            player_embeddings.append(emb_norm)
            id_to_name[p["id"]] = p.get("name", f"Player {p['id']}")

        player_embeddings_matrix = np.stack(player_embeddings, axis=0)  # shape: (num_players, emb_dim)

        # Match each track to players
        track_to_player: Dict[int, Tuple[Optional[int], float]] = {}  # track_id -> (player_id, similarity)

        for track in tracks:
            # Normalize track embedding
            emb_norm = track.embedding / (np.linalg.norm(track.embedding) + 1e-6)

            # Compute similarities to all players
            sims = player_embeddings_matrix @ emb_norm  # shape: (num_players,)

            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            # Use lower threshold for initial match, rely on tracking for stability
            if best_sim >= 0.35:  # Lower threshold, tracking provides stability
                candidate_player_id = player_ids[best_idx]
                track_to_player[track.track_id] = (candidate_player_id, best_sim)

                # Update track's match history for temporal smoothing
                track.match_history.append((candidate_player_id, best_sim))
                if len(track.match_history) > 10:  # Keep last 10 matches
                    track.match_history = track.match_history[-10:]
            else:
                track_to_player[track.track_id] = (None, best_sim)
                track.match_history.append((None, best_sim))
                if len(track.match_history) > 10:
                    track.match_history = track.match_history[-10:]

        # Ensure one-to-one mapping: each player can only be assigned to one track
        player_to_track: Dict[int, Tuple[int, float]] = {}  # player_id -> (track_id, similarity)

        for track_id, (player_id, sim) in track_to_player.items():
            if player_id is None:
                continue

            # If this player is already assigned to another track, keep the one with higher similarity
            if player_id in player_to_track:
                existing_track_id, existing_sim = player_to_track[player_id]
                if sim > existing_sim:
                    # This track has higher similarity, reassign
                    track_to_player[existing_track_id] = (None, existing_sim)  # Clear previous track
                    player_to_track[player_id] = (track_id, sim)
                else:
                    # Existing track has higher similarity, clear this track
                    track_to_player[track_id] = (None, sim)
            else:
                player_to_track[player_id] = (track_id, sim)

        # Build output with stable assignments using temporal voting
        out: List[Dict[str, Any]] = []
        for track in tracks:
            pid, sim = track_to_player.get(track.track_id, (None, 0.0))

            # Temporal voting: use majority vote from recent history for stability
            if len(track.match_history) >= 3:
                recent_matches = track.match_history[-5:]  # Last 5 frames
                pid_counts: Dict[Optional[int], int] = {}
                for match_pid, _ in recent_matches:
                    pid_counts[match_pid] = pid_counts.get(match_pid, 0) + 1

                # Use majority vote if it's strong enough
                majority_pid = max(pid_counts.items(), key=lambda x: x[1])
                if majority_pid[1] >= 3:  # At least 3 out of last 5 frames
                    pid = majority_pid[0]

            # Update track's stable player assignment
            if pid == track.player_id:
                track.stable_frames += 1
            else:
                track.player_id = pid
                track.player_name = id_to_name.get(pid) if pid else None
                track.stable_frames = 1

            pname = track.player_name

            out.append({"bbox": track.bbox, "id": pid, "name": pname, "sim": sim})

        return out

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
                    # жесты + лица (real-time detection every frame for accuracy)
                    res = await asyncio.to_thread(self._det.process_frame, frame)

                    # Run face detection on EVERY frame for accurate bounding boxes
                    faces = await asyncio.to_thread(self._safe_face_analyze, frame)
                    matches = self._match_faces(faces)

                    h, w = frame.shape[:2]
                    poly_px = self._poly_px(w, h)
                    fist_on_table = res.fist_on_table
                    if poly_px is not None:
                        fist_on_table = any((hnd.count == 0 and _point_in_poly(hnd.center, poly_px)) for hnd in res.hands)

                    def center_face(bb):
                        x1, y1, x2, y2 = bb
                        return ((x1 + x2) // 2, (y1 + y2) // 2)

                    face_centers = [(m["id"], center_face(m["bbox"])) for m in matches]

                    def _label_for_hand(h) -> Tuple[str, int]:
                        if hasattr(h, "extended") and h.extended is not None:
                            cnt = int(sum(1 for v in h.extended if v))
                        else:
                            cnt = int(getattr(h, "count", 0))
                        name = {0: "fist", 1: "one", 2: "two", 3: "three", 4: "four", 5: "open"}.get(cnt, f"{cnt}-fingers")
                        return name, cnt

                    hands_out = []
                    for hnd in res.hands:
                        owner = None
                        if face_centers:
                            cx, cy = hnd.center
                            dists = [((cx - fc[1][0]) ** 2 + (cy - fc[1][1]) ** 2, fc[0]) for fc in face_centers]
                            dists.sort(key=lambda t: t[0])
                            owner = dists[0][1]
                        label, fingers = _label_for_hand(hnd)
                        hands_out.append({
                            "bbox": hnd.bbox,
                            "center": hnd.center,
                            "count": int(hnd.count),
                            "extended": hnd.extended,
                            "owner_id": owner,
                            "label": label,
                            "fingers": fingers,
                        })

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

                    overlay = self._draw_overlay(frame, {"hands": hands_out, "fist_on_table": fist_on_table, "digit": res.digit}, matches)
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
