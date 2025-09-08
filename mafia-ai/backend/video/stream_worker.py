from __future__ import annotations
import asyncio, os, time, sys
from typing import Optional, Callable, Awaitable, Dict, Any, Tuple, List
import cv2
import numpy as np

from .gestures import GestureDetector

EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]

def _open_capture(idx: int) -> cv2.VideoCapture:
    return cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(idx)

def _point_in_poly(px: Tuple[int,int], poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly.astype(np.int32), (int(px[0]), int(px[1])), False) >= 0

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
        self._cap: Optional[cv2.VideoCapture] = None
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False

        self._last_jpeg: Optional[bytes] = None
        self._jpeg_lock = asyncio.Lock()

        self._last_frame: Optional[np.ndarray] = None
        self._frame_lock = asyncio.Lock()

        # произвольный многоугольник стола в норм. координатах (>=3 точки)
        self._table_poly_norm: Optional[List[Tuple[float,float]]] = None

    async def start(self):
        if self._running:
            return
        self._cap = _open_capture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")
        try:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        except Exception:
            pass
        self._running = True
        print(f"[GestureStream] started (camera={self.camera_index}, fps={self.fps})")
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._running = False
        if self._task:
            try:
                await self._task
            finally:
                self._task = None
        if self._cap:
            self._cap.release()
            self._cap = None
        print("[GestureStream] stopped")

    async def _read_frame(self):
        ok, frame = await asyncio.to_thread(self._cap.read)
        return ok, frame

    def _poly_px(self, w:int, h:int) -> Optional[np.ndarray]:
        if not self._table_poly_norm: return None
        pts = np.array([[p[0]*w, p[1]*h] for p in self._table_poly_norm], dtype=np.float32)
        return pts

    def set_table_polygon_norm(self, poly_norm: List[Tuple[float,float]]):
        if len(poly_norm) < 3:
            raise ValueError("table polygon must have at least 3 points (norm coords)")
        self._table_poly_norm = poly_norm

    def clear_table_polygon(self):
        self._table_poly_norm = None

    async def auto_detect_table(self) -> Optional[List[Tuple[float,float]]]:
        async with self._frame_lock:
            frame = None if self._last_frame is None else self._last_frame.copy()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        roi = frame[int(h*0.40):, :]
        roi_y0 = int(h*0.40)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        best = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < (w*h)*0.02:
            return None

        best = best.reshape(-1, 2).astype(np.int32)
        best[:,1] += roi_y0
        hull = cv2.convexHull(best).reshape(-1,2)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.01*peri, True).reshape(-1,2)

        poly_norm = [(float(x)/w, float(y)/h) for (x,y) in approx]
        self._table_poly_norm = poly_norm
        return poly_norm

    def _draw_overlay(self, frame: np.ndarray, payload: Dict[str, Any]) -> np.ndarray:
        h, w = frame.shape[:2]
        out = frame.copy()

        # запасная линия-порог
        table_y = int(self.table_y_ratio * h)
        cv2.line(out, (0, table_y), (w, table_y), (120, 120, 120), 1)

        # сводка
        txt = f"digit={payload.get('digit')} | fist={int(payload.get('fist_on_table', False))} | pistol={int(payload.get('pistol', False))}"
        cv2.putText(out, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # руки
        for hand in payload.get("hands", []):
            x, y, ww, hh = hand["bbox"]
            cx, cy = hand["center"]
            cv2.rectangle(out, (x, y), (x+ww, y+hh), (0, 255, 170), 2)
            cv2.circle(out, (cx, cy), 5, (0, 255, 255), -1)
            ext = hand["extended"]
            ex_str = "".join(k[0].upper() for k,v in ext.items() if v) or "-"
            label = f"{hand['handedness']} cnt={hand['count']} ext={ex_str}"
            cv2.putText(out, label, (x, max(18, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,170), 2, cv2.LINE_AA)

        # полигон стола
        poly_px = self._poly_px(w,h)
        if poly_px is not None:
            cv2.polylines(out, [poly_px.astype(np.int32)], True, (255, 210, 70), 2, cv2.LINE_AA)
            overlay = out.copy()
            cv2.fillPoly(overlay, [poly_px.astype(np.int32)], (50, 190, 255))
            out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)

        return out

    async def _encode_jpeg(self, frame: np.ndarray) -> bytes:
        ok, buf = await asyncio.to_thread(cv2.imencode, ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return buf.tobytes() if ok else b""

    async def _run(self):
        period = 1.0 / self.fps
        last_event = 0.0
        while self._running:
            ok, frame = await self._read_frame()
            if not ok:
                await asyncio.sleep(0.005)
                continue

            async with self._frame_lock:
                self._last_frame = frame

            # инференс жестов (в отдельном потоке)
            res = await asyncio.to_thread(self._det.process_frame, frame)
            h, w = frame.shape[:2]
            poly_px = self._poly_px(w,h)

            # fist_on_table (по многоугольнику, если есть)
            fist_on_table = res.fist_on_table
            if poly_px is not None:
                fist_on_table = any(
                    (hnd.count == 0 and _point_in_poly(hnd.center, poly_px))
                    for hnd in res.hands
                )

            # отправляем события не чаще ~5 Гц
            now = time.time()
            if now - last_event >= 0.2:
                last_event = now
                payload = {
                    "type": "gesture",
                    "digit": res.digit,
                    "fist_on_table": bool(fist_on_table),
                    "pistol": bool(any(
                        (hnd.extended["thumb"] and hnd.extended["index"] and not (hnd.extended["middle"] or hnd.extended["ring"] or hnd.extended["pinky"]))
                        for hnd in res.hands
                    )),
                    "hands": [
                        {
                            "bbox": hnd.bbox,
                            "handedness": hnd.handedness,
                            "extended": hnd.extended,
                            "count": int(hnd.count),
                            "center": hnd.center,
                        }
                        for hnd in res.hands
                    ],
                }
                try:
                    await self.on_event(payload)
                except Exception:
                    pass

            # Рисуем и кодируем КАЖДЫЙ кадр → плавный MJPEG
            overlay = self._draw_overlay(frame, {
                "digit": res.digit,
                "fist_on_table": fist_on_table,
                "pistol": False,
                "hands": [
                    {
                        "bbox": hnd.bbox,
                        "handedness": hnd.handedness,
                        "extended": hnd.extended,
                        "count": int(hnd.count),
                        "center": hnd.center,
                    }
                    for hnd in res.hands
                ],
            })
            jpeg = await self._encode_jpeg(overlay)
            async with self._jpeg_lock:
                self._last_jpeg = jpeg

            await asyncio.sleep(period)

    async def get_last_jpeg(self) -> Optional[bytes]:
        async with self._jpeg_lock:
            return self._last_jpeg
