# backend/app.py
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from fastapi import Body, FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from video.stream_worker import GestureStream
from storage import players as P
from compreface_client import get_compreface_client

# Новая архитектура
from api.routers import game as game_router
from application.services import get_container

# Voice service
from voice.voice_service import get_voice_service

# --------------------------------------------------------------------------------------
# App / CORS / Static
# --------------------------------------------------------------------------------------

os.makedirs("storage", exist_ok=True)

app = FastAPI(title="Mafia AI Backend")

# статика (миниатюры игроков и т.п.) => /static/...
app.mount("/static", StaticFiles(directory="storage"), name="static")

app.add_middleware(
    CORSMiddleware,
    # для локальной разработки фронта
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost",
        "http://127.0.0.1",
        "*",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Подключаем новые роутеры
app.include_router(game_router.router)

# --------------------------------------------------------------------------------------
# Global state
# --------------------------------------------------------------------------------------

clients: Set[WebSocket] = set()
_stream: Optional[GestureStream] = None
_compreface = get_compreface_client()

# сессия энролла (простая dict-структура)
_enroll: Optional[dict] = None
_enroll_lock = asyncio.Lock()

# speech logs state
_speech_logs: List[Dict[str, Any]] = []
_speech_logs_lock = asyncio.Lock()
_speech_logs_counter: int = 0
_speech_asr_model: Optional[Any] = None
_speech_asr_init_error: Optional[str] = None
_speech_asr_lock = asyncio.Lock()

BOUNDARY = "frame"

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

async def ws_broadcast(msg: Dict[str, Any]) -> None:
    """Отправка сообщения всем подключённым WS-клиентам."""
    dead: List[WebSocket] = []
    for ws in list(clients):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for d in dead:
        clients.discard(d)

def _lap_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _face_quality_score(img_bgr: np.ndarray) -> float:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = _lap_var(g)
    mean = float(g.mean())
    b = max(0.0, min(1.0, (blur - 30.0) / 300.0))
    m = max(0.0, min(1.0, (mean - 40.0) / 60.0))
    return 0.7 * b + 0.3 * m

def _pick_largest_face(faces: List[Dict[str, Any]]) -> Dict[str, Any]:
    def area(bb): x1, y1, x2, y2 = bb; return max(0, x2 - x1) * max(0, y2 - y1)
    return max(faces, key=lambda f: area(f["bbox"]))


def _expand_face_bbox(
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int, int],
    pad_ratio_x: float = 0.14,
    pad_ratio_y: float = 0.20,
) -> Tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(bw * pad_ratio_x)
    py = int(bh * pad_ratio_y)
    return max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)


def _compreface_crop_from_face(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Для CompreFace нужен более "свободный" кроп (не слишком tight),
    иначе их детектор часто отвечает no_face_detected.
    """
    x1, y1, x2, y2 = _expand_face_bbox(bbox, frame.shape, pad_ratio_x=0.30, pad_ratio_y=0.40)
    crop = frame[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return crop


def _encode_jpeg_bytes(img_bgr: np.ndarray) -> bytes:
    if img_bgr is None or img_bgr.size == 0:
        return b""
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return buf.tobytes() if ok else b""


def _select_evenly_spaced_samples(samples: List[bytes], max_count: int) -> List[bytes]:
    if max_count <= 0:
        return []
    n = len(samples)
    if n <= max_count:
        return list(samples)
    if max_count == 1:
        return [samples[n // 2]]

    out: List[bytes] = []
    used: Set[int] = set()
    for i in range(max_count):
        idx = int(round(i * (n - 1) / (max_count - 1)))
        if idx in used:
            continue
        used.add(idx)
        out.append(samples[idx])
    if len(out) < max_count:
        for idx, sample in enumerate(samples):
            if idx in used:
                continue
            out.append(sample)
            if len(out) >= max_count:
                break
    return out


def _player_face_subject(pid: int) -> str:
    return _compreface.subject_for_player(pid)

def _embed_diverse(en: np.ndarray, samples: List[np.ndarray], max_sim: float = 0.90) -> bool:
    if not samples:
        return True
    S = np.stack(samples, axis=0)
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-6)
    en = en / (np.linalg.norm(en) + 1e-6)
    sim = float(np.max(S @ en))
    return sim < max_sim

def _angle_from_pts5(pts5: np.ndarray) -> tuple:
    """
    Грубая оценка поворота головы по 5-точкам:
    вернём yaw (влево/вправо) и pitch (вверх/вниз) в градусах примерно.
    pts5: [[x,y], ...] - 5 точек: 0=left_eye, 1=right_eye, 2=nose, 3=mouth_left, 4=mouth_right
    """
    L, R = pts5[0], pts5[1]
    eye_dx = R[0] - L[0]
    eye_dy = R[1] - L[1]
    yaw = np.degrees(np.arctan2(eye_dy, max(1e-6, eye_dx)))  # наклон линии глаз
    # вертикальная: нос относительно середины глаз
    mid = (L + R) / 2.0
    pitch = np.degrees(np.arctan2(mid[1] - pts5[2][1], max(1e-6, abs(eye_dx))))
    return float(yaw), float(pitch)

async def _current_frame_bgr() -> Optional[np.ndarray]:
    """Берём самый свежий кадр: сначала raw-JPEG из стримера (если доступно), затем _last_frame."""
    global _stream
    if not _stream:
        return None
    try:
        raw = await _stream.get_last_jpeg(raw=True)  # type: ignore[call-arg]
        if raw:
            arr = np.frombuffer(raw, dtype=np.uint8)
            frm = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frm is not None:
                return frm
    except Exception:
        pass
    return _stream._last_frame  # noqa: SLF001

def _safe_face_analyze(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Безопасный вызов текущего face-backend с фолбэком на landmarks."""
    global _stream
    if not _stream:
        return []
    try:
        return _stream._face.analyze(frame)  # noqa: SLF001
    except Exception:
        if hasattr(_stream, "_fallback_face_backend"):
            _stream._fallback_face_backend()  # noqa: SLF001
        try:
            return _stream._face.analyze(frame)  # noqa: SLF001
        except Exception:
            return []

def _players_next_id_fallback() -> int:
    """Надёжно получаем следующий id, даже если в storage.players нет next_id()."""
    try:
        if hasattr(P, "next_id"):
            return int(P.next_id())  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        lst = P.list_players()
        mx = max([int(p.get("id", 0)) for p in lst], default=0)
        return mx + 1
    except Exception:
        return int(time.time())

def _players_add_safe(
    embedding: List[float],
    thumb_rel: str,
    name: str,
    face_subject: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Универсальная обёртка над P.add_player с поддержкой разных сигнатур:
    - add_player(embedding=..., thumb=..., name=...)
    - add_player(embedding, thumb, name)
    - add_player(name=..., embedding=..., thumb=...)
    """
    if not hasattr(P, "add_player"):
        raise RuntimeError("storage.players.add_player is missing")
    try:
        kw: Dict[str, Any] = {"embedding": embedding, "thumb_rel": thumb_rel, "name": name}
        if face_subject:
            kw["face_subject"] = face_subject
        return P.add_player(**kw)  # type: ignore[call-arg]
    except TypeError:
        try:
            return P.add_player(embedding, thumb_rel, name)  # type: ignore[misc]
        except TypeError:
            return P.add_player(name=name, embedding=embedding, thumb=thumb_rel)  # type: ignore[call-arg]


def _speech_speaker_label(speaker_id: Optional[int], speaker_name: Optional[str]) -> str:
    """Build a short speaker label for log output."""
    if speaker_id is not None:
        return f"говорящий {speaker_id}"
    if speaker_name and speaker_name.strip():
        return speaker_name.strip()
    return "говорящий ?"


def _speech_line(label: str, text: str) -> str:
    clean_text = (text or "").strip() or "..."
    return f"\"{label}\"(текст): {clean_text};"


async def _get_speech_asr_model() -> tuple[Optional[Any], Optional[str]]:
    """Lazy-init ASR model for speech logs."""
    global _speech_asr_model, _speech_asr_init_error
    if _speech_asr_model is not None:
        return _speech_asr_model, None
    if _speech_asr_init_error is not None:
        return None, _speech_asr_init_error

    async with _speech_asr_lock:
        if _speech_asr_model is not None:
            return _speech_asr_model, None
        if _speech_asr_init_error is not None:
            return None, _speech_asr_init_error

        try:
            from infrastructure.audio.faster_whisper_asr import FasterWhisperASR

            model_size = os.getenv("SPEECH_LOG_ASR_MODEL", os.getenv("ASR_MODEL", "base"))
            language = os.getenv("SPEECH_LOG_ASR_LANGUAGE", os.getenv("ASR_LANGUAGE", "ru"))
            device = os.getenv("SPEECH_LOG_ASR_DEVICE", "cpu")
            compute_type = os.getenv("SPEECH_LOG_ASR_COMPUTE_TYPE", "int8")

            _speech_asr_model = FasterWhisperASR(
                model_size=model_size,
                language=language,
                device=device,
                compute_type=compute_type,
                num_workers=1,
            )
            return _speech_asr_model, None
        except Exception as e:
            _speech_asr_init_error = str(e)
            return None, _speech_asr_init_error


async def _transcribe_speech_audio(audio: np.ndarray, sample_rate: int) -> tuple[str, Optional[str]]:
    """Transcribe speech audio. Returns (text, error)."""
    asr_model, init_error = await _get_speech_asr_model()
    if asr_model is None:
        return "", init_error
    try:
        audio_for_asr = np.asarray(audio, dtype=np.float32).flatten()
        if audio_for_asr.size == 0:
            return "", None
        audio_for_asr = np.nan_to_num(audio_for_asr, copy=False)
        peak = float(np.max(np.abs(audio_for_asr)))
        if peak > 1e-6:
            audio_for_asr = audio_for_asr / peak

        language = os.getenv("SPEECH_LOG_ASR_LANGUAGE", os.getenv("ASR_LANGUAGE", "ru"))
        transcription = await asr_model.transcribe_async(audio_for_asr, sample_rate=sample_rate, language=language)
        return (transcription.text or "").strip(), None
    except Exception as e:
        return "", str(e)


def _voice_best_guess(vs: Any, audio: np.ndarray, sample_rate: int) -> Optional[Tuple[int, str, float]]:
    """
    Soft fallback for speaker ID when strict voice activity / threshold checks reject a chunk.
    Returns the best profile match if confidence is reasonable.
    """
    try:
        profiles_map = getattr(vs, "profiles", None)
        if not isinstance(profiles_map, dict) or not profiles_map:
            return None

        query = vs.extract_features(audio, sample_rate)  # type: ignore[attr-defined]
        if not isinstance(query, np.ndarray) or query.size == 0:
            return None
        if float(np.linalg.norm(query)) <= 0:
            return None

        best_id: Optional[int] = None
        best_name: Optional[str] = None
        best_score = float("-inf")

        for profile in profiles_map.values():
            score = float(vs._profile_similarity(query, profile.embeddings))  # type: ignore[attr-defined]
            if not np.isfinite(score):
                continue
            if score > best_score:
                best_score = score
                best_id = int(profile.player_id)
                best_name = str(profile.player_name)

        min_guess_score = float(os.getenv("VOICE_LOGS_MIN_GUESS_SCORE", "0.35"))
        if best_id is None or best_name is None or best_score < min_guess_score:
            return None
        return best_id, best_name, best_score
    except Exception:
        return None

# --------------------------------------------------------------------------------------
# Health / Status
# --------------------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True, "clients": len(clients), "video_running": _stream is not None}

@app.get("/face/provider/status")
def face_provider_status():
    health = _compreface.health()
    return {
        "provider": "compreface" if _compreface.enabled else "local",
        "enabled": _compreface.enabled,
        "configured": _compreface.is_active(),
        "health": health,
    }

@app.get("/video/status")
def video_status():
    return {
        "running": _stream is not None,
        "gestures_enabled": bool(_stream.gestures_enabled) if _stream else True,
        "face_match_enabled": bool(_stream.face_match_enabled) if _stream else True,
    }

# --------------------------------------------------------------------------------------
# Video: start / stop / mjpeg
# --------------------------------------------------------------------------------------

@app.post("/video/start")
async def video_start(
    camera_index: Optional[int] = None,
    fps: Optional[int] = None,
    table_y_ratio: Optional[float] = None,
):
    """
    POST /video/start?camera_index=0&fps=30&table_y_ratio=0.8
    """
    global _stream
    if _stream:
        return {"ok": True, "status": "already_running"}

    cam = int(os.getenv("CAMERA_INDEX", "0")) if camera_index is None else int(camera_index)
    f = int(os.getenv("GESTURE_FPS", "30")) if fps is None else int(fps)  # 30fps for smooth rendering
    tyr = float(os.getenv("TABLE_Y_RATIO", "0.80")) if table_y_ratio is None else float(table_y_ratio)

    _stream = GestureStream(on_event=ws_broadcast, camera_index=cam, fps=f, table_y_ratio=tyr)
    try:
        await _stream.start()
        print(f"[app] gesture stream started (camera={cam}, fps={f}, table_y_ratio={tyr})")
        return {
            "ok": True,
            "camera_index": cam,
            "fps": f,
            "table_y_ratio": tyr,
            "gestures_enabled": _stream.gestures_enabled,
            "face_match_enabled": _stream.face_match_enabled,
        }
    except Exception as e:
        _stream = None
        print(f"[app] gesture stream failed: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/video/stop")
async def video_stop():
    global _stream
    if not _stream:
        return {"ok": True, "status": "not_running"}
    await _stream.stop()
    _stream = None
    print("[app] gesture stream stopped")
    return {"ok": True, "status": "stopped"}


class _VideoGesturesIn(BaseModel):
    enabled: bool


@app.post("/video/gestures")
async def video_set_gestures(body: _VideoGesturesIn):
    global _stream
    if not _stream:
        return {"ok": False, "error": "video not running"}
    _stream.set_gestures_enabled(body.enabled)
    return {"ok": True, "gestures_enabled": _stream.gestures_enabled}


@app.post("/video/face-match")
async def video_set_face_match(body: _VideoGesturesIn):
    global _stream
    if not _stream:
        return {"ok": False, "error": "video not running"}
    _stream.set_face_match_enabled(body.enabled)
    return {"ok": True, "face_match_enabled": _stream.face_match_enabled}

async def _mjpeg_generator():
    """MJPEG-стрим. При отсутствии новых кадров повторяет последний, до ~60 FPS."""
    try:
        while True:
            if _stream is None:
                await asyncio.sleep(0.2)
                continue
            jpeg = await _stream.get_last_jpeg()
            if jpeg:
                yield (
                    b"--" + BOUNDARY.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" +
                    jpeg + b"\r\n"
                )
            await asyncio.sleep(1 / 60)
    except asyncio.CancelledError:
        return

@app.get("/video/mjpeg")
async def video_mjpeg():
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return StreamingResponse(
        _mjpeg_generator(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers=headers,
    )

# --------------------------------------------------------------------------------------
# Startup / Shutdown
# --------------------------------------------------------------------------------------

@app.on_event("startup")
async def _startup():
    # Инициализируем ServiceContainer
    container = get_container()
    print("[App] ✅ ServiceContainer initialized")

    auto = os.getenv("AUTO_START_GESTURES", "1") == "1"
    if auto:
        await video_start()

@app.on_event("shutdown")
async def _shutdown():
    global _stream
    if _stream:
        await _stream.stop()
        _stream = None

    # Cleanup ServiceContainer
    from application.services import reset_container
    reset_container()
    print("[App] ✅ ServiceContainer cleaned up")

# --------------------------------------------------------------------------------------
# WebSocket
# --------------------------------------------------------------------------------------

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    print(f"[ws] client connected, total={len(clients)}")
    try:
        while True:
            data = await ws.receive_json()
            t = data.get("type")
            if t == "timer.start":
                seat = int(data.get("seat", 1))
                ms = int(data.get("ms", 60_000))
                asyncio.create_task(run_timer(seat, ms))
            elif t == "ping":
                await ws.send_json({"type": "pong"})
    except Exception:
        pass
    finally:
        clients.discard(ws)
        print(f"[ws] client disconnected, total={len(clients)}")

async def run_timer(seat: int, ms: int):
    end = time.monotonic() + ms / 1000.0
    while True:
        left = max(0.0, end - time.monotonic())
        await ws_broadcast({"type": "timer.tick", "seat": seat, "msLeft": int(left * 1000)})
        if left <= 0.0:
            break
        await asyncio.sleep(0.1)
    await ws_broadcast({"type": "timer.end", "seat": seat})

# --------------------------------------------------------------------------------------
# Table / ROI
# --------------------------------------------------------------------------------------

@app.get("/table/status")
def table_status():
    """Отдаём сохранённый нормализованный полигон (или null)."""
    global _stream
    poly = None
    if _stream and _stream._table_poly_norm:  # noqa: SLF001
        poly = _stream._table_poly_norm      # noqa: SLF001
    return {"poly_norm": poly}

@app.post("/table/set_roi")
async def table_set_roi(data: Dict[str, Any] = Body(...)):
    global _stream
    if not _stream:
        return {"ok": False, "error": "video not running"}
    poly = data.get("poly")
    if not isinstance(poly, list) or len(poly) < 3:
        return {"ok": False, "error": "poly must be >= 3 points"}
    try:
        _stream.set_table_polygon_norm([(float(x), float(y)) for x, y in poly])
        return {"ok": True, "poly_norm": _stream._table_poly_norm}  # noqa: SLF001
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/table/clear")
async def table_clear():
    global _stream
    if not _stream:
        return {"ok": False, "error": "video not running"}
    _stream.clear_table_polygon()
    return {"ok": True}

@app.post("/table/autodetect")
async def table_autodetect():
    global _stream
    if not _stream:
        return {"ok": False, "error": "video not running"}
    poly = await _stream.auto_detect_table()
    if poly is None:
        return {"ok": False, "error": "no rectangle found"}
    return {"ok": True, "poly_norm": poly}

@app.post("/table/begin")
async def table_begin():
    """Включить режим калибровки (рендерим только стол, не считаем лица/жесты)."""
    global _stream
    if _stream is None:
        return {"ok": False, "error": "video not running"}
    _stream.begin_table_calibration()
    return {"ok": True}

@app.post("/table/end")
async def table_end():
    """Выключить режим калибровки (вернуть полный рендер)."""
    global _stream
    if _stream is None:
        return {"ok": False, "error": "video not running"}
    _stream.end_table_calibration()
    return {"ok": True}

# --------------------------------------------------------------------------------------
# Players (CRUD)
# --------------------------------------------------------------------------------------





@app.post("/players/reset")
def players_reset():
    compreface_deleted = 0
    if _compreface.is_active():
        for p in P.list_players():
            pid = int(p.get("id", 0))
            subject = p.get("face_subject") if isinstance(p.get("face_subject"), str) else _player_face_subject(pid)
            if isinstance(subject, str) and subject and _compreface.delete_subject(subject):
                compreface_deleted += 1
        compreface_deleted += _compreface.delete_all_player_subjects()

    P.reset_players()
    thumbs_dir = os.path.join("storage", "thumbs")
    try:
        for name in os.listdir(thumbs_dir):
            if name.lower().endswith((".jpg",".jpeg",".png")):
                try: os.remove(os.path.join(thumbs_dir, name))
                except: pass
    except FileNotFoundError:
        pass
    return {"ok": True, "compreface_deleted": compreface_deleted}

class _PlayerNameIn(BaseModel):
    id: int
    name: str

@app.post("/players/name")
def players_name(body: _PlayerNameIn):
    ok = P.set_name(body.id, body.name)
    return {"ok": ok}




# --------------------------------------------------------------------------------------
# Fast single-shot enroll (по текущему кадру)
# --------------------------------------------------------------------------------------

@app.post("/players/enroll")
async def players_enroll(data: Dict[str, Any] = Body(None)):
    """
    Щёлкнуть и сразу записать игрока (без сессии и прогресса).
    """
    global _stream
    if not _stream:
        return {"ok": False, "error": "video not running"}

    name = (data or {}).get("name", "")
    frame = await _current_frame_bgr()
    if frame is None:
        return {"ok": False, "error": "no frame"}

    faces = await asyncio.to_thread(_safe_face_analyze, frame)
    if not faces:
        return {"ok": False, "error": "no face"}

    f = _pick_largest_face(faces)
    x1, y1, x2, y2 = _expand_face_bbox(f["bbox"], frame.shape)
    crop = frame[y1:y2, x1:x2]
    emb = f["embedding"].astype(float).tolist()

    thumbs_dir = os.path.join("storage", "thumbs")
    os.makedirs(thumbs_dir, exist_ok=True)
    pid = _players_next_id_fallback()
    thumb_rel = f"thumbs/{pid}.jpg"
    cv2.imwrite(os.path.join(thumbs_dir, f"{pid}.jpg"), crop)

    face_subject: Optional[str] = None
    if _compreface.is_active():
        face_subject = _player_face_subject(pid)
        cf_crop = _compreface_crop_from_face(frame, f["bbox"])
        sample = _encode_jpeg_bytes(cf_crop if cf_crop.size > 0 else crop)
        reg = await asyncio.to_thread(_compreface.register_subject_samples, face_subject, [sample])
        if not bool(reg.get("ok")):
            return {"ok": False, "error": "compreface_enroll_failed", "details": reg}

    player = _players_add_safe(embedding=emb, thumb_rel=thumb_rel, name=name, face_subject=face_subject)
    return {"ok": True, "player": player}

# --------------------------------------------------------------------------------------
# Enroll session API (start → status → step(snap) → finish/cancel)
# --------------------------------------------------------------------------------------

@app.post("/players/enroll/start")
async def enroll_start(data: Dict[str, Any] = Body(None)):
    """
    Начать сессию энролла.
    body: { "name": string (optional), "target": int (optional, default 24) }
    """
    global _enroll
    name = (data or {}).get("name", "")
    target = int((data or {}).get("target", 24))  # More samples for better accuracy and angle coverage
    _enroll = {
        "id": int(time.time() * 1000),
        "name": name,
        "target": target,
        "samples": [],       # List[np.ndarray]
        "images": [],        # List[bytes] - JPEG crops for CompreFace
        "thumb": None,       # np.ndarray (BGR)
        "last_add": 0.0,     # время последнего успешного ДОБАВЛЕНИЯ
        "last_snap": 0.0,    # время последнего СНИМКA (для антиспама)
        "hint": "Смотрите прямо в камеру",  # подсказка для пользователя
        "yaw_left": 0,       # счетчик образцов с поворотом влево
        "yaw_right": 0,      # счетчик образцов с поворотом вправо
        "pitch_up": 0,       # счетчик образцов с головой вверх
        "pitch_down": 0,     # счетчик образцов с головой вниз
        "front": 0,          # счетчик фронтальных образцов
    }
    return {"ok": True, "session": {"id": _enroll["id"], "name": name, "target": target, "count": 0}}




@app.get("/players/enroll/status")
async def enroll_status():
    global _enroll
    async with _enroll_lock:
        if not _enroll:
            return {"ok": False, "error": "no_session"}
        c = len(_enroll["samples"])
        t = _enroll["target"]
        return {"ok": True, "name": _enroll["name"], "count": c, "target": t, "progress": c / max(1, t), "hint": _enroll.get("hint", "")}

@app.post("/players/enroll/snap")
async def enroll_snap():
    """
    Снимок и попытка добавить образец.
    Логика:
      - антиспам: не чаще 180 мс
      - берём крупнейшее лицо
      - фильтр качества
      - фильтр разнообразия
      - если "застряли" (давно не добавляли) — форс-добавляем
    """
    global _stream, _enroll
    if not _stream:
        return {"ok": False, "error": "video not running"}
    if not _enroll:
        return {"ok": False, "error": "no session"}

    now = time.time()
    # антиспам: снимки не чаще ~180 мс
    if now - float(_enroll.get("last_snap", 0.0)) < 0.18:
        return {"ok": True, "added": False, "count": len(_enroll["samples"]), "target": _enroll["target"]}
    _enroll["last_snap"] = now

    frame = await _current_frame_bgr()
    if frame is None:
        return {"ok": False, "error": "no frame"}

    faces = await asyncio.to_thread(_safe_face_analyze, frame)
    if not faces:
        return {"ok": True, "added": False, "reason": "no_face", "count": len(_enroll["samples"]), "target": _enroll["target"]}

    f = _pick_largest_face(faces)
    x1, y1, x2, y2 = _expand_face_bbox(f["bbox"], frame.shape)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {"ok": True, "added": False, "reason": "bad_crop", "count": len(_enroll["samples"]), "target": _enroll["target"]}

    q = _face_quality_score(crop)
    emb: np.ndarray = f["embedding"].astype(np.float32)
    count = len(_enroll["samples"]); target = int(_enroll["target"])
    since_add = now - float(_enroll.get("last_add", 0.0))

    # пороги
    QUALITY_THR = 0.45        # Higher quality samples for better recognition
    DIVERSE_MAX_SIM = 0.88    # More diverse samples for robustness
    cf_crop = _compreface_crop_from_face(frame, f["bbox"])
    sample_jpeg = _encode_jpeg_bytes(cf_crop if cf_crop.size > 0 else crop)

    # проверка качества
    if q < QUALITY_THR and since_add < 1.6:
        return {"ok": True, "added": False, "reason": "low_quality", "quality": q, "count": count, "target": target}

    # разнообразие
    if not _embed_diverse(emb, _enroll["samples"], max_sim=DIVERSE_MAX_SIM):
        # если давно не добавляли — форсируем добавление похожего
        if since_add >= 1.2 or count == 0:
            _enroll["samples"].append(emb)
            if sample_jpeg:
                _enroll["images"].append(sample_jpeg)
            if _enroll["thumb"] is None:
                _enroll["thumb"] = crop.copy()
            _enroll["last_add"] = now
            count = len(_enroll["samples"])
            return {"ok": True, "added": True, "forced": True, "count": count, "target": target}
        else:
            return {"ok": True, "added": False, "reason": "not_diverse", "count": count, "target": target}

    # нормальная успешная добавка
    _enroll["samples"].append(emb)
    if sample_jpeg:
        _enroll["images"].append(sample_jpeg)
    if _enroll["thumb"] is None:
        _enroll["thumb"] = crop.copy()
    _enroll["last_add"] = now
    count = len(_enroll["samples"])

    # Обновляем покрытие ракурсов на основе pts5 (если есть)
    pts5 = f.get("pts5")
    if pts5 is not None and isinstance(pts5, np.ndarray) and pts5.shape == (5, 2):
        yaw, pitch = _angle_from_pts5(pts5)
        # Определяем ракурс - снижены пороги для лучшего покрытия
        if yaw > 6:  # Lowered from 8 to 6
            _enroll["yaw_right"] += 1
        elif yaw < -6:  # Lowered from -8 to -6
            _enroll["yaw_left"] += 1
        else:
            _enroll["front"] += 1

        if pitch > 4:  # Lowered from 6 to 4
            _enroll["pitch_up"] += 1
        elif pitch < -4:  # Lowered from -6 to -4
            _enroll["pitch_down"] += 1

    # Умные подсказки на основе недостающих ракурсов - comprehensive coverage
    if _enroll["front"] < 6:  # More frontal samples for primary recognition
        _enroll["hint"] = "Смотрите прямо в камеру"
    elif _enroll["yaw_left"] < 4:  # More left turn samples
        _enroll["hint"] = "Поверните голову немного влево"
    elif _enroll["yaw_right"] < 4:  # More right turn samples
        _enroll["hint"] = "Поверните голову немного вправо"
    elif _enroll["pitch_up"] < 3:  # Up angle
        _enroll["hint"] = "Поднимите голову чуть выше"
    elif _enroll["pitch_down"] < 3:  # Down angle
        _enroll["hint"] = "Опустите голову чуть ниже"
    else:
        _enroll["hint"] = "Отлично! Продолжайте"

    return {"ok": True, "added": True, "count": count, "target": target, "hint": _enroll["hint"]}

# алиас под фронтовой вызов
@app.post("/players/enroll/step")
async def enroll_step():
    return await enroll_snap()

@app.post("/players/enroll/finish")
async def enroll_finish(data: Dict[str, Any] = Body(None)):
    """Финал: усреднить эмбеддинги и сохранить игрока в БД."""
    global _enroll
    async with _enroll_lock:
        if not _enroll:
            return {"ok": False, "error": "no_session"}
        name_override = (data or {}).get("name")
        name = name_override if (isinstance(name_override, str) and name_override.strip()) else _enroll["name"]

        samples: List[np.ndarray] = _enroll["samples"]
        if len(samples) < 10:
            return {"ok": False, "error": f"need_more_samples ({len(samples)}/10)"}

        # Normalize each sample first
        normalized_samples = []
        for s in samples:
            norm = np.linalg.norm(s)
            if norm > 1e-6:
                normalized_samples.append(s / norm)
            else:
                normalized_samples.append(s)

        # Remove outliers: compute pairwise similarities and remove samples with low avg similarity
        if len(normalized_samples) >= 12:
            stack = np.stack(normalized_samples, axis=0)
            similarities = stack @ stack.T  # cosine similarity matrix
            avg_sims = similarities.mean(axis=1)
            threshold = avg_sims.mean() - 0.5 * avg_sims.std()
            filtered = [normalized_samples[i] for i in range(len(normalized_samples)) if avg_sims[i] >= threshold]
            if len(filtered) >= 10:
                normalized_samples = filtered

        # Average and normalize again
        mean = np.mean(np.stack(normalized_samples, axis=0), axis=0).astype(np.float32)
        mean = mean / (np.linalg.norm(mean) + 1e-6)
        emb_list = mean.astype(float).tolist()

        thumbs_dir = os.path.join("storage", "thumbs")
        os.makedirs(thumbs_dir, exist_ok=True)
        pid = _players_next_id_fallback()
        thumb_rel = f"thumbs/{pid}.jpg"
        face_subject: Optional[str] = None
        if _compreface.is_active():
            face_subject = _player_face_subject(pid)
            images = [bytes(x) for x in _enroll.get("images", []) if isinstance(x, (bytes, bytearray))]
            if len(images) < 4:
                return {"ok": False, "error": f"need_more_face_samples ({len(images)}/4)"}
            max_compreface_samples = max(4, int(os.getenv("COMPREFACE_ENROLL_MAX_SAMPLES", "8")))
            images_for_compreface = _select_evenly_spaced_samples(images, max_compreface_samples)
            reg = await asyncio.to_thread(_compreface.register_subject_samples, face_subject, images_for_compreface)
            if not bool(reg.get("ok")):
                reg["attempted"] = len(images_for_compreface)
                reg["captured"] = len(images)
                return {"ok": False, "error": "compreface_enroll_failed", "details": reg}

        thumb_img = _enroll["thumb"]
        if thumb_img is None:
            thumb_img = np.zeros((120, 120, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(thumbs_dir, f"{pid}.jpg"), thumb_img)

        player = _players_add_safe(
            embedding=emb_list,
            thumb_rel=thumb_rel,
            name=name,
            face_subject=face_subject,
        )
        _enroll = None
        return {"ok": True, "player": player}

@app.post("/players/enroll/cancel")
async def enroll_cancel():
    global _enroll
    async with _enroll_lock:
        _enroll = None
    return {"ok": True}

def _players_with_rev():
    out = []
    for p in P.list_players():
        q = dict(p)
        rev = 0
        thumb = q.get("thumb")
        if isinstance(thumb, str):
            path = os.path.join("storage", thumb)
            try:
                rev = int(os.path.getmtime(path))
            except OSError:
                rev = int(time.time())
        q["rev"] = rev
        out.append(q)
    return out

@app.get("/players/list")
def players_list():
    return {"players": _players_with_rev()}

class _PlayerDeleteIn(BaseModel):
    id: int

@app.post("/players/delete")
def players_delete(body: _PlayerDeleteIn):
    face_subject: Optional[str] = None
    # сначала попробуем удалить файл
    try:
        for p in P.list_players():
            if int(p.get("id", -1)) == body.id:
                thumb = p.get("thumb")
                if isinstance(thumb, str):
                    path = os.path.join("storage", thumb)
                    try: os.remove(path)
                    except FileNotFoundError: pass
                subj = p.get("face_subject")
                if isinstance(subj, str) and subj.strip():
                    face_subject = subj.strip()
                else:
                    face_subject = _player_face_subject(body.id)
                break
    except Exception:
        pass

    if _compreface.is_active() and face_subject:
        _compreface.delete_subject(face_subject)

    ok = P.delete_player(body.id)
    return {"ok": ok}

# --------------------------------------------------------------------------------------
# Voice: register / identify / manage
# --------------------------------------------------------------------------------------

class _VoiceRegisterIn(BaseModel):
    player_id: int
    player_name: str
    audio_samples: List[List[float]]  # List of audio arrays
    sample_rate: int = 16000

@app.post("/voice/register")
async def voice_register(body: _VoiceRegisterIn):
    """
    Register player voice with multiple audio samples.

    body: {
        "player_id": int,
        "player_name": str,
        "audio_samples": List[List[float]],  # multiple audio samples
        "sample_rate": int (optional, default 16000)
    }

    Returns: {"ok": bool, "samples_registered": int (if successful)}
    """
    try:
        vs = get_voice_service()

        # Convert audio samples from lists to numpy arrays
        audio_arrays = [np.array(sample, dtype=np.float32) for sample in body.audio_samples]

        success = vs.register_voice(
            player_id=body.player_id,
            player_name=body.player_name,
            audio_samples=audio_arrays,
            sr=body.sample_rate
        )

        if success:
            profile = vs.profiles.get(body.player_id)
            return {
                "ok": True,
                "samples_registered": len(profile.embeddings) if profile else 0
            }
        else:
            return {"ok": False, "error": "registration_failed"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

class _VoiceIdentifyIn(BaseModel):
    audio: List[float]  # Single audio sample
    sample_rate: int = 16000


class _VoiceSpeechRecognizeIn(BaseModel):
    audio: List[float]
    sample_rate: int = 16000
    add_to_logs: bool = True


@app.post("/voice/identify")
async def voice_identify(body: _VoiceIdentifyIn):
    """
    Identify speaker from audio sample.

    body: {
        "audio": List[float],  # audio data
        "sample_rate": int (optional, default 16000)
    }

    Returns: {
        "ok": bool,
        "player_id": int (if identified),
        "player_name": str (if identified),
        "confidence": float (if identified)
    }
    """
    try:
        vs = get_voice_service()

        # Convert audio from list to numpy array
        audio_array = np.array(body.audio, dtype=np.float32)

        result = vs.identify_speaker(audio_array, sr=body.sample_rate)

        if result:
            player_id, player_name, confidence = result
            return {
                "ok": True,
                "player_id": player_id,
                "player_name": player_name,
                "confidence": confidence
            }
        else:
            return {"ok": True, "player_id": None, "player_name": None, "confidence": 0.0}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/voice/logs/recognize")
async def voice_logs_recognize(body: _VoiceSpeechRecognizeIn):
    """
    Recognize speaker + speech text for one audio chunk and append it to speech logs.
    """
    global _speech_logs_counter
    try:
        audio_array = np.array(body.audio, dtype=np.float32)
        if audio_array.size == 0:
            return {"ok": False, "error": "empty_audio"}
        audio_array = np.nan_to_num(audio_array, copy=False)

        vs = get_voice_service()
        speaker_id: Optional[int] = None
        speaker_name: Optional[str] = None
        confidence = 0.0

        speaker = vs.identify_speaker(audio_array, sr=body.sample_rate)
        if speaker:
            speaker_id, speaker_name, confidence = speaker
        else:
            top_matches = vs.identify_top_k(audio_array, sr=body.sample_rate, k=1)
            if top_matches:
                top = top_matches[0]
                speaker_id = int(top.get("player_id")) if top.get("player_id") is not None else None
                speaker_name = str(top.get("player_name") or "").strip() or None
                confidence = float(top.get("score") or 0.0)
            else:
                guess = _voice_best_guess(vs, audio_array, body.sample_rate)
                if guess:
                    speaker_id, speaker_name, confidence = guess

        text, asr_error = await _transcribe_speech_audio(audio_array, body.sample_rate)
        label = _speech_speaker_label(speaker_id, speaker_name)
        line = _speech_line(label, text)

        entry: Optional[Dict[str, Any]] = None
        if body.add_to_logs:
            async with _speech_logs_lock:
                _speech_logs_counter += 1
                entry = {
                    "id": _speech_logs_counter,
                    "timestamp": float(time.time()),
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_name,
                    "speaker_label": label,
                    "confidence": float(confidence),
                    "text": (text or "").strip() or "...",
                    "line": line,
                }
                _speech_logs.append(entry)
                max_logs = int(os.getenv("SPEECH_LOGS_MAX", "400"))
                if max_logs > 0 and len(_speech_logs) > max_logs:
                    del _speech_logs[:-max_logs]

            await ws_broadcast({"type": "speech.log", "entry": entry})

        return {
            "ok": True,
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "speaker_label": label,
            "confidence": float(confidence),
            "text": (text or "").strip() or "...",
            "line": line,
            "asr_error": asr_error,
            "entry": entry,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/voice/logs")
async def voice_logs(limit: int = 200):
    n = max(1, min(int(limit), 1000))
    async with _speech_logs_lock:
        logs = list(_speech_logs[-n:])
    return {"ok": True, "logs": logs}


@app.post("/voice/logs/clear")
async def voice_logs_clear():
    async with _speech_logs_lock:
        _speech_logs.clear()
    return {"ok": True}


class _VoiceTestIdentifyIn(BaseModel):
    expected_player_id: int
    audio: List[float]
    sample_rate: int = 16000


@app.post("/voice/test/identify")
async def voice_test_identify(body: _VoiceTestIdentifyIn):
    """
    Тестовый роут проверки корректности распознавания.

    body: {
        "expected_player_id": int,
        "audio": List[float],
        "sample_rate": int (optional, default 16000)
    }

    Returns:
    {
        "ok": bool,
        "correct": bool,
        "expected_player_id": int,
        "expected_player_name": str | null,
        "predicted_player_id": int | null,
        "predicted_player_name": str | null,
        "confidence": float,
        "top_matches": [{"player_id": int, "player_name": str, "score": float}]
    }
    """
    try:
        vs = get_voice_service()
        audio_array = np.array(body.audio, dtype=np.float32)
        expected_id = int(body.expected_player_id)

        prediction = vs.identify_speaker(audio_array, sr=body.sample_rate)
        top_matches = vs.identify_top_k(audio_array, sr=body.sample_rate, k=3)
        expected_profile = vs.profiles.get(expected_id)

        predicted_id: Optional[int] = None
        predicted_name: Optional[str] = None
        confidence = 0.0
        if prediction:
            predicted_id, predicted_name, confidence = prediction

        return {
            "ok": True,
            "correct": bool(predicted_id == expected_id),
            "expected_player_id": expected_id,
            "expected_player_name": expected_profile.player_name if expected_profile else None,
            "predicted_player_id": predicted_id,
            "predicted_player_name": predicted_name,
            "confidence": float(confidence),
            "top_matches": top_matches,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


class _VoiceTestSampleIn(BaseModel):
    expected_player_id: int
    audio: List[float]
    sample_rate: int = 16000


class _VoiceTestEvaluateIn(BaseModel):
    samples: List[_VoiceTestSampleIn]


@app.post("/voice/test/evaluate")
async def voice_test_evaluate(body: _VoiceTestEvaluateIn):
    """
    Батч-оценка качества распознавания на наборе тестовых сэмплов.

    body: {
      "samples": [
        {"expected_player_id": int, "audio": List[float], "sample_rate": int}
      ]
    }
    """
    try:
        vs = get_voice_service()
        results: List[Dict[str, Any]] = []
        correct = 0

        for index, sample in enumerate(body.samples):
            audio_array = np.array(sample.audio, dtype=np.float32)
            expected_id = int(sample.expected_player_id)
            prediction = vs.identify_speaker(audio_array, sr=sample.sample_rate)
            top_matches = vs.identify_top_k(audio_array, sr=sample.sample_rate, k=3)

            predicted_id: Optional[int] = None
            predicted_name: Optional[str] = None
            confidence = 0.0
            if prediction:
                predicted_id, predicted_name, confidence = prediction

            is_correct = predicted_id == expected_id
            if is_correct:
                correct += 1

            results.append(
                {
                    "index": index,
                    "correct": bool(is_correct),
                    "expected_player_id": expected_id,
                    "predicted_player_id": predicted_id,
                    "predicted_player_name": predicted_name,
                    "confidence": float(confidence),
                    "top_matches": top_matches,
                }
            )

        total = len(results)
        accuracy = float(correct / total) if total else 0.0

        return {
            "ok": True,
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "results": results,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/voice/profiles")
async def voice_profiles():
    """
    List all registered voice profiles.

    Returns: {"ok": bool, "profiles": List[Dict]}
    """
    try:
        vs = get_voice_service()
        profiles = vs.list_profiles()
        return {"ok": True, "profiles": profiles}
    except Exception as e:
        return {"ok": False, "error": str(e)}

class _VoiceDeleteIn(BaseModel):
    player_id: int

@app.post("/voice/profile/delete")
async def voice_profile_delete(body: _VoiceDeleteIn):
    """
    Delete voice profile for a player.

    body: {"player_id": int}

    Returns: {"ok": bool}
    """
    try:
        vs = get_voice_service()
        success = vs.delete_profile(body.player_id)
        return {"ok": success}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/voice/clear")
async def voice_clear():
    """
    Clear all voice profiles.

    Returns: {"ok": bool}
    """
    try:
        vs = get_voice_service()
        vs.clear_all()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
