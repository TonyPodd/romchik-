from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import asyncio, time, os
from fastapi import Body

from video.stream_worker import GestureStream

app = FastAPI(title="Mafia AI Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

clients = set()
_stream: GestureStream | None = None

async def ws_broadcast(msg: dict):
    dead = []
    for ws in list(clients):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for d in dead:
        clients.discard(d)

@app.get("/health")
def health():
    return {
        "ok": True,
        "clients": len(clients),
        "video_running": _stream is not None
    }

@app.get("/video/status")
def video_status():
    return {"running": _stream is not None}

@app.post("/video/start")
async def video_start(camera_index: int = None, fps: int = None, table_y_ratio: float = None):
    global _stream
    if _stream:
        return {"ok": True, "status": "already_running"}

    cam = int(os.getenv("CAMERA_INDEX", "0")) if camera_index is None else int(camera_index)
    f = int(os.getenv("GESTURE_FPS", "15")) if fps is None else int(fps)
    tyr = float(os.getenv("TABLE_Y_RATIO", "0.80")) if table_y_ratio is None else float(table_y_ratio)

    _stream = GestureStream(on_event=ws_broadcast, camera_index=cam, fps=f, table_y_ratio=tyr)
    try:
        await _stream.start()
        print(f"[app] gesture stream started (camera={cam})")
        return {"ok": True, "camera_index": cam, "fps": f, "table_y_ratio": tyr}
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

BOUNDARY = "frame"

async def _mjpeg_generator():
    # отдаём ~15 FPS, либо последний доступный кадр
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
        await asyncio.sleep(1/60)

@app.get("/video/mjpeg")
async def video_mjpeg():
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return StreamingResponse(_mjpeg_generator(),
                             media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                             headers=headers)

@app.on_event("startup")
async def _startup():
    auto = os.getenv("AUTO_START_GESTURES", "1") == "1"
    if auto:
        await video_start()

@app.on_event("shutdown")
async def _shutdown():
    global _stream
    if _stream:
        await _stream.stop()
        _stream = None

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
                ms = int(data.get("ms", 60000))
                asyncio.create_task(run_timer(seat, ms))
            elif t == "ping":
                await ws.send_json({"type": "pong"})
    finally:
        clients.discard(ws)
        print(f"[ws] client disconnected, total={len(clients)}")

async def run_timer(seat: int, ms: int):
    end = time.monotonic() + ms / 1000
    while True:
        left = max(0.0, end - time.monotonic())
        await ws_broadcast({"type": "timer.tick", "seat": seat, "msLeft": int(left * 1000)})
        if left <= 0.0:
            break
        await asyncio.sleep(0.1)
    await ws_broadcast({"type": "timer.end", "seat": seat})

@app.get("/table/status")
def table_status():
    # отдаём сохранённый нормализованный полигон (или null)
    global _stream
    poly = None
    if _stream and _stream._table_poly_norm:
        poly = _stream._table_poly_norm
    return {"poly_norm": poly}

@app.post("/table/set_roi")
async def table_set_roi(data: dict = Body(...)):
    global _stream
    if not _stream:
        return {"ok": False, "error": "video not running"}
    poly = data.get("poly")
    if not isinstance(poly, list) or len(poly) < 3:
        return {"ok": False, "error": "poly must be >= 3 points"}
    try:
        _stream.set_table_polygon_norm([(float(x), float(y)) for x, y in poly])
        return {"ok": True, "poly_norm": _stream._table_poly_norm}
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