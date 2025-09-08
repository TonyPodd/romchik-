# init_mafia_ai.py
from pathlib import Path

root = Path("mafia-ai")
files = {}

# -------- repo scaffolding --------
files[".gitignore"] = """# Python
__pycache__/
*.pyc
.venv/
# Node
node_modules/
dist/
.next/
.DS_Store
"""

files["README.md"] = """# Mafia AI — MVP

## Backend (FastAPI)
- WebSocket /ws для событий реального времени
- Таймер игрока (tick/end), базовый broadcast
- Запуск: uvicorn app:app --reload

## Frontend (React + Vite + TS)
- Подключение к WS, стол на 10 мест, простой таймер
- Запуск: npm run dev
"""

# -------- backend --------
b = root / "backend"
(b / "audio").mkdir(parents=True, exist_ok=True)
(b / "video").mkdir(parents=True, exist_ok=True)
(b / "rules").mkdir(parents=True, exist_ok=True)
(b / "storage").mkdir(parents=True, exist_ok=True)

files["backend/requirements.txt"] = """fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
"""

files["backend/app.py"] = r'''from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio, time

app = FastAPI(title="Mafia AI Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

clients = set()

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
    return {"ok": True}

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
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

async def run_timer(seat: int, ms: int):
    end = time.monotonic() + ms / 1000
    while True:
        left = max(0.0, end - time.monotonic())
        await ws_broadcast({"type": "timer.tick", "seat": seat, "msLeft": int(left * 1000)})
        if left <= 0.0:
            break
        await asyncio.sleep(0.1)
    await ws_broadcast({"type": "timer.end", "seat": seat})
'''

files["backend/fsm.py"] = r'''from enum import Enum, auto

class Phase(Enum):
    FREE_SEATING = auto()
    NIGHT_INTRO = auto()
    DAY_DISCUSS = auto()
    NOMINATIONS = auto()
    VOTE = auto()
    TIE_30S = auto()
    REVOTE = auto()
    LAST_WORD = auto()
    NIGHT = auto()
'''

files["backend/rules/config.yaml"] = """timers:
  speech_ms: 60000
  last_word_ms: 60000
  vote_window_ms: 1500
  tie_speech_ms: 30000
"""

files["backend/storage/db.py"] = r'''# заглушка под БД (SQLite/Postgres позже)
from typing import List, Dict, Any
logs: List[Dict[str, Any]] = []
def add_log(event: Dict[str, Any]): logs.append(event)
'''

# placeholders for modules
files["backend/audio/vad.py"] = "def detect_voice_segments(wav):\n    return []\n"
files["backend/audio/diarize.py"] = "def diarize(segments):\n    return []\n"
files["backend/audio/asr.py"] = "def transcribe(segment):\n    return \"\"\n"
files["backend/video/capture.py"] = "def frames():\n    yield None\n"
files["backend/video/faces.py"] = "def recognize(frame):\n    return []\n"
files["backend/video/gestures.py"] = "def detect_gestures(frame):\n    return []\n"

# -------- frontend (React + Vite + TS) --------
f = root / "frontend" / "src" / "components"
f.mkdir(parents=True, exist_ok=True)

files["frontend/package.json"] = r'''{
  "name": "mafia-ai-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@types/react": "^18.3.5",
    "@types/react-dom": "^18.3.0",
    "typescript": "^5.6.2",
    "vite": "^5.4.6"
  }
}
'''

files["frontend/tsconfig.json"] = r'''{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "Bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true
  },
  "include": ["src"]
}
'''

files["frontend/vite.config.ts"] = r'''import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, host: true }
})
'''

files["frontend/index.html"] = r'''<!doctype html>
<html lang="ru">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Mafia AI — UI</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
'''

files["frontend/src/main.tsx"] = r'''import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'

createRoot(document.getElementById('root')!).render(<App />)
'''

files["frontend/src/App.tsx"] = r'''import React, { useEffect, useMemo, useState } from 'react'
import Table from './components/Table'

export default function App(){
  const [log, setLog] = useState<string[]>([])
  const ws = useMemo(()=> new WebSocket('ws://localhost:8000/ws'), [])
  useEffect(()=>{
    ws.onmessage = (e)=>{
      const m = JSON.parse(e.data)
      if(m.type==='timer.tick') setLog(l=>[`tick seat ${m.seat}: ${m.msLeft}ms`, ...l].slice(0,5))
      if(m.type==='timer.end') setLog(l=>[`END seat ${m.seat}`, ...l].slice(0,5))
    }
    ws.onopen = ()=> setLog(l=>['WS connected', ...l].slice(0,5))
    ws.onclose = ()=> setLog(l=>['WS disconnected', ...l].slice(0,5))
    return ()=> ws.close()
  }, [])

  const startTimer = (seat:number, ms:number)=> ws.send(JSON.stringify({type:'timer.start', seat, ms}))
  return (
    <div style={{fontFamily:'Inter, system-ui, sans-serif', padding:16}}>
      <h1>Mafia AI — Console UI</h1>
      <p>Запустите бэкенд на :8000, потом можно стартовать таймеры мест.</p>
      <Table onStart={(seat)=> startTimer(seat, 60000)} />
      <div style={{marginTop:16}}>
        <h3>Log</h3>
        <ul>{log.map((s,i)=><li key={i}>{s}</li>)}</ul>
      </div>
    </div>
  )
}
'''

files["frontend/src/components/Table.tsx"] = r'''import React from 'react'
import Timer from './Timer'

export default function Table({ onStart }:{ onStart:(seat:number)=>void }){
  const seats = Array.from({length:10}, (_,i)=>i+1)
  return (
    <div style={{display:'grid', gridTemplateColumns:'repeat(5, 1fr)', gap:12}}>
      {seats.map(seat=>(
        <div key={seat} style={{border:'1px solid #ddd', borderRadius:12, padding:12}}>
          <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
            <strong>#{seat}</strong>
            <button onClick={()=> onStart(seat)} style={{padding:'4px 8px'}}>Start 60s</button>
          </div>
          <Timer seat={seat}/>
        </div>
      ))}
    </div>
  )
}
'''

files["frontend/src/components/Timer.tsx"] = r'''import React, { useEffect, useState } from 'react'

export default function Timer({ seat }:{ seat:number }){
  const [ms, setMs] = useState(0)
  useEffect(()=>{
    const onMsg = (e: MessageEvent)=>{
      try{
        const m = JSON.parse(e.data as string)
        if(m.type==='timer.tick' && m.seat===seat) setMs(m.msLeft)
        if(m.type==='timer.end' && m.seat===seat) setMs(0)
      }catch{}
    }
    const wsList = (window as any).webSocketListeners ?? ((window as any).webSocketListeners = [])
    wsList.push(onMsg)
    // patch global to hook the single WS created in App
    const orig = WebSocket.prototype.onmessage
    Object.defineProperty(WebSocket.prototype, 'onmessage', {
      set(fn){
        orig
        this.addEventListener('message', (e)=> wsList.forEach((h:any)=>h(e)))
        this.addEventListener('message', fn as any)
      }
    })
    return ()=>{}
  }, [seat])
  const s = Math.floor(ms/1000)
  const d = String(Math.floor((ms%1000)/100)).padStart(1,'0')
  return <div style={{fontFamily:'monospace', fontSize:24}}>{s}.{d}s</div>
}
'''

# -------- docker (optional backend only) --------
files["docker-compose.yml"] = r'''services:
  backend:
    image: python:3.11-slim
    working_dir: /app
    volumes:
      - ./backend:/app
    command: bash -lc "pip install -r requirements.txt && uvicorn app:app --host 0.0.0.0 --port 8000"
    ports:
      - "8000:8000"
'''

# write all files
for rel, content in files.items():
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

print("Scaffold created at ./mafia-ai")
