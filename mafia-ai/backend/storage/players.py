from __future__ import annotations
import json, os, time
from typing import List, Dict, Any, Optional
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # backend/
STORE_DIR = ROOT / "storage"
THUMBS_DIR = STORE_DIR / "thumbs"
DB_PATH = STORE_DIR / "players.json"

THUMBS_DIR.mkdir(parents=True, exist_ok=True)
if not DB_PATH.exists():
    DB_PATH.write_text(json.dumps({"players":[]}, ensure_ascii=False, indent=2), encoding="utf-8")

def _load() -> Dict[str, Any]:
    return json.loads(DB_PATH.read_text(encoding="utf-8"))

def _save(data: Dict[str, Any]):
    DB_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def list_players() -> List[Dict[str, Any]]:
    return _load().get("players", [])

def reset_players():
    _save({"players":[]})
    # миниатюры оставляем — не критично; при желании можно чистить папку

def next_id() -> int:
    players = list_players()
    return (max([p["id"] for p in players]) + 1) if players else 1

def add_player(embedding: list[float], thumb_rel: str, name: str = "") -> Dict[str, Any]:
    pid = next_id()
    player = {"id": pid, "name": name, "embedding": embedding, "thumb": thumb_rel}
    data = _load()
    data["players"].append(player)
    _save(data)
    return player

def set_name(pid: int, name: str) -> bool:
    data = _load()
    ok = False
    for p in data["players"]:
        if p["id"] == pid:
            p["name"] = name
            ok = True
            break
    if ok: _save(data)
    return ok

def delete_player(pid: int) -> bool:
    data = _load()
    before = len(data["players"])
    data["players"] = [p for p in data["players"] if p["id"] != pid]
    ok = len(data["players"]) < before
    if ok: _save(data)
    # миниатюру по желанию можно удалить
    return ok
