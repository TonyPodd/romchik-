"""JSON implementation of game repository"""

import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import numpy as np

from core.domain.entities.game import Game
from core.domain.entities.player import Player
from core.domain.value_objects.player_id import PlayerId
from core.domain.value_objects.game_phase import GamePhase
from core.domain.value_objects.role import Role
from core.interfaces.repositories import IGameRepository
from config.settings import settings


class JsonGameRepository(IGameRepository):
    """
    JSON-based game repository
    Хранит игры в JSON файле
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or settings.games_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.db_path.exists():
            self._save({"games": [], "current_game_id": None})

    def _load(self) -> dict:
        """Загрузить данные"""
        try:
            return json.loads(self.db_path.read_text(encoding="utf-8"))
        except Exception:
            return {"games": [], "current_game_id": None}

    def _save(self, data: dict) -> None:
        """Сохранить данные"""
        self.db_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8"
        )

    def _game_to_dict(self, game: Game) -> dict:
        """Конвертировать Game в dict"""
        return {
            "id": game.id,
            "phase": game.phase.value,
            "current_speaker": int(game.current_speaker) if game.current_speaker else None,
            "day_number": game.day_number,
            "start_time": game.start_time.isoformat() if game.start_time else None,
            "end_time": game.end_time.isoformat() if game.end_time else None,
            "players": [
                {
                    "id": int(p.id),
                    "name": p.name,
                    "seat": p.seat,
                    "role": p.role.value if p.role else None,
                    "is_alive": p.is_alive,
                    "embedding": p.face_embedding.tolist(),
                    "thumb": p.thumb_path,
                }
                for p in game.players
            ],
            "eliminated_players": {
                str(day): [int(pid) for pid in pids]
                for day, pids in game.eliminated_players.items()
            },
        }

    def _dict_to_game(self, data: dict) -> Game:
        """Конвертировать dict в Game"""
        # Восстановить игроков
        players = []
        for p_data in data.get("players", []):
            player = Player(
                id=PlayerId(p_data["id"]),
                name=p_data["name"],
                seat=p_data["seat"],
                face_embedding=np.array(p_data["embedding"], dtype=np.float32),
                thumb_path=p_data.get("thumb"),
                role=Role(p_data["role"]) if p_data.get("role") else None,
                is_alive=p_data.get("is_alive", True),
            )
            players.append(player)

        # Восстановить игру
        game = Game(
            id=data["id"],
            players=players,
            phase=GamePhase(data.get("phase", "idle")),
            current_speaker=PlayerId(data["current_speaker"]) if data.get("current_speaker") else None,
            day_number=data.get("day_number", 0),
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
        )

        # Восстановить eliminated_players
        if "eliminated_players" in data:
            game.eliminated_players = {
                int(day): [PlayerId(pid) for pid in pids]
                for day, pids in data["eliminated_players"].items()
            }

        return game

    async def create(self, game: Game) -> Game:
        """Создать новую игру"""
        data = self._load()
        data["games"].append(self._game_to_dict(game))
        data["current_game_id"] = game.id
        self._save(data)
        return game

    async def get(self, game_id: str) -> Optional[Game]:
        """Получить игру по ID"""
        data = self._load()
        for g_data in data.get("games", []):
            if g_data["id"] == game_id:
                return self._dict_to_game(g_data)
        return None

    async def get_current(self) -> Optional[Game]:
        """Получить текущую активную игру"""
        data = self._load()
        current_id = data.get("current_game_id")
        if current_id:
            return await self.get(current_id)
        return None

    async def update(self, game: Game) -> bool:
        """Обновить игру"""
        data = self._load()
        games = data.get("games", [])

        for i, g in enumerate(games):
            if g["id"] == game.id:
                games[i] = self._game_to_dict(game)
                self._save(data)
                return True
        return False

    async def delete(self, game_id: str) -> bool:
        """Удалить игру"""
        data = self._load()
        initial_count = len(data.get("games", []))
        data["games"] = [g for g in data.get("games", []) if g["id"] != game_id]

        if data.get("current_game_id") == game_id:
            data["current_game_id"] = None

        if len(data["games"]) < initial_count:
            self._save(data)
            return True
        return False

    async def list_all(self) -> List[Game]:
        """Получить все игры"""
        data = self._load()
        return [self._dict_to_game(g) for g in data.get("games", [])]
