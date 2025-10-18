"""JSON implementation of player repository - совместимость с текущим storage/players.py"""

import json
import os
from pathlib import Path
from typing import List, Optional
import numpy as np

from core.domain.entities.player import Player
from core.domain.value_objects.player_id import PlayerId
from core.interfaces.repositories import IPlayerRepository
from config.settings import settings


class JsonPlayerRepository(IPlayerRepository):
    """
    JSON-based player repository
    Совместим с текущим storage/players.json
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or settings.players_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir = self.db_path.parent / "thumbs"
        self.thumbs_dir.mkdir(exist_ok=True)

        if not self.db_path.exists():
            self._save({"players": []})

    def _load(self) -> dict:
        """Загрузить данные из JSON"""
        try:
            return json.loads(self.db_path.read_text(encoding="utf-8"))
        except Exception:
            return {"players": []}

    def _save(self, data: dict) -> None:
        """Сохранить данные в JSON"""
        self.db_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def _next_id(self) -> int:
        """Получить следующий ID"""
        data = self._load()
        players = data.get("players", [])
        if not players:
            return 1
        return max(p["id"] for p in players) + 1

    def _player_to_dict(self, player: Player) -> dict:
        """Конвертировать Player в dict для JSON"""
        return {
            "id": int(player.id),
            "name": player.name,
            "seat": player.seat,
            "embedding": player.face_embedding.tolist(),
            "thumb": player.thumb_path,
        }

    def _dict_to_player(self, data: dict) -> Player:
        """Конвертировать dict в Player"""
        return Player(
            id=PlayerId(data["id"]),
            name=data.get("name", ""),
            seat=data.get("seat", 1),
            face_embedding=np.array(data["embedding"], dtype=np.float32),
            thumb_path=data.get("thumb"),
        )

    async def add(
        self,
        name: str,
        seat: int,
        face_embedding: np.ndarray,
        thumb_path: Optional[str]
    ) -> Player:
        """Добавить нового игрока"""
        player_id = PlayerId(self._next_id())
        player = Player(
            id=player_id,
            name=name,
            seat=seat,
            face_embedding=face_embedding,
            thumb_path=thumb_path,
        )

        data = self._load()
        data["players"].append(self._player_to_dict(player))
        self._save(data)

        return player

    async def get(self, player_id: PlayerId) -> Optional[Player]:
        """Получить игрока по ID"""
        data = self._load()
        for p_data in data.get("players", []):
            if p_data["id"] == int(player_id):
                return self._dict_to_player(p_data)
        return None

    async def list_all(self) -> List[Player]:
        """Получить всех игроков"""
        data = self._load()
        return [self._dict_to_player(p) for p in data.get("players", [])]

    async def update(self, player: Player) -> bool:
        """Обновить игрока"""
        data = self._load()
        players = data.get("players", [])

        for i, p in enumerate(players):
            if p["id"] == int(player.id):
                players[i] = self._player_to_dict(player)
                self._save(data)
                return True
        return False

    async def delete(self, player_id: PlayerId) -> bool:
        """Удалить игрока"""
        data = self._load()
        initial_count = len(data.get("players", []))
        data["players"] = [p for p in data.get("players", []) if p["id"] != int(player_id)]

        if len(data["players"]) < initial_count:
            self._save(data)
            # Попытка удалить миниатюру
            player = await self.get(player_id)
            if player and player.thumb_path:
                thumb_full_path = self.db_path.parent / player.thumb_path
                try:
                    thumb_full_path.unlink(missing_ok=True)
                except Exception:
                    pass
            return True
        return False

    async def reset(self) -> None:
        """Очистить всех игроков"""
        self._save({"players": []})
        # Опционально: очистить миниатюры
        try:
            for file in self.thumbs_dir.glob("*.jpg"):
                file.unlink()
        except Exception:
            pass

    async def find_by_embedding(
        self,
        embedding: np.ndarray,
        threshold: float = 0.4
    ) -> Optional[Player]:
        """
        Найти игрока по эмбеддингу лица (cosine similarity)
        """
        players = await self.list_all()
        if not players:
            return None

        # Нормализация входного эмбеддинга
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-6)

        best_player = None
        best_sim = -1.0

        for player in players:
            # Нормализация эмбеддинга игрока
            player_emb = player.face_embedding / (np.linalg.norm(player.face_embedding) + 1e-6)

            # Cosine similarity
            similarity = float(np.dot(emb_norm, player_emb))

            if similarity > best_sim:
                best_sim = similarity
                best_player = player

        if best_sim >= threshold:
            return best_player
        return None
