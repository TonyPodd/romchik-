"""Repository interfaces - абстракции для хранилища"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from ..domain.entities.player import Player
from ..domain.entities.game import Game
from ..domain.value_objects.player_id import PlayerId


class IPlayerRepository(ABC):
    """
    Интерфейс репозитория игроков
    Легко заменить JSON на SQL
    """

    @abstractmethod
    async def add(self, name: str, seat: int, face_embedding: np.ndarray, thumb_path: Optional[str]) -> Player:
        """Добавить нового игрока"""
        pass

    @abstractmethod
    async def get(self, player_id: PlayerId) -> Optional[Player]:
        """Получить игрока по ID"""
        pass

    @abstractmethod
    async def list_all(self) -> List[Player]:
        """Получить всех игроков"""
        pass

    @abstractmethod
    async def update(self, player: Player) -> bool:
        """Обновить игрока"""
        pass

    @abstractmethod
    async def delete(self, player_id: PlayerId) -> bool:
        """Удалить игрока"""
        pass

    @abstractmethod
    async def reset(self) -> None:
        """Очистить всех игроков"""
        pass

    @abstractmethod
    async def find_by_embedding(self, embedding: np.ndarray, threshold: float = 0.4) -> Optional[Player]:
        """Найти игрока по эмбеддингу лица (cosine similarity)"""
        pass


class IGameRepository(ABC):
    """
    Интерфейс репозитория игр
    """

    @abstractmethod
    async def create(self, game: Game) -> Game:
        """Создать новую игру"""
        pass

    @abstractmethod
    async def get(self, game_id: str) -> Optional[Game]:
        """Получить игру по ID"""
        pass

    @abstractmethod
    async def get_current(self) -> Optional[Game]:
        """Получить текущую активную игру"""
        pass

    @abstractmethod
    async def update(self, game: Game) -> bool:
        """Обновить игру"""
        pass

    @abstractmethod
    async def delete(self, game_id: str) -> bool:
        """Удалить игру"""
        pass

    @abstractmethod
    async def list_all(self) -> List[Game]:
        """Получить все игры"""
        pass
