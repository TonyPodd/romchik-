"""Player entity - игрок"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

from ..value_objects.player_id import PlayerId
from ..value_objects.role import Role
from .foul import Foul


@dataclass
class Player:
    """
    Игрок в мафии

    Attributes:
        id: Уникальный идентификатор
        name: Имя игрока
        seat: Номер места за столом (1-10)
        face_embedding: Эмбеддинг лица для распознавания (512D вектор)
        thumb_path: Путь к миниатюре лица
        role: Роль в игре (присваивается при старте)
        is_alive: Жив ли игрок
        fouls: Список фолов игрока
    """
    id: PlayerId
    name: str
    seat: int
    face_embedding: np.ndarray
    thumb_path: Optional[str] = None
    role: Optional[Role] = None
    is_alive: bool = True
    fouls: List[Foul] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.id, PlayerId):
            raise TypeError("id must be PlayerId instance")
        if not isinstance(self.seat, int) or self.seat < 1:
            raise ValueError("seat must be positive integer")
        if not isinstance(self.face_embedding, np.ndarray):
            raise TypeError("face_embedding must be numpy array")
        if self.face_embedding.ndim != 1:
            raise ValueError("face_embedding must be 1D array")

    def assign_role(self, role: Role) -> None:
        """Назначить роль игроку"""
        if not isinstance(role, Role):
            raise TypeError("role must be Role instance")
        self.role = role

    def add_foul(self, foul: Foul) -> None:
        """Добавить фол игроку"""
        if not isinstance(foul, Foul):
            raise TypeError("foul must be Foul instance")
        if foul.player_id != self.id:
            raise ValueError("foul player_id doesn't match player id")
        self.fouls.append(foul)

    def eliminate(self) -> None:
        """Исключить игрока из игры"""
        self.is_alive = False

    @property
    def foul_count(self) -> int:
        """Количество фолов"""
        return len(self.fouls)

    @property
    def is_mafia(self) -> bool:
        """Принадлежит ли к мафии"""
        return self.role is not None and self.role.is_mafia_team

    @property
    def is_civilian(self) -> bool:
        """Принадлежит ли к мирным"""
        return self.role is not None and self.role.is_civilian_team

    def to_dict(self) -> dict:
        """Преобразовать в словарь для API"""
        return {
            "id": int(self.id),
            "name": self.name,
            "seat": self.seat,
            "thumb": self.thumb_path,
            "role": self.role.value if self.role else None,
            "is_alive": self.is_alive,
            "foul_count": self.foul_count,
        }
