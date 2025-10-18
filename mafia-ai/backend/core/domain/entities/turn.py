"""Turn entity - ход игрока"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from ..value_objects.player_id import PlayerId
from ..value_objects.game_phase import GamePhase
from .foul import Foul


@dataclass
class Turn:
    """
    Ход игрока (минута речи)

    Attributes:
        player_id: ID игрока
        phase: Фаза игры
        start_time: Время начала
        duration_seconds: Длительность в секундах
        end_time: Время окончания (заполняется при завершении)
        speech_text: Транскрипция речи
        fouls_detected: Фолы обнаруженные во время хода
        completed: Завершен ли ход
    """
    player_id: PlayerId
    phase: GamePhase
    start_time: datetime
    duration_seconds: int = 60
    end_time: Optional[datetime] = None
    speech_text: Optional[str] = None
    fouls_detected: List[Foul] = field(default_factory=list)
    completed: bool = False

    def __post_init__(self):
        if not isinstance(self.player_id, PlayerId):
            raise TypeError("player_id must be PlayerId instance")
        if not isinstance(self.phase, GamePhase):
            raise TypeError("phase must be GamePhase instance")

    def complete(self, end_time: datetime) -> None:
        """Завершить ход"""
        self.end_time = end_time
        self.completed = True

    def add_foul(self, foul: Foul) -> None:
        """Добавить фол во время хода"""
        if foul.player_id != self.player_id:
            raise ValueError("foul player_id doesn't match turn player_id")
        self.fouls_detected.append(foul)

    @property
    def actual_duration(self) -> Optional[float]:
        """Фактическая длительность в секундах"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def is_overtime(self) -> bool:
        """Превышено ли отведенное время"""
        if self.actual_duration:
            return self.actual_duration > self.duration_seconds
        return False
