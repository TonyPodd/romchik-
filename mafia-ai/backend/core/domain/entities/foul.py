"""Foul entity - фол игрока"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..value_objects.foul_type import FoulType
from ..value_objects.player_id import PlayerId


@dataclass
class Foul:
    """
    Фол (нарушение правил)
    """
    player_id: PlayerId
    foul_type: FoulType
    timestamp: datetime
    phase: str  # Фаза игры когда произошел фол
    description: Optional[str] = None
    auto_detected: bool = True  # Автоматически обнаружен или вручную

    def __post_init__(self):
        if not isinstance(self.player_id, PlayerId):
            raise TypeError("player_id must be PlayerId instance")
        if not isinstance(self.foul_type, FoulType):
            raise TypeError("foul_type must be FoulType instance")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be datetime instance")
