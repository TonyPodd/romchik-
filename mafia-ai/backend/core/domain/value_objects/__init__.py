"""Value objects - неизменяемые объекты без идентичности"""

from .game_phase import GamePhase
from .role import Role
from .foul_type import FoulType
from .player_id import PlayerId

__all__ = ["GamePhase", "Role", "FoulType", "PlayerId"]
