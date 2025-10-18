"""Domain entities - сущности с идентичностью"""

from .player import Player
from .game import Game
from .turn import Turn
from .foul import Foul

__all__ = ["Player", "Game", "Turn", "Foul"]
