from enum import Enum, auto

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
