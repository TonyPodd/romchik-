"""Foul types - типы нарушений"""

from enum import Enum


class FoulType(str, Enum):
    """Типы фолов в игре"""

    SPEAKING_OUT_OF_TURN = "speaking_out_of_turn"       # Говорит не в свою минуту
    GESTURING_DURING_VOTE = "gesturing_during_vote"     # Жестикулирует во время голосования
    TIMEOUT_EXCEEDED = "timeout_exceeded"               # Превышение времени речи
    INVALID_ACTION = "invalid_action"                   # Недопустимое действие

    @property
    def display_name(self) -> str:
        """Отображаемое название фола"""
        names = {
            self.SPEAKING_OUT_OF_TURN: "Речь не в свою минуту",
            self.GESTURING_DURING_VOTE: "Жестикуляция во время голосования",
            self.TIMEOUT_EXCEEDED: "Превышение времени речи",
            self.INVALID_ACTION: "Недопустимое действие",
        }
        return names.get(self, str(self))

    @property
    def severity(self) -> int:
        """Серьезность фола (1-3)"""
        severity_map = {
            self.SPEAKING_OUT_OF_TURN: 2,
            self.GESTURING_DURING_VOTE: 3,
            self.TIMEOUT_EXCEEDED: 1,
            self.INVALID_ACTION: 2,
        }
        return severity_map.get(self, 1)
