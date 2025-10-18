"""Game phases - фазы игры"""

from enum import Enum, auto


class GamePhase(str, Enum):
    """Фазы игры в мафию"""

    # Подготовка
    IDLE = "idle"                           # Ожидание
    SETUP = "setup"                         # Регистрация игроков, калибровка
    ROLE_ASSIGNMENT = "role_assignment"     # Раздача ролей

    # Игровой процесс
    NIGHT_0 = "night_0"                     # Нулевая ночь (знакомство мафии)
    DAY_DISCUSSION = "day_discussion"       # Дневное обсуждение (60s на игрока)
    NOMINATIONS = "nominations"             # Выдвижение кандидатов на вылет
    VOTING = "voting"                       # Голосование (БЕЗ жестов!)
    TIE_SPEECH = "tie_speech"               # Речь при равенстве голосов (30s)
    REVOTE = "revote"                       # Переголосование
    LAST_WORD = "last_word"                 # Последнее слово (60s)
    NIGHT = "night"                         # Ночь (мафия убивает)

    # Завершение
    GAME_END = "game_end"                   # Конец игры

    def is_speaking_allowed(self) -> bool:
        """Разрешена ли речь в этой фазе"""
        return self in {
            self.DAY_DISCUSSION,
            self.NOMINATIONS,
            self.TIE_SPEECH,
            self.LAST_WORD,
        }

    def is_gesture_allowed(self) -> bool:
        """Разрешены ли жесты в этой фазе"""
        # Жесты запрещены только во время голосования
        return self != self.VOTING

    def requires_timer(self) -> bool:
        """Требуется ли таймер в этой фазе"""
        return self in {
            self.DAY_DISCUSSION,
            self.TIE_SPEECH,
            self.LAST_WORD,
        }

    def get_timer_duration(self) -> int:
        """Длительность таймера в секундах"""
        if self == self.DAY_DISCUSSION:
            return 60
        elif self == self.LAST_WORD:
            return 60
        elif self == self.TIE_SPEECH:
            return 30
        return 0
