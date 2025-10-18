"""Player ID value object"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerId:
    """
    Идентификатор игрока (value object)
    Неизменяемый, используется для ссылок на игроков
    """
    value: int

    def __post_init__(self):
        if not isinstance(self.value, int) or self.value <= 0:
            raise ValueError(f"PlayerId must be a positive integer, got {self.value}")

    def __int__(self) -> int:
        return self.value

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"PlayerId({self.value})"

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other) -> bool:
        if isinstance(other, PlayerId):
            return self.value == other.value
        if isinstance(other, int):
            return self.value == other
        return False
