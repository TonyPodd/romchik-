"""Player roles - роли игроков"""

from enum import Enum


class Role(str, Enum):
    """Роли в игре мафия"""

    CIVILIAN = "civilian"   # Мирный житель
    MAFIA = "mafia"         # Мафия
    SHERIFF = "sheriff"     # Шериф
    DON = "don"             # Дон мафии

    @property
    def is_mafia_team(self) -> bool:
        """Относится ли к команде мафии"""
        return self in {self.MAFIA, self.DON}

    @property
    def is_civilian_team(self) -> bool:
        """Относится ли к команде мирных"""
        return self in {self.CIVILIAN, self.SHERIFF}

    @property
    def display_name(self) -> str:
        """Отображаемое название роли"""
        names = {
            self.CIVILIAN: "Мирный житель",
            self.MAFIA: "Мафия",
            self.SHERIFF: "Шериф",
            self.DON: "Дон",
        }
        return names.get(self, str(self))
