"""Game entity - игра"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from uuid import uuid4

from ..value_objects.game_phase import GamePhase
from ..value_objects.player_id import PlayerId
from .player import Player
from .turn import Turn


@dataclass
class Game:
    """
    Игра в мафию

    Attributes:
        id: Уникальный идентификатор игры
        players: Список игроков
        phase: Текущая фаза игры
        current_speaker: ID текущего говорящего
        day_number: Номер игрового дня
        start_time: Время начала игры
        end_time: Время окончания
        turns: История ходов
        eliminated_players: ID исключенных игроков по дням
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    players: List[Player] = field(default_factory=list)
    phase: GamePhase = GamePhase.IDLE
    current_speaker: Optional[PlayerId] = None
    day_number: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    turns: List[Turn] = field(default_factory=list)
    eliminated_players: Dict[int, List[PlayerId]] = field(default_factory=dict)

    def add_player(self, player: Player) -> None:
        """Добавить игрока в игру"""
        if not isinstance(player, Player):
            raise TypeError("player must be Player instance")
        if any(p.id == player.id for p in self.players):
            raise ValueError(f"Player with id {player.id} already exists")
        if any(p.seat == player.seat for p in self.players):
            raise ValueError(f"Seat {player.seat} is already taken")
        self.players.append(player)

    def remove_player(self, player_id: PlayerId) -> bool:
        """Удалить игрока из игры"""
        initial_count = len(self.players)
        self.players = [p for p in self.players if p.id != player_id]
        return len(self.players) < initial_count

    def get_player(self, player_id: PlayerId) -> Optional[Player]:
        """Получить игрока по ID"""
        return next((p for p in self.players if p.id == player_id), None)

    def start_game(self) -> None:
        """Начать игру"""
        if len(self.players) < 4:
            raise ValueError("Need at least 4 players to start")
        if self.phase != GamePhase.IDLE:
            raise ValueError("Game already started")
        self.phase = GamePhase.ROLE_ASSIGNMENT
        self.start_time = datetime.now()
        self.day_number = 0

    def change_phase(self, new_phase: GamePhase) -> None:
        """Сменить фазу игры"""
        if not isinstance(new_phase, GamePhase):
            raise TypeError("new_phase must be GamePhase instance")
        self.phase = new_phase

    def start_turn(self, player_id: PlayerId, duration: int = 60) -> Turn:
        """Начать ход игрока"""
        player = self.get_player(player_id)
        if not player:
            raise ValueError(f"Player {player_id} not found")
        if not player.is_alive:
            raise ValueError(f"Player {player_id} is not alive")

        turn = Turn(
            player_id=player_id,
            phase=self.phase,
            start_time=datetime.now(),
            duration_seconds=duration,
        )
        self.turns.append(turn)
        self.current_speaker = player_id
        return turn

    def end_turn(self) -> None:
        """Завершить текущий ход"""
        if self.current_speaker and self.turns:
            last_turn = self.turns[-1]
            if not last_turn.completed:
                last_turn.complete(datetime.now())
        self.current_speaker = None

    def eliminate_player(self, player_id: PlayerId) -> None:
        """Исключить игрока"""
        player = self.get_player(player_id)
        if not player:
            raise ValueError(f"Player {player_id} not found")
        player.eliminate()

        if self.day_number not in self.eliminated_players:
            self.eliminated_players[self.day_number] = []
        self.eliminated_players[self.day_number].append(player_id)

    @property
    def alive_players(self) -> List[Player]:
        """Живые игроки"""
        return [p for p in self.players if p.is_alive]

    @property
    def alive_mafia(self) -> List[Player]:
        """Живая мафия"""
        return [p for p in self.alive_players if p.is_mafia]

    @property
    def alive_civilians(self) -> List[Player]:
        """Живые мирные"""
        return [p for p in self.alive_players if p.is_civilian]

    @property
    def is_game_over(self) -> bool:
        """Закончена ли игра"""
        mafia_count = len(self.alive_mafia)
        civilian_count = len(self.alive_civilians)

        # Мафия победила если их >= мирных
        if mafia_count >= civilian_count:
            return True
        # Мирные победили если мафии нет
        if mafia_count == 0:
            return True
        return False

    @property
    def winner(self) -> Optional[str]:
        """Победитель игры"""
        if not self.is_game_over:
            return None
        if len(self.alive_mafia) == 0:
            return "civilians"
        return "mafia"

    def end_game(self) -> None:
        """Завершить игру"""
        self.phase = GamePhase.GAME_END
        self.end_time = datetime.now()
