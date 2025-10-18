"""Game Engine - управление состоянием и логикой игры"""

from typing import List, Optional, Dict
from datetime import datetime
from statemachine import StateMachine, State
from statemachine.exceptions import TransitionNotAllowed

from core.domain.entities.game import Game
from core.domain.entities.player import Player
from core.domain.entities.turn import Turn
from core.domain.entities.foul import Foul
from core.domain.value_objects.game_phase import GamePhase
from core.domain.value_objects.role import Role
from core.domain.value_objects.foul_type import FoulType
from core.domain.value_objects.player_id import PlayerId


class MafiaGameStateMachine(StateMachine):
    """
    State Machine для игры в Мафию

    Переходы между фазами:
    IDLE → SETUP → ROLE_ASSIGNMENT → NIGHT_0 → DAY_DISCUSSION ↔ NOMINATIONS ↔ VOTING → LAST_WORD → NIGHT → ...
    В любой момент можно перейти в GAME_END
    """

    # Определяем состояния
    idle = State(initial=True, value=GamePhase.IDLE)
    setup = State(value=GamePhase.SETUP)
    role_assignment = State(value=GamePhase.ROLE_ASSIGNMENT)
    night_0 = State(value=GamePhase.NIGHT_0)
    day_discussion = State(value=GamePhase.DAY_DISCUSSION)
    nominations = State(value=GamePhase.NOMINATIONS)
    voting = State(value=GamePhase.VOTING)
    last_word = State(value=GamePhase.LAST_WORD)
    night = State(value=GamePhase.NIGHT)
    game_end = State(value=GamePhase.GAME_END, final=True)

    # Переходы
    start_setup = idle.to(setup)
    assign_roles = setup.to(role_assignment)
    start_night_0 = role_assignment.to(night_0)
    start_day = night_0.to(day_discussion) | night.to(day_discussion)
    open_nominations = day_discussion.to(nominations)
    start_voting = nominations.to(voting)
    back_to_discussion = voting.to(day_discussion)
    give_last_word = voting.to(last_word)
    start_night = last_word.to(night)
    end_game = (
        idle.to(game_end) |
        setup.to(game_end) |
        role_assignment.to(game_end) |
        night_0.to(game_end) |
        day_discussion.to(game_end) |
        nominations.to(game_end) |
        voting.to(game_end) |
        last_word.to(game_end) |
        night.to(game_end)
    )


class GameEngine:
    """
    Game Engine - управляет игровым процессом

    Отвечает за:
    - Переходы между фазами
    - Валидацию действий игроков
    - Обнаружение фолов
    - Определение победителя
    """

    def __init__(self, game: Game):
        """
        Args:
            game: Игровая сессия
        """
        self.game = game
        self.state_machine = MafiaGameStateMachine()
        self.current_turn: Optional[Turn] = None
        self.current_speaker_id: Optional[int] = None

        # Синхронизируем state machine с текущей фазой игры
        self._sync_state_from_game()

    def _sync_state_from_game(self):
        """Синхронизация state machine с текущей фазой игры"""
        phase = self.game.phase

        # Переводим state machine в нужное состояние
        state_map = {
            GamePhase.IDLE: "idle",
            GamePhase.SETUP: "setup",
            GamePhase.ROLE_ASSIGNMENT: "role_assignment",
            GamePhase.NIGHT_0: "night_0",
            GamePhase.DAY_DISCUSSION: "day_discussion",
            GamePhase.NOMINATIONS: "nominations",
            GamePhase.VOTING: "voting",
            GamePhase.LAST_WORD: "last_word",
            GamePhase.NIGHT: "night",
            GamePhase.GAME_END: "game_end",
        }

        target_state = state_map.get(phase)
        if target_state:
            # Принудительно устанавливаем состояние
            self.state_machine.current_state = getattr(self.state_machine, target_state)

    # ========== Управление фазами ==========

    def start_game(self, players: List[Player]) -> bool:
        """
        Начать новую игру

        Args:
            players: Список игроков (должно быть >= 10)

        Returns:
            True если успешно
        """
        if len(players) < 10:
            print(f"[GameEngine] ❌ Minimum 10 players required, got {len(players)}")
            return False

        try:
            self.state_machine.start_setup()
            self.game.players = players
            self.game.phase = GamePhase.SETUP
            self.game.start_time = datetime.now()

            print(f"[GameEngine] ✅ Game started with {len(players)} players")
            return True

        except TransitionNotAllowed as e:
            print(f"[GameEngine] ❌ Cannot start game: {e}")
            return False

    def assign_roles(self, role_assignments: Dict[int, Role]) -> bool:
        """
        Назначить роли игрокам

        Args:
            role_assignments: Словарь {player_id: role}

        Returns:
            True если успешно
        """
        try:
            self.state_machine.assign_roles()

            # Назначаем роли
            for player in self.game.players:
                if player.id.value in role_assignments:
                    player.role = role_assignments[player.id.value]

            self.game.phase = GamePhase.ROLE_ASSIGNMENT

            print(f"[GameEngine] ✅ Roles assigned to {len(role_assignments)} players")
            return True

        except TransitionNotAllowed as e:
            print(f"[GameEngine] ❌ Cannot assign roles: {e}")
            return False

    def start_night_zero(self) -> bool:
        """Начать нулевую ночь (знакомство мафии)"""
        try:
            self.state_machine.start_night_0()
            self.game.phase = GamePhase.NIGHT_0
            self.game.day_number = 0

            print(f"[GameEngine] ✅ Night 0 started")
            return True

        except TransitionNotAllowed as e:
            print(f"[GameEngine] ❌ Cannot start night 0: {e}")
            return False

    def start_day_discussion(self) -> bool:
        """Начать дневное обсуждение"""
        try:
            self.state_machine.start_day()
            self.game.phase = GamePhase.DAY_DISCUSSION
            self.game.day_number += 1

            print(f"[GameEngine] ✅ Day {self.game.day_number} discussion started")
            return True

        except TransitionNotAllowed as e:
            print(f"[GameEngine] ❌ Cannot start day: {e}")
            return False

    def start_nominations(self) -> bool:
        """Начать выдвижение кандидатов"""
        try:
            self.state_machine.open_nominations()
            self.game.phase = GamePhase.NOMINATIONS

            print(f"[GameEngine] ✅ Nominations opened")
            return True

        except TransitionNotAllowed as e:
            print(f"[GameEngine] ❌ Cannot start nominations: {e}")
            return False

    def start_voting(self) -> bool:
        """Начать голосование"""
        try:
            self.state_machine.start_voting()
            self.game.phase = GamePhase.VOTING

            print(f"[GameEngine] ✅ Voting started")
            return True

        except TransitionNotAllowed as e:
            print(f"[GameEngine] ❌ Cannot start voting: {e}")
            return False

    def give_last_word(self, player_id: int) -> bool:
        """Дать последнее слово игроку перед выбыванием"""
        try:
            self.state_machine.give_last_word()
            self.game.phase = GamePhase.LAST_WORD
            self.current_speaker_id = player_id

            print(f"[GameEngine] ✅ Last word for player {player_id}")
            return True

        except TransitionNotAllowed as e:
            print(f"[GameEngine] ❌ Cannot give last word: {e}")
            return False

    def start_night(self) -> bool:
        """Начать ночь"""
        try:
            self.state_machine.start_night()
            self.game.phase = GamePhase.NIGHT

            print(f"[GameEngine] ✅ Night started")
            return True

        except TransitionNotAllowed as e:
            print(f"[GameEngine] ❌ Cannot start night: {e}")
            return False

    def end_game(self, winning_team: Optional[str] = None) -> bool:
        """
        Завершить игру

        Args:
            winning_team: Победившая команда ("mafia" или "citizens")
        """
        try:
            self.state_machine.end_game()
            self.game.phase = GamePhase.GAME_END
            self.game.end_time = datetime.now()

            if winning_team:
                print(f"[GameEngine] ✅ Game ended. Winner: {winning_team}")
            else:
                print(f"[GameEngine] ✅ Game ended")

            return True

        except TransitionNotAllowed as e:
            print(f"[GameEngine] ❌ Cannot end game: {e}")
            return False

    # ========== Управление ходами ==========

    def start_turn(self, player_id: int, duration_seconds: int = 60) -> Turn:
        """
        Начать ход игрока

        Args:
            player_id: ID игрока
            duration_seconds: Длительность хода в секундах

        Returns:
            Объект хода
        """
        player = self._get_player_by_id(player_id)
        if not player:
            raise ValueError(f"Player {player_id} not found")

        turn = Turn(
            player_id=PlayerId(player_id),
            phase=self.game.phase,
            start_time=datetime.now(),
            duration_seconds=duration_seconds
        )

        self.current_turn = turn
        self.current_speaker_id = player_id
        self.game.turns.append(turn)

        print(f"[GameEngine] ✅ Turn started for player {player_id} ({duration_seconds}s)")
        return turn

    def end_turn(self) -> bool:
        """Завершить текущий ход"""
        if not self.current_turn:
            print(f"[GameEngine] ⚠️  No active turn to end")
            return False

        self.current_turn.complete(datetime.now())
        self.current_turn = None
        self.current_speaker_id = None

        print(f"[GameEngine] ✅ Turn ended")
        return True

    # ========== Обнаружение фолов ==========

    def check_speech_foul(self, speaker_id: int, timestamp: float) -> Optional[Foul]:
        """
        Проверить фол: говорит не тот игрок

        Args:
            speaker_id: ID игрока который говорит
            timestamp: Время обнаружения (секунды)

        Returns:
            Фол если обнаружен
        """
        # Проверяем что есть активный ход
        if not self.current_turn or not self.current_speaker_id:
            return None

        # Проверяем что говорит правильный игрок
        if speaker_id != self.current_speaker_id:
            foul = self._create_foul(
                player_id=speaker_id,
                foul_type=FoulType.SPEAKING_OUT_OF_TURN,
                description=f"Игрок {speaker_id} говорит не в свой ход (ход игрока {self.current_speaker_id})"
            )
            return foul

        return None

    def check_gesture_foul(self, player_id: int, gesture_type: str) -> Optional[Foul]:
        """
        Проверить фол: жест во время голосования

        Args:
            player_id: ID игрока
            gesture_type: Тип жеста

        Returns:
            Фол если обнаружен
        """
        # Жесты запрещены во время голосования
        if self.game.phase == GamePhase.VOTING:
            foul = self._create_foul(
                player_id=player_id,
                foul_type=FoulType.GESTURING_DURING_VOTE,
                description=f"Жест '{gesture_type}' во время голосования"
            )
            return foul

        return None

    def check_time_foul(self) -> Optional[Foul]:
        """
        Проверить фол: превышение времени хода

        Returns:
            Фол если обнаружен
        """
        if not self.current_turn or not self.current_speaker_id:
            return None

        # Проверяем что ход не истек
        elapsed = (datetime.now() - self.current_turn.start_time).total_seconds()

        if elapsed > self.current_turn.duration_seconds:
            foul = self._create_foul(
                player_id=self.current_speaker_id,
                foul_type=FoulType.TIMEOUT_EXCEEDED,
                description=f"Превышено время хода ({elapsed:.1f}s > {self.current_turn.duration_seconds}s)"
            )
            return foul

        return None

    def add_foul(self, foul: Foul) -> bool:
        """
        Добавить фол игроку

        Args:
            foul: Объект фола

        Returns:
            True если успешно
        """
        player = self._get_player_by_id(foul.player_id.value)
        if not player:
            print(f"[GameEngine] ❌ Player {foul.player_id.value} not found")
            return False

        player.fouls.append(foul)

        print(f"[GameEngine] ⚠️  Foul added: {foul.foul_type.value} for player {foul.player_id.value}")

        # Проверяем количество фолов (3 фола = удаление)
        if len(player.fouls) >= 3:
            player.is_alive = False
            print(f"[GameEngine] ❌ Player {player.id.value} eliminated (3 fouls)")

        return True

    # ========== Вспомогательные методы ==========

    def _create_foul(
        self,
        player_id: int,
        foul_type: FoulType,
        description: str
    ) -> Foul:
        """Создать объект фола"""
        return Foul(
            player_id=PlayerId(player_id),
            foul_type=foul_type,
            timestamp=datetime.now(),
            phase=self.game.phase.value,
            description=description,
            auto_detected=True
        )

    def _get_player_by_id(self, player_id: int) -> Optional[Player]:
        """Получить игрока по ID"""
        for player in self.game.players:
            if player.id.value == player_id:
                return player
        return None

    def get_alive_players(self) -> List[Player]:
        """Получить живых игроков"""
        return [p for p in self.game.players if p.is_alive]

    def get_alive_mafia(self) -> List[Player]:
        """Получить живую мафию"""
        return [
            p for p in self.game.players
            if p.is_alive and p.role in [Role.MAFIA, Role.DON]
        ]

    def get_alive_citizens(self) -> List[Player]:
        """Получить живых мирных"""
        return [
            p for p in self.game.players
            if p.is_alive and p.role not in [Role.MAFIA, Role.DON]
        ]

    def check_win_condition(self) -> Optional[str]:
        """
        Проверить условие победы

        Returns:
            "mafia" если мафия победила
            "citizens" если мирные победили
            None если игра продолжается
        """
        alive_mafia = len(self.get_alive_mafia())
        alive_citizens = len(self.get_alive_citizens())

        # Мафия победила если их >= мирных
        if alive_mafia >= alive_citizens:
            return "mafia"

        # Мирные победили если мафия вся убита
        if alive_mafia == 0:
            return "citizens"

        return None

    def get_game_state(self) -> Dict:
        """
        Получить текущее состояние игры

        Returns:
            Словарь с состоянием
        """
        return {
            "phase": self.game.phase.value,
            "day": self.game.day_number,
            "players_alive": len(self.get_alive_players()),
            "mafia_alive": len(self.get_alive_mafia()),
            "citizens_alive": len(self.get_alive_citizens()),
            "current_speaker_id": self.current_speaker_id,
            "turn_active": self.current_turn is not None,
            "total_fouls": sum(len(p.fouls) for p in self.game.players),
        }
