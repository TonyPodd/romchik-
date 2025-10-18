"""Game API Router - управление игрой"""

from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from application.game_engine import GameEngine
from application.services import get_container, ServiceContainer
from core.domain.entities.game import Game
from core.domain.value_objects.role import Role
from core.domain.value_objects.game_phase import GamePhase


router = APIRouter(prefix="/api/game", tags=["game"])


# ========== Pydantic Models ==========

class GameStateResponse(BaseModel):
    """Текущее состояние игры"""
    phase: str
    day: int
    players_alive: int
    mafia_alive: int
    citizens_alive: int
    current_speaker_id: Optional[int]
    turn_active: bool
    total_fouls: int


class StartGameRequest(BaseModel):
    """Запрос на начало игры"""
    player_ids: List[int]


class AssignRolesRequest(BaseModel):
    """Назначение ролей"""
    role_assignments: Dict[int, str]  # {player_id: role}


class StartTurnRequest(BaseModel):
    """Начать ход игрока"""
    player_id: int
    duration_seconds: int = 60


class LastWordRequest(BaseModel):
    """Дать последнее слово"""
    player_id: int


class CheckSpeechFoulRequest(BaseModel):
    """Проверить фол речи"""
    speaker_id: int


class CheckGestureFoulRequest(BaseModel):
    """Проверить фол жеста"""
    player_id: int
    gesture_type: str


# ========== Global State ==========

# Текущая активная игра (в production можно хранить в Redis/DB)
_current_game: Optional[Game] = None
_current_engine: Optional[GameEngine] = None


def get_game_engine() -> GameEngine:
    """Dependency для получения текущего engine"""
    global _current_engine
    if _current_engine is None:
        raise HTTPException(status_code=400, detail="No active game")
    return _current_engine


# ========== Endpoints ==========

@router.get("/status", response_model=GameStateResponse)
async def get_game_status():
    """
    Получить текущее состояние игры

    Returns:
        Состояние игры или сообщение если игра не активна
    """
    global _current_engine

    if _current_engine is None:
        # Нет активной игры
        return GameStateResponse(
            phase="idle",
            day=0,
            players_alive=0,
            mafia_alive=0,
            citizens_alive=0,
            current_speaker_id=None,
            turn_active=False,
            total_fouls=0
        )

    state = _current_engine.get_game_state()
    return GameStateResponse(**state)


@router.post("/start")
async def start_game(
    request: StartGameRequest,
    container: ServiceContainer = Depends(get_container)
):
    """
    Начать новую игру

    Args:
        request: Список ID игроков

    Returns:
        Статус начала игры
    """
    global _current_game, _current_engine

    # Получаем игроков из repository
    player_repo = container.player_repository
    players = []

    for player_id in request.player_ids:
        player = await player_repo.get(player_id)
        if not player:
            raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
        players.append(player)

    # Создаем новую игру
    game = Game()
    engine = GameEngine(game)

    # Запускаем игру
    success = engine.start_game(players)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to start game")

    # Сохраняем в глобальном состоянии
    _current_game = game
    _current_engine = engine

    # Сохраняем в repository
    game_repo = container.game_repository
    await game_repo.add(game)

    return {
        "success": True,
        "game_id": game.id,
        "players_count": len(players),
        "phase": game.phase.value
    }


@router.post("/assign-roles")
async def assign_roles(
    request: AssignRolesRequest,
    engine: GameEngine = Depends(get_game_engine)
):
    """
    Назначить роли игрокам

    Args:
        request: Словарь {player_id: role}

    Returns:
        Статус назначения ролей
    """
    # Конвертируем строки в Role enum
    role_assignments = {}
    for player_id_str, role_str in request.role_assignments.items():
        try:
            player_id = int(player_id_str)
            role = Role(role_str)
            role_assignments[player_id] = role
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid role or player_id: {e}")

    success = engine.assign_roles(role_assignments)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to assign roles")

    return {
        "success": True,
        "phase": engine.game.phase.value
    }


@router.post("/phase/night-zero")
async def start_night_zero(engine: GameEngine = Depends(get_game_engine)):
    """Начать нулевую ночь"""
    success = engine.start_night_zero()
    if not success:
        raise HTTPException(status_code=400, detail="Cannot start night zero")
    return {"success": True, "phase": engine.game.phase.value}


@router.post("/phase/day")
async def start_day(engine: GameEngine = Depends(get_game_engine)):
    """Начать дневное обсуждение"""
    success = engine.start_day_discussion()
    if not success:
        raise HTTPException(status_code=400, detail="Cannot start day")
    return {"success": True, "phase": engine.game.phase.value, "day": engine.game.day_number}


@router.post("/phase/nominations")
async def start_nominations(engine: GameEngine = Depends(get_game_engine)):
    """Начать выдвижение кандидатов"""
    success = engine.start_nominations()
    if not success:
        raise HTTPException(status_code=400, detail="Cannot start nominations")
    return {"success": True, "phase": engine.game.phase.value}


@router.post("/phase/voting")
async def start_voting(engine: GameEngine = Depends(get_game_engine)):
    """Начать голосование"""
    success = engine.start_voting()
    if not success:
        raise HTTPException(status_code=400, detail="Cannot start voting")
    return {"success": True, "phase": engine.game.phase.value}


@router.post("/phase/last-word")
async def give_last_word(
    request: LastWordRequest,
    engine: GameEngine = Depends(get_game_engine)
):
    """Дать последнее слово игроку"""
    success = engine.give_last_word(request.player_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot give last word")
    return {"success": True, "phase": engine.game.phase.value}


@router.post("/phase/night")
async def start_night(engine: GameEngine = Depends(get_game_engine)):
    """Начать ночь"""
    success = engine.start_night()
    if not success:
        raise HTTPException(status_code=400, detail="Cannot start night")
    return {"success": True, "phase": engine.game.phase.value}


@router.post("/turn/start")
async def start_turn(
    request: StartTurnRequest,
    engine: GameEngine = Depends(get_game_engine)
):
    """Начать ход игрока"""
    try:
        turn = engine.start_turn(request.player_id, request.duration_seconds)
        return {
            "success": True,
            "player_id": request.player_id,
            "duration_seconds": request.duration_seconds,
            "started_at": turn.start_time.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/turn/end")
async def end_turn(engine: GameEngine = Depends(get_game_engine)):
    """Завершить текущий ход"""
    success = engine.end_turn()
    if not success:
        raise HTTPException(status_code=400, detail="No active turn to end")
    return {"success": True}


@router.post("/fouls/check-speech")
async def check_speech_foul(
    request: CheckSpeechFoulRequest,
    engine: GameEngine = Depends(get_game_engine)
):
    """Проверить фол: говорит не тот игрок"""
    import time
    foul = engine.check_speech_foul(request.speaker_id, time.time())

    if foul:
        engine.add_foul(foul)
        return {
            "foul_detected": True,
            "player_id": foul.player_id.value,
            "foul_type": foul.foul_type.value,
            "description": foul.description
        }

    return {"foul_detected": False}


@router.post("/fouls/check-gesture")
async def check_gesture_foul(
    request: CheckGestureFoulRequest,
    engine: GameEngine = Depends(get_game_engine)
):
    """Проверить фол: жест во время голосования"""
    foul = engine.check_gesture_foul(request.player_id, request.gesture_type)

    if foul:
        engine.add_foul(foul)
        return {
            "foul_detected": True,
            "player_id": foul.player_id.value,
            "foul_type": foul.foul_type.value,
            "description": foul.description
        }

    return {"foul_detected": False}


@router.post("/fouls/check-time")
async def check_time_foul(engine: GameEngine = Depends(get_game_engine)):
    """Проверить фол: превышение времени"""
    foul = engine.check_time_foul()

    if foul:
        engine.add_foul(foul)
        return {
            "foul_detected": True,
            "player_id": foul.player_id.value,
            "foul_type": foul.foul_type.value,
            "description": foul.description
        }

    return {"foul_detected": False}


@router.get("/win-condition")
async def check_win_condition(engine: GameEngine = Depends(get_game_engine)):
    """Проверить условие победы"""
    winner = engine.check_win_condition()

    if winner:
        return {
            "game_over": True,
            "winner": winner,
            "mafia_alive": len(engine.get_alive_mafia()),
            "citizens_alive": len(engine.get_alive_citizens())
        }

    return {
        "game_over": False,
        "winner": None,
        "mafia_alive": len(engine.get_alive_mafia()),
        "citizens_alive": len(engine.get_alive_citizens())
    }


@router.post("/end")
async def end_game(
    container: ServiceContainer = Depends(get_container),
    engine: GameEngine = Depends(get_game_engine)
):
    """Завершить игру"""
    global _current_game, _current_engine

    # Проверяем победителя
    winner = engine.check_win_condition()

    # Завершаем игру
    success = engine.end_game(winning_team=winner)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to end game")

    # Сохраняем финальное состояние
    game_repo = container.game_repository
    await game_repo.update(engine.game)

    # Очищаем глобальное состояние
    result = {
        "success": True,
        "game_id": engine.game.id,
        "winner": winner,
        "total_days": engine.game.day_number,
        "total_fouls": sum(len(p.fouls) for p in engine.game.players),
        "duration_seconds": (engine.game.end_time - engine.game.start_time).total_seconds() if engine.game.end_time and engine.game.start_time else 0
    }

    _current_game = None
    _current_engine = None

    return result


@router.post("/reset")
async def reset_game():
    """Сбросить текущую игру (для тестов и отладки)"""
    global _current_game, _current_engine

    _current_game = None
    _current_engine = None

    return {"success": True, "message": "Game reset"}
