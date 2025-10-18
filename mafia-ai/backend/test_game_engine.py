"""Тест Game Engine"""

import asyncio
import time
from datetime import datetime

from application.game_engine import GameEngine
from core.domain.entities.game import Game
from core.domain.entities.player import Player
from core.domain.value_objects.player_id import PlayerId
from core.domain.value_objects.role import Role
from core.domain.value_objects.game_phase import GamePhase
import numpy as np


def create_test_players(count: int = 10) -> list[Player]:
    """Создать тестовых игроков"""
    players = []
    for i in range(1, count + 1):
        player = Player(
            id=PlayerId(i),
            name=f"Игрок {i}",
            seat=i,
            face_embedding=np.random.randn(512).astype(np.float32)
        )
        players.append(player)
    return players


def test_game_flow():
    """Тест полного игрового цикла"""
    print("\n" + "="*60)
    print("  Testing Game Engine - Full Game Flow")
    print("="*60)

    # Создаем игру
    game = Game()  # ID генерируется автоматически
    engine = GameEngine(game)

    # Проверяем начальное состояние
    print("\n1️⃣  Initial state:")
    state = engine.get_game_state()
    print(f"   Phase: {state['phase']}")
    print(f"   Day: {state['day']}")
    assert state['phase'] == GamePhase.IDLE.value

    # Создаем игроков
    players = create_test_players(10)
    print(f"\n2️⃣  Created {len(players)} players")

    # Начинаем игру
    print("\n3️⃣  Starting game...")
    success = engine.start_game(players)
    assert success, "Failed to start game"
    state = engine.get_game_state()
    assert state['phase'] == GamePhase.SETUP.value
    print(f"   ✅ Game started, phase: {state['phase']}")

    # Назначаем роли
    print("\n4️⃣  Assigning roles...")
    role_assignments = {
        1: Role.DON,
        2: Role.MAFIA,
        3: Role.MAFIA,
        4: Role.SHERIFF,
        5: Role.CIVILIAN,
        6: Role.CIVILIAN,
        7: Role.CIVILIAN,
        8: Role.CIVILIAN,
        9: Role.CIVILIAN,
        10: Role.CIVILIAN,
    }
    success = engine.assign_roles(role_assignments)
    assert success, "Failed to assign roles"
    state = engine.get_game_state()
    assert state['phase'] == GamePhase.ROLE_ASSIGNMENT.value
    print(f"   ✅ Roles assigned")
    print(f"   Mafia: 3 (players 1, 2, 3)")
    print(f"   Sheriff: 1 (player 4)")
    print(f"   Citizens: 6 (players 5-10)")

    # Нулевая ночь
    print("\n5️⃣  Starting night 0...")
    success = engine.start_night_zero()
    assert success, "Failed to start night 0"
    state = engine.get_game_state()
    assert state['phase'] == GamePhase.NIGHT_0.value
    print(f"   ✅ Night 0 started")

    # Первый день
    print("\n6️⃣  Starting day 1...")
    success = engine.start_day_discussion()
    assert success, "Failed to start day"
    state = engine.get_game_state()
    assert state['phase'] == GamePhase.DAY_DISCUSSION.value
    assert state['day'] == 1
    print(f"   ✅ Day 1 discussion started")

    # Тестируем ходы
    print("\n7️⃣  Testing turns...")

    # Ход игрока 1
    turn1 = engine.start_turn(player_id=1, duration_seconds=60)
    state = engine.get_game_state()
    assert state['turn_active'] == True
    assert state['current_speaker_id'] == 1
    print(f"   ✅ Turn started for player 1 (60s)")

    # Симулируем речь
    time.sleep(0.1)

    # Проверяем фол: говорит другой игрок
    print("\n8️⃣  Testing speech foul detection...")
    foul = engine.check_speech_foul(speaker_id=2, timestamp=time.time())
    assert foul is not None, "Should detect speech out of turn"
    print(f"   ✅ Detected foul: player 2 speaking during player 1's turn")

    engine.add_foul(foul)
    assert sum(len(p.fouls) for p in game.players) == 1
    assert len(players[1].fouls) == 1
    print(f"   ✅ Foul added to player 2")

    # Завершаем ход
    success = engine.end_turn()
    assert success, "Failed to end turn"
    state = engine.get_game_state()
    assert state['turn_active'] == False
    print(f"   ✅ Turn ended")

    # Выдвижение кандидатов
    print("\n9️⃣  Testing nominations...")
    success = engine.start_nominations()
    assert success, "Failed to start nominations"
    state = engine.get_game_state()
    assert state['phase'] == GamePhase.NOMINATIONS.value
    print(f"   ✅ Nominations opened")

    # Голосование
    print("\n🔟 Testing voting...")
    success = engine.start_voting()
    assert success, "Failed to start voting"
    state = engine.get_game_state()
    assert state['phase'] == GamePhase.VOTING.value
    print(f"   ✅ Voting started")

    # Проверяем фол: жест во время голосования
    print("\n1️⃣1️⃣  Testing gesture foul during voting...")
    foul = engine.check_gesture_foul(player_id=5, gesture_type="finger_on_table")
    assert foul is not None, "Should detect gesture during voting"
    print(f"   ✅ Detected foul: gesture during voting")

    engine.add_foul(foul)
    assert sum(len(p.fouls) for p in game.players) == 2
    print(f"   ✅ Foul added to player 5")

    # Последнее слово
    print("\n1️⃣2️⃣  Testing last word...")
    success = engine.give_last_word(player_id=5)
    assert success, "Failed to give last word"
    state = engine.get_game_state()
    assert state['phase'] == GamePhase.LAST_WORD.value
    print(f"   ✅ Last word given to player 5")

    # Ночь
    print("\n1️⃣3️⃣  Testing night transition...")
    success = engine.start_night()
    assert success, "Failed to start night"
    state = engine.get_game_state()
    assert state['phase'] == GamePhase.NIGHT.value
    print(f"   ✅ Night started")

    # Проверяем условие победы
    print("\n1️⃣4️⃣  Testing win condition...")
    winner = engine.check_win_condition()
    assert winner is None, "Game should continue (no winner yet)"
    print(f"   ✅ No winner yet (game continues)")
    print(f"   Mafia alive: {state['mafia_alive']}")
    print(f"   Citizens alive: {state['citizens_alive']}")

    # Симулируем победу мирных (убиваем всю мафию)
    print("\n1️⃣5️⃣  Simulating citizens victory...")
    for player in engine.get_alive_mafia():
        player.is_alive = False
        print(f"   Eliminated player {player.id.value} (mafia)")

    winner = engine.check_win_condition()
    assert winner == "citizens", "Citizens should win"
    print(f"   ✅ Citizens win!")

    # Завершаем игру
    print("\n1️⃣6️⃣  Ending game...")
    success = engine.end_game(winning_team="citizens")
    assert success, "Failed to end game"
    state = engine.get_game_state()
    assert state['phase'] == GamePhase.GAME_END.value
    print(f"   ✅ Game ended")

    # Итоговая статистика
    print("\n" + "="*60)
    print("  Game Statistics")
    print("="*60)
    print(f"   Total fouls: {sum(len(p.fouls) for p in game.players)}")
    print(f"   Total turns: {len(game.turns)}")
    print(f"   Final day: {game.day_number}")
    print(f"   Duration: {(game.end_time - game.start_time).total_seconds():.2f}s")
    print(f"   Winner: {winner}")

    print("\n✅ ALL TESTS PASSED!")


def test_multiple_fouls():
    """Тест системы фолов (3 фола = удаление)"""
    print("\n" + "="*60)
    print("  Testing Foul System (3 fouls = elimination)")
    print("="*60)

    game = Game()  # ID генерируется автоматически
    engine = GameEngine(game)
    players = create_test_players(10)
    engine.start_game(players)

    player_id = 1
    player = players[0]

    print(f"\nPlayer {player_id} status:")
    print(f"   Alive: {player.is_alive}")
    print(f"   Fouls: {len(player.fouls)}")

    # Добавляем 3 фола
    for i in range(3):
        from core.domain.value_objects.foul_type import FoulType

        foul = engine._create_foul(
            player_id=player_id,
            foul_type=FoulType.SPEAKING_OUT_OF_TURN,
            description=f"Test foul {i+1}"
        )

        engine.add_foul(foul)
        print(f"\n   Foul {i+1} added")
        print(f"   Total fouls: {len(player.fouls)}")
        print(f"   Alive: {player.is_alive}")

    assert len(player.fouls) == 3, "Should have 3 fouls"
    assert player.is_alive == False, "Should be eliminated after 3 fouls"

    print(f"\n✅ Player {player_id} eliminated after 3 fouls")


if __name__ == "__main__":
    try:
        test_game_flow()
        test_multiple_fouls()

        print("\n" + "="*60)
        print("  ✅ ALL GAME ENGINE TESTS PASSED!")
        print("="*60)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
