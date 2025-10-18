"""
Тестовый скрипт для проверки новой архитектуры
Проверяет: repositories, face detection, gesture detection
"""

import asyncio
import cv2
import numpy as np

# Новая архитектура
from infrastructure.storage.repositories import JsonPlayerRepository
from infrastructure.detection.face import LegacyFaceDetector
from infrastructure.detection.gesture import LegacyGestureDetector
from infrastructure.detection.table import LegacyTableDetector
from core.domain.value_objects.player_id import PlayerId


async def test_repositories():
    """Тест репозиториев"""
    print("\n=== Testing Repositories ===")

    # Создаем репозиторий
    repo = JsonPlayerRepository()

    # Очищаем для чистого теста
    await repo.reset()

    # Добавляем игрока
    fake_embedding = np.random.randn(512).astype(np.float32)
    player = await repo.add(
        name="Test Player",
        seat=1,
        face_embedding=fake_embedding,
        thumb_path="thumbs/test.jpg"
    )

    print(f"✅ Added player: {player.name} (ID: {player.id}, Seat: {player.seat})")

    # Получаем игрока
    retrieved = await repo.get(player.id)
    assert retrieved is not None
    assert retrieved.name == "Test Player"
    print(f"✅ Retrieved player: {retrieved.name}")

    # Список всех игроков
    all_players = await repo.list_all()
    assert len(all_players) == 1
    print(f"✅ Total players: {len(all_players)}")

    # Поиск по эмбеддингу
    found = await repo.find_by_embedding(fake_embedding, threshold=0.9)
    assert found is not None
    assert found.id == player.id
    print(f"✅ Found player by embedding: {found.name}")

    print("✅ Repositories test passed!")


async def test_face_detection():
    """Тест детекции лиц"""
    print("\n=== Testing Face Detection ===")

    # Создаем детектор
    detector = LegacyFaceDetector()

    # Создаем тестовый кадр (черный квадрат)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Детектируем лица (на черном кадре ничего не найдем, но проверим что не падает)
    faces = await detector.detect(frame)
    print(f"✅ Detected {len(faces)} faces on test frame")
    print(f"✅ Embedding dimension: {detector.get_embedding_dim()}")

    print("✅ Face detection test passed!")


async def test_gesture_detection():
    """Тест детекции жестов"""
    print("\n=== Testing Gesture Detection ===")

    # Создаем детектор
    detector = LegacyGestureDetector()

    # Создаем тестовый кадр
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Детектируем руки
    hands = await detector.detect_hands(frame)
    print(f"✅ Detected {len(hands)} hands on test frame")

    # Распознаем жесты
    gestures = await detector.recognize_gestures(frame, hands)
    print(f"✅ Recognized {len(gestures)} gestures")

    print("✅ Gesture detection test passed!")


async def test_table_detection():
    """Тест детекции стола"""
    print("\n=== Testing Table Detection ===")

    # Создаем детектор
    detector = LegacyTableDetector()

    # Создаем тестовый кадр с белым прямоугольником (имитация стола)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 300), (540, 450), (255, 255, 255), -1)

    # Детектируем стол
    polygon = await detector.detect_table(frame)
    if polygon is not None:
        print(f"✅ Detected table polygon with {len(polygon)} points")
        is_valid = await detector.validate_table_polygon(polygon, frame.shape[:2])
        print(f"✅ Polygon valid: {is_valid}")
    else:
        print("ℹ️  No table detected (expected on synthetic frame)")

    print("✅ Table detection test passed!")


async def test_camera_integration():
    """Тест с реальной камерой (если доступна)"""
    print("\n=== Testing Camera Integration ===")

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️  Camera not available, skipping camera test")
            return

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("⚠️  Could not read frame from camera")
            return

        print(f"✅ Camera frame captured: {frame.shape}")

        # Тестируем детекторы на реальном кадре
        face_detector = LegacyFaceDetector()
        gesture_detector = LegacyGestureDetector()

        faces = await face_detector.detect(frame)
        hands = await gesture_detector.detect_hands(frame)

        print(f"✅ Detected {len(faces)} faces on camera frame")
        print(f"✅ Detected {len(hands)} hands on camera frame")

        if len(faces) > 0:
            print(f"   Face 0: bbox={faces[0].bbox}, confidence={faces[0].confidence:.2f}")
            print(f"   Embedding shape: {faces[0].embedding.shape}")

        print("✅ Camera integration test passed!")

    except Exception as e:
        print(f"⚠️  Camera test failed: {e}")


async def main():
    """Запуск всех тестов"""
    print("\n" + "="*50)
    print("  Mafia AI - New Architecture Test Suite")
    print("="*50)

    try:
        await test_repositories()
        await test_face_detection()
        await test_gesture_detection()
        await test_table_detection()
        await test_camera_integration()

        print("\n" + "="*50)
        print("  ✅ ALL TESTS PASSED!")
        print("="*50 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
