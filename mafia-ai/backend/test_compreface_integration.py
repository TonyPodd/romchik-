"""Тест интеграции CompreFace"""

import asyncio
import sys
from pathlib import Path

# Добавляем backend в путь
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Settings
from infrastructure.detection.face.compreface_detector import CompreFaceDetector
from infrastructure.detection.face.compreface_manager import CompreFaceManager


async def test_compreface_connection():
    """Тест подключения к CompreFace"""
    print("=" * 60)
    print("🧪 Тест подключения к CompreFace")
    print("=" * 60)
    
    settings = Settings()
    
    print(f"\n📋 Конфигурация:")
    print(f"  Face Detector: {settings.face_detector_type}")
    print(f"  API URL: {settings.compreface_api_url}")
    print(f"  API Key: {'✅ Установлен' if settings.compreface_api_key else '❌ Не установлен'}")
    print(f"  Recognition Key: {'✅ Установлен' if settings.compreface_recognition_key else '❌ Не установлен'}")
    print(f"  Detection Threshold: {settings.compreface_det_threshold}")
    print(f"  Recognition Threshold: {settings.face_recognition_threshold}")
    
    if not settings.compreface_api_key or not settings.compreface_recognition_key:
        print("\n❌ ОШИБКА: API ключи не установлены!")
        print("\n📝 Инструкция:")
        print("  1. Откройте http://localhost:8001")
        print("  2. Создайте приложение и Recognition Service")
        print("  3. Скопируйте API ключи")
        print("  4. Добавьте их в файл .env:")
        print("     COMPREFACE_API_KEY=ваш-api-key")
        print("     COMPREFACE_RECOGNITION_KEY=ваш-recognition-key")
        return False
    
    try:
        # Инициализация детектора
        print("\n🔧 Инициализация CompreFace детектора...")
        detector = CompreFaceDetector(
            api_url=settings.compreface_api_url,
            api_key=settings.compreface_api_key,
            recognition_service_key=settings.compreface_recognition_key,
            det_prob_threshold=settings.compreface_det_threshold,
        )
        
        if not detector.is_initialized:
            print("❌ CompreFace не инициализирован")
            print("   Проверьте что CompreFace запущен: docker-compose ps")
            return False
        
        print("✅ CompreFace детектор инициализирован")
        
        # Инициализация менеджера
        print("\n🔧 Инициализация CompreFace менеджера...")
        manager = CompreFaceManager(
            detector=detector,
            recognition_threshold=settings.face_recognition_threshold,
            min_enrollment_samples=settings.compreface_min_enrollment,
        )
        print("✅ CompreFace менеджер инициализирован")
        
        # Получение списка игроков
        print("\n👥 Получение списка зарегистрированных игроков...")
        players = await manager.list_persons()
        
        if players:
            print(f"✅ Найдено игроков: {len(players)}")
            for i, player in enumerate(players, 1):
                print(f"   {i}. {player}")
        else:
            print("ℹ️  База игроков пуста (это нормально для первого запуска)")
        
        print("\n" + "=" * 60)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("=" * 60)
        print("\n📖 Следующие шаги:")
        print("  1. Зарегистрируйте игроков через API или UI")
        print("  2. Используйте manager.enroll_person_single() для регистрации")
        print("  3. Используйте manager.recognize_faces() для распознавания")
        print("\n📚 Документация: backend/COMPREFACE_SETUP.md")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        print("\n🔧 Возможные решения:")
        print("  1. Проверьте что CompreFace запущен:")
        print("     docker-compose ps")
        print("  2. Проверьте API ключи в .env")
        print("  3. Проверьте логи:")
        print("     docker-compose logs compreface-api")
        return False


async def test_service_container():
    """Тест ServiceContainer"""
    print("\n" + "=" * 60)
    print("🧪 Тест ServiceContainer")
    print("=" * 60)
    
    try:
        from application.services import get_container
        
        print("\n🔧 Получение ServiceContainer...")
        container = get_container()
        print("✅ ServiceContainer получен")
        
        print("\n🔧 Получение face detector...")
        face_detector = container.face_detector
        print(f"✅ Face detector: {type(face_detector).__name__}")
        
        print("\n🔧 Получение CompreFace manager...")
        manager = container.compreface_manager
        
        if manager:
            print(f"✅ CompreFace manager: {type(manager).__name__}")
            
            # Проверяем список игроков
            players = await manager.list_persons()
            print(f"✅ Подключение к API работает (игроков: {len(players)})")
        else:
            print("ℹ️  CompreFace manager не активен (face_detector_type != 'compreface')")
        
        return True
        
    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Главная функция тестирования"""
    print("\n🚀 Тестирование интеграции CompreFace")
    print("=" * 60)
    
    # Тест подключения
    test1 = await test_compreface_connection()
    
    # Тест ServiceContainer
    test2 = await test_service_container()
    
    # Итоговый результат
    print("\n" + "=" * 60)
    if test1 and test2:
        print("🎉 ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ!")
        print("=" * 60)
        print("\n✅ CompreFace готов к использованию!")
        print("\n📖 Читайте документацию:")
        print("   - COMPREFACE_QUICKSTART.md - быстрый старт")
        print("   - backend/COMPREFACE_SETUP.md - полная документация")
    else:
        print("❌ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
        print("=" * 60)
        print("\n📖 Читайте COMPREFACE_QUICKSTART.md для настройки")


if __name__ == "__main__":
    asyncio.run(main())

