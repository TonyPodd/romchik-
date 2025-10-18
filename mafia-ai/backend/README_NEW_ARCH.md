# Новая архитектура Mafia AI

## Что сделано

### ✅ Domain Layer (core/)
- **Entities**: Player, Game, Turn, Foul
- **Value Objects**: GamePhase, Role, FoulType, PlayerId
- **Interfaces**: IPlayerRepository, IGameRepository, IFaceDetector, IGestureDetector, ITableDetector, IAudioProcessor

### ✅ Infrastructure Layer (infrastructure/)
- **Repositories**: JsonPlayerRepository, JsonGameRepository (работают с текущим JSON storage)
- **Detectors**:
  - LegacyFaceDetector (обертка над текущим ArcFace/MediaPipe)
  - LegacyGestureDetector (обертка над текущим GestureDetector)
  - LegacyTableDetector (обертка над автодетекцией стола)

### ✅ Configuration (config/)
- Pydantic Settings с поддержкой .env файлов

## Как протестировать

### 1. Установить зависимости

```bash
cd mafia-ai/backend
pip install -r requirements.txt
```

**Примечание**: Некоторые пакеты могут не установиться с первого раза:
- `insightface==0.7.3` - может требовать дополнительных зависимостей
- `pyaudio==0.2.14` - опциональный, нужен только для микрофона
- `faster-whisper==1.0.3` - пока не используется в тестах

Минимальные зависимости для теста:
```bash
pip install fastapi uvicorn pydantic pydantic-settings opencv-python numpy mediapipe
```

### 2. Запустить тесты

```bash
cd mafia-ai/backend
python test_new_arch.py
```

### Что тестирует скрипт

1. **Repositories Test**:
   - Создание/получение/поиск игроков
   - Поиск по face embedding (cosine similarity)
   - Работа с JSON storage

2. **Face Detection Test**:
   - Инициализация детектора
   - Проверка размерности эмбеддингов
   - Детекция на синтетическом кадре

3. **Gesture Detection Test**:
   - Детекция рук
   - Распознавание жестов
   - Проверка на синтетическом кадре

4. **Table Detection Test**:
   - Автодетекция контура стола
   - Валидация полигона

5. **Camera Integration Test** (если камера доступна):
   - Захват кадра с камеры
   - Детекция лиц на реальном кадре
   - Детекция рук на реальном кадре

### Ожидаемый вывод

```
==================================================
  Mafia AI - New Architecture Test Suite
==================================================

=== Testing Repositories ===
✅ Added player: Test Player (ID: 1, Seat: 1)
✅ Retrieved player: Test Player
✅ Total players: 1
✅ Found player by embedding: Test Player
✅ Repositories test passed!

=== Testing Face Detection ===
[LegacyFaceDetector] initialized with backend: _FaceBackendONNX
✅ Detected 0 faces on test frame
✅ Embedding dimension: 512
✅ Face detection test passed!

=== Testing Gesture Detection ===
[LegacyGestureDetector] initialized
✅ Detected 0 hands on test frame
✅ Recognized 0 gestures
✅ Gesture detection test passed!

=== Testing Table Detection ===
[LegacyTableDetector] initialized
ℹ️  No table detected (expected on synthetic frame)
✅ Table detection test passed!

=== Testing Camera Integration ===
✅ Camera frame captured: (720, 1280, 3)
✅ Detected N faces on camera frame
✅ Detected M hands on camera frame
✅ Camera integration test passed!

==================================================
  ✅ ALL TESTS PASSED!
==================================================
```

## Что дальше

После успешного теста можно:

1. **Улучшить детекторы**:
   - Заменить LegacyFaceDetector на YOLOv8FaceDetector
   - Добавить InsightFace для более точного распознавания
   - Улучшить gesture detection

2. **Добавить game engine**:
   - State machine для игры
   - Use cases (start game, process turn, detect fouls)
   - Event bus для событий

3. **Переписать app.py**:
   - Использовать DI для зависимостей
   - Подключить новые repositories
   - Интегрировать detectors

4. **Добавить audio pipeline**:
   - VAD (Silero)
   - ASR (Faster-Whisper)
   - Speaker identification

## Структура проекта

```
backend/
├── core/                       # Domain Layer
│   ├── domain/
│   │   ├── entities/          # Player, Game, Turn, Foul
│   │   └── value_objects/     # GamePhase, Role, etc
│   └── interfaces/            # Абстракции
│       ├── repositories.py    # IPlayerRepository, IGameRepository
│       ├── detectors.py       # IFaceDetector, IGestureDetector
│       └── audio_processor.py # IAudioProcessor
│
├── infrastructure/            # Infrastructure Layer
│   ├── storage/
│   │   └── repositories/      # JsonPlayerRepository, JsonGameRepository
│   └── detection/
│       ├── face/             # LegacyFaceDetector
│       ├── gesture/          # LegacyGestureDetector
│       └── table/            # LegacyTableDetector
│
├── config/                    # Configuration
│   └── settings.py           # Pydantic Settings
│
├── utils/                     # Utilities
│   └── async_utils.py
│
└── test_new_arch.py          # Тестовый скрипт
```

## Принципы новой архитектуры

1. **Dependency Inversion**: Domain не зависит от Infrastructure
2. **Single Responsibility**: Каждый модуль делает одну вещь
3. **Interface Segregation**: Маленькие, фокусированные интерфейсы
4. **Open/Closed**: Легко добавлять новые реализации без изменения domain

## Миграция на SQL (в будущем)

Благодаря Repository Pattern, миграция на SQL будет простой:

```python
# Вместо
repo = JsonPlayerRepository()

# Просто замените на
repo = SqlPlayerRepository()  # Реализация IPlayerRepository для SQL
```

Domain layer останется без изменений!

## Troubleshooting

### Ошибка: ModuleNotFoundError

Убедитесь что вы запускаете скрипт из директории `backend/`:
```bash
cd mafia-ai/backend
python test_new_arch.py
```

### Ошибка: Camera not available

Это нормально если у вас нет камеры. Тест просто пропустит этот шаг.

### Ошибка: ArcFace model download failed

Скрипт попытается скачать модель ArcFace (~40MB). Если скачивание не удалось,
детектор автоматически переключится на MediaPipe Landmarks (fallback).

### Ошибка: Import error for insightface

InsightFace пока не используется в legacy детекторах. Можно не устанавливать.

---

**Автор**: Claude Code
**Дата**: 2025-10-19
**Версия**: 2.0
