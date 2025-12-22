# Mafia AI — Система для игры в спортивную мафию

AI-система для автоматизации игры в спортивную мафию с распознаванием лиц, жестов и голоса.

## 🎯 Основные возможности

- 👤 **Распознавание лиц** - автоматическая идентификация игроков через CompreFace
- 🖐️ **Детекция жестов** - отслеживание голосования и действий
- 🎤 **Распознавание речи** - транскрипция высказываний игроков
- ⏱️ **Управление игрой** - таймеры, фазы, правила
- 📊 **Статистика** - аналитика игр и игроков

## 🚀 Быстрый старт

### 1. Запустите все сервисы

```bash
cd mafia-ai
docker-compose up -d
```

Это запустит:
- **Backend API** (FastAPI) - http://localhost:8000
- **CompreFace UI** - http://localhost:8001
- **PostgreSQL** - база данных для лиц
- Все необходимые сервисы CompreFace

### 2. Настройте CompreFace

📖 **[Читайте COMPREFACE_QUICKSTART.md](COMPREFACE_QUICKSTART.md)** - настройка за 5 минут

Кратко:
1. Откройте http://localhost:8001
2. Зарегистрируйтесь
3. Создайте приложение "Mafia AI"
4. Создайте Recognition Service
5. Скопируйте API ключи в `backend/.env`

### 3. Запустите Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

### 4. Запустите Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend будет доступен на http://localhost:5173

## 📋 Архитектура

### Backend (FastAPI)

- **WebSocket** `/ws` - события реального времени
- **REST API** - управление игрой и игроками
- **CompreFace** - распознавание лиц (state-of-the-art)
- **Faster Whisper** - распознавание речи
- **YOLOv8** - детекция жестов
- **Clean Architecture** - domain-driven design

### Frontend (React + Vite + TypeScript)

- Подключение к WebSocket
- Визуализация игрового стола (10 мест)
- Таймеры и управление игрой
- Регистрация игроков
- Статистика

### CompreFace (Face Recognition)

- **Детекция лиц** - высокая точность
- **Распознавание** - state-of-the-art модели
- **UI панель** - управление базой лиц
- **REST API** - простая интеграция

## 🔧 Конфигурация

### Переменные окружения

Создайте `backend/.env`:

```env
# Face Recognition (CompreFace)
FACE_DETECTOR=compreface
COMPREFACE_API_URL=http://compreface-api:8080
COMPREFACE_API_KEY=your-api-key
COMPREFACE_RECOGNITION_KEY=your-recognition-key
FACE_THRESHOLD=0.85

# Server
HOST=0.0.0.0
PORT=8000

# Camera
CAMERA_INDEX=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
```

Полный пример: [backend/.env.example](backend/.env.example)

## 📚 Документация

### CompreFace Integration

- 📖 **[COMPREFACE_QUICKSTART.md](COMPREFACE_QUICKSTART.md)** - Быстрый старт (5 минут)
- 📖 **[backend/COMPREFACE_SETUP.md](backend/COMPREFACE_SETUP.md)** - Полная документация

### Архитектура

- 📖 **[ARCHITECTURE.md](../ARCHITECTURE.md)** - Архитектура проекта
- 📖 **[backend/README_NEW_ARCH.md](backend/README_NEW_ARCH.md)** - Новая архитектура
- 📖 **[backend/README_INTEGRATION.md](backend/README_INTEGRATION.md)** - Интеграция компонентов

## 🧪 Тестирование

### Проверка интеграции CompreFace

```bash
cd backend
python test_compreface_integration.py
```

Этот скрипт проверит:
- ✅ Подключение к CompreFace API
- ✅ Инициализацию детектора
- ✅ ServiceContainer
- ✅ Список зарегистрированных игроков

### Другие тесты

```bash
# Тесты детекторов
python test_detectors.py

# Тесты игрового движка
python test_game_engine.py

# Тесты новой архитектуры
python test_new_arch.py
```

## 🎮 Использование

### 1. Регистрация игроков

Перед игрой зарегистрируйте всех игроков:

```python
from application.services import get_container
import cv2

container = get_container()
manager = container.compreface_manager

# Регистрация игрока
frame = cv2.imread("player_photo.jpg")
await manager.enroll_person_single("Игрок 1", frame)
```

Или через UI: http://localhost:8001

### 2. Игра

Запустите систему - она автоматически:
- Распознает игроков по лицам
- Отслеживает жесты голосования
- Транскрибирует речь
- Управляет таймерами и фазами

### 3. После игры

Опционально очистите базу лиц:

```python
await manager.delete_person("Игрок 1")
```

## 🛠️ Технологии

### Backend
- **FastAPI** - веб-фреймворк
- **CompreFace** - распознавание лиц
- **Faster Whisper** - ASR
- **YOLOv8** - детекция объектов
- **OpenCV** - обработка видео
- **PostgreSQL** - база данных (для CompreFace)

### Frontend
- **React** - UI библиотека
- **TypeScript** - типизация
- **Vite** - сборщик
- **WebSocket** - real-time связь

### Infrastructure
- **Docker Compose** - оркестрация сервисов
- **NGINX** - веб-сервер (в CompreFace)

## 📊 Статус проекта

✅ CompreFace интегрирован  
✅ Детекция лиц работает  
✅ Детекция жестов работает  
✅ Распознавание речи работает  
✅ Игровой движок работает  
✅ WebSocket работает  
✅ Frontend работает  

## 🐛 Troubleshooting

### CompreFace не запускается?

```bash
# Проверьте логи
docker-compose logs compreface-api

# Перезапустите
docker-compose restart compreface-api compreface-core
```

### Backend не подключается к CompreFace?

Проверьте:
1. API ключи в `.env`
2. CompreFace запущен: `docker-compose ps`
3. URL правильный: `http://compreface-api:8080`

### Подробнее

📖 **[backend/COMPREFACE_SETUP.md](backend/COMPREFACE_SETUP.md)** - секция Troubleshooting

## 📞 Поддержка

- 📖 Документация в папке `docs/`
- 🐛 Issues на GitHub
- 📧 Email: support@example.com

## 📄 Лицензия

MIT License

---

**Удачной игры в Мафию!** 🎲🕵️
