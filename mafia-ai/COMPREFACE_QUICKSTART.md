# 🚀 CompreFace Quick Start Guide

## Быстрая интеграция CompreFace за 5 минут

### 1️⃣ Запустите Docker Compose

```bash
cd mafia-ai
docker-compose up -d
```

Подождите 2-3 минуты пока все сервисы запустятся.

### 2️⃣ Настройте CompreFace

1. Откройте CompreFace UI: **http://localhost:8001**

2. **Зарегистрируйтесь:**
   - Email: `admin@mafia.ai` (любой email)
   - Password: `admin123` (любой пароль)
   - First name: `Admin`
   - Last name: `Admin`

3. **Создайте приложение:**
   - Нажмите **"Create Application"**
   - Name: `Mafia AI`
   - Click **"Create"**

4. **Создайте Recognition Service:**
   - В приложении "Mafia AI" нажмите **"Add Recognition Service"**
   - Name: `Mafia Players`
   - Click **"Create"**

5. **Скопируйте API ключи:**
   - Нажмите на иконку глаза 👁️ рядом с **API Key** (верхний)
   - Скопируйте **API Key**
   - Нажмите на иконку глаза 👁️ рядом с **"Mafia Players"** (нижний)
   - Скопируйте **Service API Key**

### 3️⃣ Настройте Backend

Создайте файл `mafia-ai/backend/.env`:

```env
# CompreFace Settings
FACE_DETECTOR=compreface
COMPREFACE_API_URL=http://compreface-api:8080
COMPREFACE_API_KEY=вставьте-ваш-api-key
COMPREFACE_RECOGNITION_KEY=вставьте-ваш-service-key

# Recognition Settings
FACE_THRESHOLD=0.85
COMPREFACE_DET_THRESHOLD=0.8
```

Или скопируйте пример:

```bash
cp backend/.env.example backend/.env
# Отредактируйте файл и добавьте ваши ключи
```

### 4️⃣ Запустите Backend

```bash
cd mafia-ai/backend
pip install -r requirements.txt
uvicorn app:app --reload
```

### 5️⃣ Проверьте работу

```python
# test_compreface.py
import asyncio
import cv2
from application.services import get_container

async def test():
    container = get_container()
    manager = container.compreface_manager
    
    if not manager:
        print("❌ CompreFace не настроен")
        return
    
    # Список игроков
    players = await manager.list_persons()
    print(f"✅ Игроки в базе: {players}")
    
    # Регистрация тестового игрока
    frame = cv2.imread("test_photo.jpg")
    result = await manager.enroll_person_single("Тестовый игрок", frame)
    print(f"✅ Регистрация: {result}")

asyncio.run(test())
```

---

## 🎯 Что дальше?

### Полная документация:
📖 **[Читайте COMPREFACE_SETUP.md](backend/COMPREFACE_SETUP.md)**

Там вы найдете:
- Подробную настройку
- API примеры
- Troubleshooting
- Тонкую настройку порогов
- Использование в игре

---

## ⚡ Основные команды

```bash
# Запустить CompreFace
docker-compose up -d

# Остановить
docker-compose down

# Посмотреть логи
docker-compose logs -f compreface-api

# Перезапустить
docker-compose restart compreface-api compreface-core

# Проверить статус
docker-compose ps
```

---

## 🔧 Troubleshooting

### CompreFace UI не открывается?

```bash
# Проверьте статус сервисов
docker-compose ps

# Если не все "Up", перезапустите
docker-compose down
docker-compose up -d
```

### Backend не подключается?

1. Убедитесь что API ключи правильные
2. Проверьте URL: должен быть `http://compreface-api:8080`
3. Перезапустите backend

### Низкая точность?

Снизьте порог в `.env`:

```env
FACE_THRESHOLD=0.75  # Вместо 0.85
```

---

## 📊 Статус интеграции

✅ CompreFace добавлен в docker-compose  
✅ Python SDK установлен  
✅ CompreFaceDetector создан  
✅ CompreFaceManager создан  
✅ ServiceContainer обновлен  
✅ Конфигурация настроена  
✅ Документация готова  

**Все готово к использованию!** 🎉

---

## 🎮 Использование в игре

1. **Перед игрой**: Зарегистрируйте всех игроков через UI
2. **Во время игры**: Система автоматически распознает лица
3. **После игры**: Опционально очистите базу

**Удачной игры! 🎲**




