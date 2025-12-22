# Интеграция CompreFace - Руководство по настройке

## 📋 Содержание

- [Что такое CompreFace?](#что-такое-compreface)
- [Преимущества](#преимущества)
- [Быстрый старт](#быстрый-старт)
- [Подробная настройка](#подробная-настройка)
- [Использование API](#использование-api)
- [Миграция с InsightFace](#миграция-с-insightface)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Что такое CompreFace?

**CompreFace** - это бесплатная open-source система распознавания лиц, разработанная Exadel Inc.

### Основные возможности:
- 🔍 **Детекция лиц** - обнаружение лиц на изображениях
- 👤 **Распознавание** - идентификация людей по базе
- ✅ **Верификация** - сравнение двух лиц
- 🎯 **Landmarks** - определение ключевых точек лица
- 📊 **Возраст/пол** - определение демографических характеристик
- 😷 **Маски** - детекция медицинских масок

**GitHub:** https://github.com/exadel-inc/CompreFace

---

## 🚀 Преимущества

### Почему CompreFace лучше встроенного решения (YOLOv8 + InsightFace)?

| Критерий | CompreFace | YOLOv8 + InsightFace |
|----------|------------|----------------------|
| **Точность** | ⭐⭐⭐⭐⭐ SOTA | ⭐⭐⭐⭐ Хорошая |
| **Настройка** | ✅ Простая (REST API) | ⚠️ Сложная (модели, onnx) |
| **UI** | ✅ Встроенный админ-панель | ❌ Нет |
| **Управление базой** | ✅ API + UI | ⚠️ Только код |
| **Документация** | ✅ Отличная | ⚠️ Разрозненная |
| **Поддержка** | ✅ Активная (7.1k ⭐) | ⚠️ Комьюнити |
| **Масштабируемость** | ✅ Docker-native | ⚠️ Требует настройки |

### Производительность:
- **Детекция**: ~50-100ms на кадр (CPU)
- **Распознавание**: ~100-200ms на кадр (CPU)
- **GPU**: в 5-10 раз быстрее

---

## ⚡ Быстрый старт

### Шаг 1: Запуск CompreFace

CompreFace уже добавлен в `docker-compose.yml`:

```bash
cd mafia-ai
docker-compose up -d
```

Это запустит:
- **CompreFace UI**: http://localhost:8001
- **CompreFace API**: http://localhost:8080 (внутренний порт)
- **PostgreSQL**: база данных для хранения лиц

### Шаг 2: Создание приложения в CompreFace

1. Откройте http://localhost:8001
2. Зарегистрируйтесь (первый пользователь становится админом)
3. Войдите в систему
4. Создайте новое приложение:
   - Нажмите **"Create Application"**
   - Название: `Mafia AI`
   - Описание: `Face recognition for Mafia game`

5. В приложении создайте **Recognition Service**:
   - Нажмите **"Add Recognition Service"**
   - Название: `Mafia Players`

6. Скопируйте **API ключи**:
   - **API Key** (ключ приложения)
   - **Recognition Service Key** (ключ сервиса распознавания)

### Шаг 3: Настройка Backend

Создайте файл `.env` в папке `mafia-ai/backend/`:

```env
# CompreFace Configuration
FACE_DETECTOR=compreface
COMPREFACE_API_URL=http://compreface-api:8080
COMPREFACE_API_KEY=ваш-api-key
COMPREFACE_RECOGNITION_KEY=ваш-recognition-service-key

# Face Recognition Settings
FACE_THRESHOLD=0.85                # Порог схожести (0-1)
COMPREFACE_DET_THRESHOLD=0.8      # Порог детекции (0-1)
COMPREFACE_LIMIT=10               # Макс. лиц на кадре
COMPREFACE_MIN_ENROLLMENT=5       # Мин. фото для регистрации
```

### Шаг 4: Запуск Backend

```bash
cd mafia-ai/backend
pip install -r requirements.txt
uvicorn app:app --reload
```

**Готово!** 🎉 CompreFace интегрирован и готов к использованию.

---

## 🔧 Подробная настройка

### Переменные окружения

| Переменная | По умолчанию | Описание |
|-----------|--------------|----------|
| `FACE_DETECTOR` | `compreface` | Тип детектора: `compreface`, `hybrid`, `legacy` |
| `COMPREFACE_API_URL` | `http://compreface-api:8080` | URL CompreFace API |
| `COMPREFACE_API_KEY` | - | API ключ приложения |
| `COMPREFACE_RECOGNITION_KEY` | - | Ключ Recognition Service |
| `FACE_THRESHOLD` | `0.85` | Порог схожести для распознавания (0-1) |
| `COMPREFACE_DET_THRESHOLD` | `0.8` | Порог уверенности детекции (0-1) |
| `COMPREFACE_LIMIT` | `10` | Максимальное количество лиц на кадре |
| `COMPREFACE_MIN_ENROLLMENT` | `5` | Мин. фото для качественной регистрации |

### Тонкая настройка порогов

#### `FACE_THRESHOLD` (порог распознавания)

Контролирует, насколько похожими должны быть лица для распознавания:

- **0.95-1.0**: Очень строгий (только почти идентичные лица)
- **0.85-0.95**: Рекомендуемый (баланс точности и удобства) ✅
- **0.70-0.85**: Мягкий (может быть ложных срабатываний)
- **< 0.70**: Слишком мягкий (много ошибок)

**Рекомендация:** Начните с `0.85`, затем настройте под вашу среду.

#### `COMPREFACE_DET_THRESHOLD` (порог детекции)

Контролирует уверенность детекции лица:

- **0.9-1.0**: Строгий (только явные лица)
- **0.8-0.9**: Оптимальный (пропускает хорошие лица) ✅
- **0.6-0.8**: Мягкий (детектирует больше, но может быть шум)
- **< 0.6**: Слишком мягкий (много ложных детекций)

**Рекомендация:** `0.8` для большинства случаев.

### Docker Compose конфигурация

CompreFace состоит из нескольких сервисов:

```yaml
services:
  compreface-postgres-db:    # База данных
  compreface-admin:          # Сервис администрирования
  compreface-api:            # REST API
  compreface-fe:             # Frontend UI (port 8001)
  compreface-core:           # ML движок
```

**Порты:**
- `8001` - CompreFace UI (внешний)
- `8080` - CompreFace API (внутренний, только для backend)

**Память:**
По умолчанию каждый сервис использует до 8GB RAM (`-Xmx8g`).

Для слабых машин измените в `docker-compose.yml`:

```yaml
environment:
  - ADMIN_JAVA_OPTS=-Xmx2g  # Вместо -Xmx8g
  - API_JAVA_OPTS=-Xmx2g
  - CORE_JAVA_OPTS=-Xmx2g
```

---

## 💻 Использование API

### Регистрация игрока (Enrollment)

```python
from infrastructure.detection.face import CompreFaceDetector, CompreFaceManager
import cv2

# Инициализация
detector = CompreFaceDetector(
    api_url="http://compreface-api:8080",
    api_key="your-api-key",
    recognition_service_key="your-recognition-key"
)

manager = CompreFaceManager(
    detector=detector,
    recognition_threshold=0.85
)

# Регистрация игрока
frame = cv2.imread("player_photo.jpg")

result = await manager.enroll_person_single(
    person_name="Игрок1",
    frame=frame
)

print(result)
# {
#   "success": True,
#   "person_name": "Игрок1",
#   "result": {...}
# }
```

### Распознавание лиц

```python
# Распознавание на кадре
frame = cv2.imread("game_frame.jpg")

results = await manager.recognize_faces(frame)

for result in results:
    if result.is_recognized:
        print(f"Найден: {result.person_name} (схожесть: {result.similarity:.2f})")
        x1, y1, x2, y2 = result.face.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, result.person_name, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        print("Неизвестное лицо")

cv2.imshow("Recognition", frame)
```

### Управление базой лиц

```python
# Список всех зарегистрированных игроков
players = await manager.list_persons()
print(f"Игроки: {players}")

# Удаление игрока
await manager.delete_person("Игрок1")

# Прогресс регистрации
count, quality = manager.get_enrollment_progress("Игрок1")
print(f"Добавлено фото: {count}, средняя оценка: {quality:.2f}")
```

### Использование через ServiceContainer

```python
from application.services import get_container

# Получить контейнер
container = get_container()

# CompreFace manager (если настроен)
manager = container.compreface_manager

if manager:
    # Регистрация
    await manager.enroll_person_single("Игрок1", frame)
    
    # Распознавание
    results = await manager.recognize_faces(frame)
```

---

## 🔄 Миграция с InsightFace

### Если у вас уже были зарегистрированы игроки:

**Вариант 1: Переключение на CompreFace (без миграции)**

Просто измените `.env`:

```env
FACE_DETECTOR=compreface
```

⚠️ **Внимание:** Все игроки должны быть зарегистрированы заново через CompreFace.

**Вариант 2: Временное использование обоих систем**

Можно временно использовать оба детектора:

```python
# В коде создайте оба детектора
hybrid_detector = HybridFaceDetector(...)
compreface_detector = CompreFaceDetector(...)

# Попробуйте сначала CompreFace, при ошибке - Hybrid
try:
    faces = await compreface_detector.detect(frame)
except:
    faces = await hybrid_detector.detect(frame)
```

**Вариант 3: Полный переход**

1. Экспортируйте список игроков из старой системы
2. Соберите их фотографии
3. Зарегистрируйте в CompreFace через API
4. Переключите `FACE_DETECTOR=compreface`

---

## 🛠️ Troubleshooting

### Проблема: CompreFace не запускается

**Симптомы:**
```
ERROR: compreface-api exited with code 1
```

**Решения:**

1. **Проверьте память:**
   ```bash
   docker stats
   ```
   Если нехватает RAM, уменьшите `-Xmx8g` до `-Xmx2g` в `docker-compose.yml`

2. **Проверьте логи:**
   ```bash
   docker-compose logs compreface-api
   docker-compose logs compreface-core
   ```

3. **Перезапустите:**
   ```bash
   docker-compose down
   docker-compose up -d
   ```

### Проблема: Backend не может подключиться к CompreFace

**Симптомы:**
```
[CompreFace] ⚠️  Initialization failed: Connection refused
```

**Решения:**

1. **Проверьте URL:**
   - В Docker: `COMPREFACE_API_URL=http://compreface-api:8080`
   - Локально: `COMPREFACE_API_URL=http://localhost:8000`

2. **Проверьте API ключи:**
   Убедитесь что скопировали правильные ключи из UI

3. **Проверьте сеть:**
   ```bash
   docker network inspect mafia-network
   ```

### Проблема: Низкая точность распознавания

**Симптомы:**
- Не распознает известных игроков
- Путает разных людей

**Решения:**

1. **Увеличьте количество фото:**
   ```env
   COMPREFACE_MIN_ENROLLMENT=10  # Вместо 5
   ```

2. **Отрегулируйте пороги:**
   - Снизьте `FACE_THRESHOLD` до `0.75-0.80`
   - Повысьте `COMPREFACE_DET_THRESHOLD` до `0.85`

3. **Улучшите качество фото:**
   - Хорошее освещение
   - Лицо анфас
   - Минимум 200x200 пикселей
   - Без очков/шляп при регистрации

### Проблема: Медленная работа

**Симптомы:**
- Долгая обработка кадров (>1 секунда)

**Решения:**

1. **Уменьшите разрешение кадров:**
   ```python
   frame = cv2.resize(frame, (640, 480))  # Вместо 1920x1080
   ```

2. **Уменьшите лимит лиц:**
   ```env
   COMPREFACE_LIMIT=3  # Вместо 10
   ```

3. **Используйте GPU** (если доступно):
   - В `docker-compose.yml` добавьте для `compreface-core`:
     ```yaml
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: 1
               capabilities: [gpu]
     ```

### Проблема: CompreFace UI не открывается

**Симптомы:**
- http://localhost:8001 не отвечает

**Решения:**

1. **Проверьте статус:**
   ```bash
   docker-compose ps
   ```
   Все сервисы должны быть `Up`

2. **Проверьте порт:**
   ```bash
   netstat -an | grep 8001
   ```
   Порт должен слушаться

3. **Проверьте firewall:**
   Убедитесь что порт 8001 открыт

---

## 📚 Дополнительные ресурсы

### Официальная документация CompreFace:
- **GitHub**: https://github.com/exadel-inc/CompreFace
- **Website**: https://exadel.com/accelerator-showcase/compreface/
- **REST API Docs**: https://github.com/exadel-inc/CompreFace/tree/master/docs
- **Community**: https://github.com/exadel-inc/CompreFace/discussions

### Наши файлы:
- `compreface_detector.py` - Интеграция с CompreFace API
- `compreface_manager.py` - Высокоуровневый сервис управления
- `config/settings.py` - Настройки конфигурации
- `application/services.py` - Dependency injection

---

## 🎮 Использование в игре Мафия

### Workflow регистрации игроков:

1. **Перед игрой**: Регистрируем всех игроков
   - Каждый игрок делает 5-10 фото
   - Система добавляет их в CompreFace
   - Проверяем качество регистрации

2. **Во время игры**: Распознаем игроков в реальном времени
   - Камера снимает игровой стол
   - CompreFace распознает лица
   - Система сопоставляет с игровыми позициями

3. **После игры**: Опционально очищаем базу
   - Удаляем игроков через API
   - Или оставляем для следующей игры

---

## ✅ Checklist успешной интеграции

- [ ] CompreFace запущен и доступен на http://localhost:8001
- [ ] Создано приложение "Mafia AI" в UI
- [ ] Создан Recognition Service "Mafia Players"
- [ ] API ключи скопированы в `.env`
- [ ] Backend запущен с `FACE_DETECTOR=compreface`
- [ ] Зарегистрирован тестовый игрок
- [ ] Тестовый игрок успешно распознается
- [ ] Пороги настроены под вашу среду

**Поздравляем! 🎉** CompreFace интегрирован и готов к использованию.

---

## 📞 Поддержка

Если возникли проблемы:

1. Проверьте [Troubleshooting](#troubleshooting)
2. Прочитайте логи: `docker-compose logs`
3. Изучите официальную документацию CompreFace
4. Создайте issue на GitHub

**Удачной игры в Мафию!** 🎲🕵️

