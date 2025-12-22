# CompreFace Setup - Интеграция с Mafia AI

## Что такое CompreFace?

CompreFace - это профессиональная open-source система распознавания лиц с REST API, которая заменяет нашу самодельную систему и обеспечивает гораздо более высокую точность.

## Установка CompreFace

### 1. Запуск через Docker Compose

```bash
# Из корня проекта
cd /Users/tony/Desktop/CODE/Projects/diplom

# Запуск CompreFace (первый запуск займет 2-3 минуты)
docker-compose up -d

# Проверка статуса
docker-compose ps

# Логи (если что-то не работает)
docker-compose logs -f
```

CompreFace будет доступен по адресу: http://localhost:8001

### 2. Первоначальная настройка через UI

1. Откройте http://localhost:8001 в браузере
2. Зарегистрируйте администратора:
   - Email: admin@mafia-ai.local
   - Password: admin123 (или любой другой)

3. Войдите в систему

4. Создайте Application:
   - Name: "Mafia Game"
   - Description: "Face recognition for Mafia AI game"

5. Создайте Recognition Service:
   - Name: "Player Recognition"
   - Type: Recognition

6. Скопируйте API Key из Recognition Service
   - Нажмите на иконку ключа рядом с сервисом
   - Скопируйте API Key

### 3. Настройка Backend

Создайте файл `.env` в `mafia-ai/backend/` (если еще не создан):

```bash
cd mafia-ai/backend
touch .env
```

Добавьте в `.env`:

```
# CompreFace Configuration
COMPREFACE_ENABLED=true
COMPREFACE_URL=http://localhost:8001
COMPREFACE_API_KEY=ваш_api_key_здесь
```

Замените `ваш_api_key_здесь` на скопированный API Key из шага 2.6.

### 4. Установка зависимостей

```bash
cd mafia-ai/backend

# Установка httpx для работы с CompreFace API
pip install httpx==0.27.0

# Или установить все зависимости заново
pip install -r requirements.txt
```

### 5. Перезапуск Backend

```bash
# Остановить текущий backend (Ctrl+C если запущен)

# Запустить заново
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Проверка работы

### Тест через curl

```bash
# Проверка, что CompreFace работает
curl http://localhost:8001/api/v1/recognition/subjects \
  -H "x-api-key: ваш_api_key"

# Должен вернуть пустой список subjects: {"subjects":[]}
```

## Как это работает

### Регистрация игрока (Enrollment)

Когда пользователь регистрирует нового игрока:

1. Фронтенд отправляет фото на `/players/enroll/snap`
2. Backend собирает 24 образца с разных ракурсов
3. При завершении (`/players/enroll/finish`):
   - Все 24 образца отправляются в CompreFace через `add_face()`
   - CompreFace создает профиль игрока с именем subject
   - Backend сохраняет только метаданные (имя, ID, thumbnail)

### Распознавание во время игры (Recognition)

Каждый кадр (30 FPS):

1. MediaPipe детектирует лица в кадре (bbox)
2. Для каждого обнаруженного лица:
   - Вырезается crop изображения
   - Отправляется в CompreFace через `recognize_face()`
3. CompreFace возвращает:
   - Имя игрока (subject)
   - Similarity score (0.0-1.0)
   - Bbox координаты
4. Система отображает имя над лицом

### Преимущества CompreFace

- ✅ **Профессиональное качество** - точность 99%+
- ✅ **Стабильное распознавание** - работает при поворотах головы
- ✅ **Нет путаницы** - правильно различает разных людей
- ✅ **Быстрое распознавание** - ~50-100ms на запрос
- ✅ **Поддержка GPU** - можно ускорить при наличии NVIDIA GPU

## Остановка и удаление

```bash
# Остановить CompreFace
docker-compose stop

# Запустить снова
docker-compose start

# Полное удаление (с данными!)
docker-compose down -v
```

## Troubleshooting

### CompreFace не запускается

```bash
# Проверить логи
docker-compose logs compreface-core
docker-compose logs compreface-api

# Часто помогает перезапуск
docker-compose restart
```

### "Connection refused" ошибки

- Убедитесь, что CompreFace запущен: `docker-compose ps`
- Проверьте, что порт 8001 свободен: `lsof -i :8001`
- Проверьте URL в `.env`: должен быть `http://localhost:8001`

### API Key не работает

- Проверьте, что скопировали правильный ключ из Recognition Service (не из Application)
- Ключ должен быть длинным UUID без пробелов
- Попробуйте пересоздать Recognition Service и скопировать новый ключ

### Медленная работа

- CompreFace требует ~4GB RAM
- Первый запуск медленный (загрузка моделей)
- Последующие запросы должны быть быстрыми (~50-100ms)
- Для ускорения можно использовать GPU (требует NVIDIA GPU)

## Следующие шаги

После настройки CompreFace:

1. Удалите всех старых игроков из системы
2. Зарегистрируйте игроков заново через новую систему
3. Протестируйте распознавание с 2+ игроками
4. Наслаждайтесь стабильным распознаванием! 🎉
