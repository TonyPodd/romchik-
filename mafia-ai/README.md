# Mafia AI — MVP

## Backend (FastAPI)
- WebSocket /ws для событий реального времени
- Таймер игрока (tick/end), базовый broadcast
- Запуск: uvicorn app:app --reload
- Распознавание лиц: CompreFace (через REST API)

### Настройка CompreFace для face recognition
1. Подними стек: `docker compose up -d compreface-postgres-db compreface-core compreface-api compreface-admin compreface-ui`
2. Открой UI: `http://localhost:8001`
3. Создай Recognition service и получи API key
4. Экспортируй ключ для backend:
   - `export COMPREFACE_ENABLED=1`
   - `export FACE_PROVIDER=COMPREFACE`
   - `export COMPREFACE_URL=http://127.0.0.1:8002`
   - `export COMPREFACE_API_KEY=<your_key>`
5. Проверь статус провайдера: `GET /face/provider/status`

## Frontend (React + Vite + TS)
- Подключение к WS, стол на 10 мест, простой таймер
- Запуск: npm run dev
