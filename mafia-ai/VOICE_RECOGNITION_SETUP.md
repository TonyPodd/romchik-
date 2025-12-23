# 🎤 Голосовое распознавание игроков - Руководство

## ✅ Что уже работает

Система готова к использованию! Voice enrollment интегрирован в проект:

### Backend:
- ✅ API endpoints для регистрации голоса (`/players/enroll/voice`)
- ✅ Модуль `VoiceEnrollmentService` для создания voice embeddings
- ✅ Хранение voice embeddings в `players.json`
- ✅ Graceful degradation (работает без pyannote.audio)

### Frontend:
- ✅ Обновленная страница `VoiceRegistrationPage` с реальной записью аудио
- ✅ Web Audio API для записи через микрофон (3 секунды)
- ✅ Автоматическая отправка на backend
- ✅ Визуальный прогресс и статусы

---

## 📝 Как использовать (БЕЗ установки pyannote.audio)

### Текущий режим:
Backend **уже работает** и принимает голосовые записи, но **не создает embeddings** (требуется pyannote.audio).

Это полезно для:
- Тестирования UI
- Сохранения голосовых образцов игроков (WAV файлы)
- Подготовки данных для будущего обучения

### Шаги:
1. **Backend уже запущен** ✅
2. **Frontend:**
   - Обновите страницу: http://localhost:5173
   - Перейдите на страницу регистрации голосов
   - Нажмите "Записать" и произнесите фразу
   - Аудио сохранится в `backend/storage/voice_samples/`

---

## 🚀 Полная установка (С распознаванием)

### Если нужно реальное распознавание голосов:

#### 1. Установите зависимости:
```bash
cd mafia-ai/backend
pip install torch==2.2.0 torchaudio==2.2.0 pyannote.audio==3.1.1 speechbrain==0.5.16
```

**⚠️  Внимание:**
- PyTorch занимает ~2GB
- Установка может занять 5-10 минут
- Требуется ~5GB свободного места

#### 2. Перезапустите backend:
```bash
$env:COMPREFACE_API_URL="http://localhost:8080"
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### 3. Проверьте логи:
Вы должны увидеть:
```
[VoiceEnrollment] Loading model: pyannote/wespeaker-voxceleb-resnet34-LM...
[VoiceEnrollment] ✅ Model loaded successfully
```

---

## 🎯 Архитектура

### Backend:
```
backend/
├── infrastructure/
│   └── audio/
│       └── voice_enrollment.py      # Voice embedding service
├── storage/
│   ├── players.json                 # Игроки + voice_embedding
│   └── voice_samples/               # WAV файлы голосов
└── app.py                           # API endpoints
```

### Endpoints:
- `POST /players/enroll/voice` - Регистрация голоса
  - Params: `player_id: int`, `audio_file: UploadFile`
  - Returns: `{ok: true, voice_embedding: [...], embedding_dim: 512}`

- `GET /players/{player_id}/voice` - Информация о голосе
  - Returns: `{ok: true, has_voice: true, voice_path: "..."}`

### Модель игрока:
```json
{
  "id": 1,
  "name": "Игрок 1",
  "embedding": [...],           // Face embedding (ArcFace)
  "thumb": "thumbs/1.jpg",
  "voice_embedding": [...],     // Voice embedding (pyannote)
  "voice_path": "voice_samples/1.wav"
}
```

---

## 🔧 Voice Embedding Service

### Основные методы:

```python
from infrastructure.audio.voice_enrollment import get_voice_service

voice_service = get_voice_service()

# Создать embedding из WAV bytes
embedding = voice_service.create_embedding(audio_data, sample_rate=16000)

# Сохранить голосовой образец
path = voice_service.save_voice_sample(player_id, audio_data)

# Вычислить сходство между голосами
similarity = voice_service.cosine_similarity(emb1, emb2)
```

### Speaker Recognition Pipeline:
1. **Запись** → Web Audio API → WAV blob
2. **Отправка** → FormData → `/players/enroll/voice`
3. **Обработка** → pyannote.audio → embedding (512D)
4. **Сохранение** → `players.json` + `voice_samples/`

---

## 📊 Формат данных

### Audio:
- **Формат:** WAV (mono, 16kHz)
- **Длительность:** 3 секунды
- **Размер:** ~96KB на образец

### Embedding:
- **Модель:** wespeaker-voxceleb-resnet34-LM
- **Размерность:** 512 float32
- **Нормализация:** L2 (косинусное сходство)

---

## ⚙️  Конфигурация

### Frontend (VoiceRegistrationPage.tsx):
```typescript
const stream = await navigator.mediaDevices.getUserMedia({ 
  audio: {
    channelCount: 1,
    sampleRate: 16000,
    echoCancellation: true,
    noiseSuppression: true,
  } 
});

const duration = 3000; // 3 секунды записи
```

### Backend (voice_enrollment.py):
```python
model_name = "pyannote/wespeaker-voxceleb-resnet34-LM"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 🐛 Troubleshooting

### "Voice recognition model not loaded"
- **Причина:** pyannote.audio не установлен
- **Решение:** Установите зависимости (см. раздел "Полная установка")
- **Workaround:** Система продолжит работать, но без embeddings

### "Не удалось получить доступ к микрофону"
- **Причина:** Браузер не имеет разрешения
- **Решение:** Разрешите доступ к микрофону в настройках браузера

### "Ошибка при регистрации голоса"
- Проверьте логи backend: `terminals/17.txt`
- Убедитесь что player с таким ID существует
- Проверьте что аудио файл валидный

---

## 🎮 Использование в игре

После регистрации голосов, система может:
1. **Идентифицировать говорящего** по голосу
2. **Сопоставлять аудио с видео** (по face + voice)
3. **Детектировать уникальных спикеров** (speaker diarization)

### Пример использования:
```python
# В будущем: распознавание во время игры
voice_service = get_voice_service()

# Получаем embedding из микрофона
current_embedding = voice_service.create_embedding(audio_chunk)

# Сравниваем со всеми игроками
for player in players:
    if player.voice_embedding:
        similarity = voice_service.cosine_similarity(
            current_embedding,
            np.array(player.voice_embedding)
        )
        if similarity > 0.7:  # Threshold
            print(f"Говорит: {player.name}")
```

---

## 📚 Дополнительно

### Модели speaker recognition:
- [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) (используется)
- [pyannote/embedding](https://huggingface.co/pyannote/embedding) (альтернатива)
- [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (другая библиотека)

### Полезные ссылки:
- [pyannote.audio docs](https://github.com/pyannote/pyannote-audio)
- [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [MediaRecorder API](https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder)

---

## ✨ Что дальше?

1. ✅ Базовая регистрация голосов - **ГОТОВО**
2. 🔄 Установка pyannote.audio - **Опционально**
3. 🎯 Распознавание в реальном времени - **TODO**
4. 🔊 Speaker diarization во время игры - **TODO**
5. 🎙️  Улучшение качества записи (шумоподавление) - **TODO**

---

**🎉 Система голосового распознавания готова к использованию!**




