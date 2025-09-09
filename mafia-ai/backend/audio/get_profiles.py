import queue
import threading
import sounddevice as sd
import numpy as np
import speech_recognition as sr
from pyannote.audio import Inference
import torch
import tempfile
import wave

# -----------------------------
# Настройка
# -----------------------------
SAMPLE_RATE = 16000
SEGMENT_DURATION = 3  # секунд для обработки сегмента

recognizer = sr.Recognizer()
device = torch.device("cpu")  

TOKEN = "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"

audio_queue = queue.Queue()
speaker_embeddings = {}

inference = Inference("pyannote/embedding", device=device, use_auth_token=TOKEN)

# -----------------------------
# Функции
# -----------------------------
def normalize(vec):
    return vec / np.linalg.norm(vec)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def save_temp_wav(data):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((data * 32767).astype(np.int16).tobytes())
    return tmp.name

def identify_speaker(embedding):
    best_score = -1
    best_name = "Неизвестный"
    for name, emb in speaker_embeddings.items():
        score = cosine_similarity(embedding, emb)
        if score > best_score:
            best_score = score
            best_name = name
    print(best_name, best_score)
    return best_name

# -----------------------------
# Калибровка спикеров
# -----------------------------
num_speakers = int(input("Введите количество спикеров: "))

calib_phrases = [
    "Добрый день, я люблю программировать",
    "Сегодня отличная погода для работы",
    "Я рад участвовать в этой системе"
]

for _ in range(num_speakers):
    name = input("Введите имя спикера: ")
    embeddings = []

    for phrase in calib_phrases:
        print(f"{name}, пожалуйста, скажите фразу: \"{phrase}\"")

        with sr.Microphone(sample_rate=SAMPLE_RATE) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=7)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio.get_wav_data())
            tmp_path = tmp.name

        emb = inference(tmp_path)
        emb_mean = emb.data.mean(axis=0)
        embeddings.append(normalize(emb_mean))

    # Усредняем векторы
    speaker_embeddings[name] = normalize(np.mean(embeddings, axis=0))
    print(f"{name} откалиброван.\n")

# -----------------------------
# Обработка аудио
# -----------------------------
def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def process_audio():
    buffer = np.zeros((0,))
    while True:
        data = audio_queue.get()
        buffer = np.concatenate((buffer, data[:, 0]))
        if len(buffer) >= SAMPLE_RATE * SEGMENT_DURATION:
            segment = buffer[:SAMPLE_RATE*SEGMENT_DURATION]
            buffer = buffer[SAMPLE_RATE*SEGMENT_DURATION:]
            wav_file = save_temp_wav(segment)

            # speaker
            emb = inference(wav_file)
            emb_mean = normalize(emb.data.mean(axis=0))
            speaker_name = identify_speaker(emb_mean)

            # speech-to-text
            with sr.AudioFile(wav_file) as source:
                audio = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio, language="ru-RU")
                except sr.UnknownValueError:
                    text = "[Не удалось распознать]"
                except sr.RequestError as e:
                    text = f"[Ошибка сервиса: {e}]"

            print(f"{speaker_name}: {text}")

# -----------------------------
# Запуск
# -----------------------------
threading.Thread(target=process_audio, daemon=True).start()

with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=audio_callback):
    print("Говорите! Текст будет выводиться в реальном времени...\n")
    while True:
        sd.sleep(1000)
