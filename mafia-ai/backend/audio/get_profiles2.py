import queue
import threading
import sounddevice as sd
import numpy as np
import speech_recognition as sr
from pyannote.audio import Inference, Model
import torch
import tempfile
import wave
import os
from scipy.spatial.distance import cosine
import time

# -----------------------------
# Настройка
# -----------------------------
SAMPLE_RATE = 16000
SEGMENT_DURATION = 3  # секунд для обработки сегмента
CALIBRATION_DURATION = 30  # секунд для записи калибровочного аудио

recognizer = sr.Recognizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка предобученной модели для эмбеддингов спикеров
model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
inference = Inference(model, window="whole")

audio_queue = queue.Queue()
speaker_embeddings = {}
speaker_records_dir = "speaker_records"
os.makedirs(speaker_records_dir, exist_ok=True)

# -----------------------------
# Функции
# -----------------------------
def normalize(vec):
    return vec / np.linalg.norm(vec)

def cosine_similarity(a, b):
    return 1 - cosine(a, b)

def save_temp_wav(data):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((data * 32767).astype(np.int16).tobytes())
    return tmp.name

def save_speaker_record(data, speaker_name):
    path = os.path.join(speaker_records_dir, f"{speaker_name}_calib.wav")
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((data * 32767).astype(np.int16).tobytes())
    return path

def identify_speaker(embedding):
    best_score = -1
    best_name = "Неизвестный"
    for name, emb in speaker_embeddings.items():
        score = cosine_similarity(embedding, emb)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name

# -----------------------------
# Текст для калибровки
# -----------------------------
calibration_text = (
    "Сегодня мы поговорим о технологиях, которые меняют наш мир. "
    "Каждый день появляются новые устройства, программы и сервисы, "
    "которые делают нашу жизнь проще и интереснее. "
    "Программирование, искусственный интеллект, обработка данных и робототехника "
    "влияют на то, как мы работаем и отдыхаем. "
    "Важно учиться использовать современные инструменты и развивать свои навыки. "
    "Даже небольшие эксперименты и проекты помогают лучше понимать, как работают технологии, "
    "и создают возможности для будущего. "
    "Чтение, практика и обмен опытом с другими людьми делают процесс обучения увлекательным и эффективным."
)

# -----------------------------
# Калибровка спикеров с ограничением времени
# -----------------------------
num_speakers = int(input("Введите количество спикеров: "))

for _ in range(num_speakers):
    name = input("Введите имя спикера: ")
    print(f"\n{name}, пожалуйста, прочитайте следующий текст вслух в течение {CALIBRATION_DURATION} секунд:\n")
    print(calibration_text + "\n")
    print("Начинаем запись через 3 секунды...")
    time.sleep(3)
    print("Запись началась!")

    # Запись аудио с ограничением по времени
    recording = sd.rec(int(CALIBRATION_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("Запись завершена!\n")

    # Сохраняем и создаем эмбеддинг
    record_path = save_speaker_record(recording[:, 0], name)
    emb = inference(record_path)
    emb_mean = normalize(np.mean(np.array(emb), axis=0).flatten())
    speaker_embeddings[name] = emb_mean
    print(f"{name} успешно откалиброван.\n")

# -----------------------------
# Обработка аудио в реальном времени
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
            buffer = buffer[SEGMENT_DURATION*SAMPLE_RATE:]
            wav_file = save_temp_wav(segment)

            emb = inference(wav_file)
            emb_mean = normalize(np.mean(np.array(emb), axis=0).flatten())
            speaker_name = identify_speaker(emb_mean)

            # Распознавание речи
            with sr.AudioFile(wav_file) as source:
                audio_segment = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_segment, language="ru-RU")
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
