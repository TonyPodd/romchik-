"""Faster-Whisper ASR - быстрая транскрипция речи"""

import numpy as np
from typing import Optional

from core.interfaces.audio_processor import Transcription


class FasterWhisperASR:
    """
    Faster-Whisper - оптимизированный Whisper (в 3 раза быстрее)

    Преимущества:
    - 3-4x быстрее чем оригинальный Whisper
    - Такая же точность
    - Меньше памяти
    - Поддержка CPU и GPU
    """

    def __init__(
        self,
        model_size: str = "base",  # tiny, base, small, medium, large-v2
        device: str = "cpu",  # cpu, cuda, auto
        compute_type: str = "int8",  # int8, float16, float32
        language: str = "ru",
        num_workers: int = 1,
    ):
        """
        Args:
            model_size: Размер модели (tiny - fastest, large - best quality)
            device: Устройство (cpu, cuda, auto)
            compute_type: Тип вычислений (int8 - fastest, float32 - best quality)
            language: Язык по умолчанию
            num_workers: Количество потоков для CPU
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Install with: pip install faster-whisper"
            )

        self.model_size = model_size
        self.device = device
        self.language = language

        # Загружаем модель
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=num_workers if device == "cpu" else 0,
            num_workers=num_workers if device == "cpu" else 1,
        )

        print(f"[FasterWhisper] initialized (model={model_size}, device={device}, type={compute_type})")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        task: str = "transcribe",  # transcribe или translate
    ) -> Transcription:
        """
        Транскрибировать аудио в текст

        Args:
            audio: Аудио данные (numpy array, float32, mono)
            sample_rate: Частота дискретизации
            language: Язык (None = автодетект)
            task: transcribe или translate (в английский)

        Returns:
            Транскрипция с текстом и метаданными
        """
        # Faster-Whisper принимает float32 audio
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Нормализация амплитуды
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))

        # Транскрипция
        segments, info = self.model.transcribe(
            audio,
            language=language or self.language,
            task=task,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,  # Используем встроенный VAD
            vad_parameters=dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100,
            ),
        )

        # Собираем текст и сегменты
        full_text = []
        segments_list = []

        for segment in segments:
            full_text.append(segment.text)
            segments_list.append({
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": segment.words if hasattr(segment, "words") else None,
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
            })

        # Результат
        transcription = Transcription(
            text=" ".join(full_text).strip(),
            confidence=1.0 - info.language_probability if hasattr(info, "language_probability") else 0.9,
            language=info.language if hasattr(info, "language") else (language or self.language),
            segments=segments_list
        )

        return transcription

    async def transcribe_async(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> Transcription:
        """Async wrapper для транскрипции"""
        import asyncio
        return await asyncio.to_thread(
            self.transcribe, audio, sample_rate, language
        )
