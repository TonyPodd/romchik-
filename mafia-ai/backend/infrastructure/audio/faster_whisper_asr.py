"""Faster-Whisper ASR - быстрая транскрипция речи"""

import os
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
        self.beam_size = max(1, int(os.getenv("SPEECH_LOG_ASR_BEAM_SIZE", os.getenv("ASR_BEAM_SIZE", "2"))))
        self.best_of = max(1, int(os.getenv("SPEECH_LOG_ASR_BEST_OF", os.getenv("ASR_BEST_OF", str(self.beam_size)))))
        self.temperature = float(os.getenv("SPEECH_LOG_ASR_TEMPERATURE", os.getenv("ASR_TEMPERATURE", "0.0")))
        self.vad_threshold = float(os.getenv("SPEECH_LOG_ASR_VAD_THRESHOLD", "0.35"))
        self.vad_min_speech_ms = int(os.getenv("SPEECH_LOG_ASR_VAD_MIN_SPEECH_MS", "150"))
        self.vad_min_silence_ms = int(os.getenv("SPEECH_LOG_ASR_VAD_MIN_SILENCE_MS", "220"))
        self.repetition_penalty = float(os.getenv("SPEECH_LOG_ASR_REPETITION_PENALTY", "1.03"))
        self.no_repeat_ngram_size = int(os.getenv("SPEECH_LOG_ASR_NO_REPEAT_NGRAM_SIZE", "3"))
        self.no_speech_threshold = float(os.getenv("SPEECH_LOG_ASR_NO_SPEECH_THRESHOLD", "0.55"))
        self.log_prob_threshold = float(os.getenv("SPEECH_LOG_ASR_LOG_PROB_THRESHOLD", "-1.2"))
        self.compression_ratio_threshold = float(os.getenv("SPEECH_LOG_ASR_COMPRESSION_RATIO_THRESHOLD", "2.4"))
        self.initial_prompt = os.getenv("SPEECH_LOG_ASR_INITIAL_PROMPT", "").strip()
        self.retry_without_vad = os.getenv("SPEECH_LOG_ASR_RETRY_WITHOUT_VAD", "1").strip().lower() not in {"0", "false", "no"}

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
        transcribe_kwargs = {
            "language": language or self.language,
            "task": task,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "temperature": self.temperature,
            "condition_on_previous_text": False,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "no_speech_threshold": self.no_speech_threshold,
            "log_prob_threshold": self.log_prob_threshold,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "without_timestamps": True,
            "vad_filter": True,
            "vad_parameters": dict(
                threshold=self.vad_threshold,
                min_speech_duration_ms=self.vad_min_speech_ms,
                min_silence_duration_ms=self.vad_min_silence_ms,
            ),
        }
        if self.initial_prompt:
            transcribe_kwargs["initial_prompt"] = self.initial_prompt

        segments, info = self.model.transcribe(
            audio,
            **transcribe_kwargs,
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

        # На коротких чанках встроенный VAD иногда срезает полезную речь полностью.
        if self.retry_without_vad and not " ".join(full_text).strip():
            retry_kwargs = {
                "language": language or self.language,
                "task": task,
                "beam_size": max(self.beam_size, 6),
                "best_of": max(self.best_of, 6),
                "temperature": 0.0,
                "condition_on_previous_text": False,
                "repetition_penalty": self.repetition_penalty,
                "no_repeat_ngram_size": self.no_repeat_ngram_size,
                "no_speech_threshold": self.no_speech_threshold,
                "log_prob_threshold": self.log_prob_threshold,
                "compression_ratio_threshold": self.compression_ratio_threshold,
                "without_timestamps": True,
                "vad_filter": False,
            }
            if self.initial_prompt:
                retry_kwargs["initial_prompt"] = self.initial_prompt

            segments, info = self.model.transcribe(
                audio,
                **retry_kwargs,
            )

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
