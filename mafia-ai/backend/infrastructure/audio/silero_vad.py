"""Silero VAD - Voice Activity Detection"""

import numpy as np
import torch
from typing import List

from core.interfaces.audio_processor import VoiceSegment


class SileroVAD:
    """
    Silero VAD - быстрая и точная детекция речи

    Преимущества:
    - Работает на CPU (быстро)
    - Высокая точность (SOTA для VAD)
    - Робастность к шуму
    - Маленькая модель (~1MB)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
    ):
        """
        Args:
            threshold: Порог вероятности речи (0.0-1.0)
            sampling_rate: Частота дискретизации (8000 или 16000)
            min_speech_duration_ms: Минимальная длительность речи
            min_silence_duration_ms: Минимальная длительность тишины
        """
        try:
            # Загружаем модель Silero VAD
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            (self.get_speech_timestamps, _, self.read_audio, _, _) = utils

        except Exception as e:
            raise ImportError(
                f"Failed to load Silero VAD: {e}\n"
                "Install with: pip install torch torchaudio"
            )

        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms

        print(f"[SileroVAD] initialized (sr={sampling_rate}Hz, threshold={threshold})")

    def detect_voice_activity(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000
    ) -> bool:
        """
        Простая детекция: есть ли речь в чанке

        Args:
            audio_chunk: Аудио данные (numpy array)
            sample_rate: Частота дискретизации

        Returns:
            True если обнаружена речь
        """
        # Конвертируем в torch tensor
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk).float()
        else:
            audio_tensor = audio_chunk

        # Resample если нужно
        if sample_rate != self.sampling_rate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.sampling_rate
            )
            audio_tensor = resampler(audio_tensor)

        # Вычисляем вероятность речи
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sampling_rate).item()

        return speech_prob >= self.threshold

    def extract_speech_segments(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        min_speech_duration: float = 0.5
    ) -> List[VoiceSegment]:
        """
        Извлечь сегменты речи из аудио

        Args:
            audio: Аудио данные
            sample_rate: Частота дискретизации
            min_speech_duration: Минимальная длительность речи в секундах

        Returns:
            Список сегментов речи
        """
        # Конвертируем в torch tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio

        # Resample если нужно
        if sample_rate != self.sampling_rate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.sampling_rate
            )
            audio_tensor = resampler(audio_tensor)

        # Получаем timestamps речи
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sampling_rate,
            min_speech_duration_ms=max(
                self.min_speech_duration_ms,
                int(min_speech_duration * 1000)
            ),
            min_silence_duration_ms=self.min_silence_duration_ms,
        )

        segments = []
        for ts in speech_timestamps:
            start_sample = ts['start']
            end_sample = ts['end']

            start_time = start_sample / self.sampling_rate
            end_time = end_sample / self.sampling_rate

            # Извлекаем аудио сегмент
            audio_segment = audio_tensor[start_sample:end_sample].numpy()

            segment = VoiceSegment(
                start_time=start_time,
                end_time=end_time,
                audio_data=audio_segment,
                confidence=1.0  # Silero не возвращает confidence per segment
            )
            segments.append(segment)

        return segments
