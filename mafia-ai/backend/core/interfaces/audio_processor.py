"""Audio processor interface - абстракция для аудио обработки"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class VoiceSegment:
    """Сегмент речи"""
    start_time: float      # Начало в секундах
    end_time: float        # Конец в секундах
    audio_data: np.ndarray # Аудио данные
    confidence: float      # Уверенность VAD


@dataclass
class Transcription:
    """Транскрипция речи"""
    text: str              # Распознанный текст
    confidence: float      # Уверенность распознавания
    language: str          # Язык
    segments: List[dict]   # Детальные сегменты (если есть)


@dataclass
class SpeakerIdentification:
    """Идентификация говорящего"""
    player_id: Optional[int]  # ID игрока (если определен)
    confidence: float          # Уверенность идентификации
    method: str                # Метод идентификации (audio_visual, face_correlation, etc)


class IAudioProcessor(ABC):
    """
    Интерфейс обработчика аудио
    Включает VAD, ASR, Speaker ID
    """

    @abstractmethod
    async def detect_voice_activity(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Voice Activity Detection - есть ли речь в аудио чанке

        Args:
            audio_chunk: Аудио данные (numpy array)
            sample_rate: Частота дискретизации

        Returns:
            True если обнаружена речь
        """
        pass

    @abstractmethod
    async def extract_speech_segments(
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
        pass

    @abstractmethod
    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000, language: str = "ru") -> Transcription:
        """
        Распознать речь в текст (ASR)

        Args:
            audio: Аудио данные
            sample_rate: Частота дискретизации
            language: Язык речи

        Returns:
            Транскрипция
        """
        pass

    @abstractmethod
    async def identify_speaker(
        self,
        audio_segment: VoiceSegment,
        video_frame: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None
    ) -> SpeakerIdentification:
        """
        Определить кто говорит (Speaker ID)

        Args:
            audio_segment: Сегмент речи
            video_frame: Видео кадр (для audio-visual correlation)
            timestamp: Временная метка

        Returns:
            Идентификация говорящего
        """
        pass
