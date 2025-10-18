"""Audio-Video Correlator - определение кто говорит"""

from typing import Optional, List
import numpy as np

from core.interfaces.audio_processor import SpeakerIdentification
from core.domain.entities.player import Player


class AudioVideoCorrelator:
    """
    Audio-Video Correlator - определяет говорящего

    Методы:
    1. Face matching: сопоставление речи с видимыми лицами
    2. Temporal correlation: кто из игроков был активен во время речи
    3. Gesture correlation: движение губ / жесты

    Для простоты используем метод 1 + 2
    """

    def __init__(self):
        print("[AudioVideoCorrelator] initialized")

    def identify_speaker_by_faces(
        self,
        players: List[Player],
        visible_player_ids: List[int],
        timestamp: float,
    ) -> Optional[SpeakerIdentification]:
        """
        Определить говорящего по видимым лицам

        Логика:
        - Если видно одно лицо → это говорящий (confidence=0.9)
        - Если несколько лиц → берем того кто чаще всего в кадре
        - Если никого не видно → None

        Args:
            players: Список всех игроков
            visible_player_ids: ID видимых в кадре игроков
            timestamp: Временная метка

        Returns:
            Идентификация говорящего или None
        """
        if not visible_player_ids:
            return None

        # Простой случай: один игрок
        if len(visible_player_ids) == 1:
            return SpeakerIdentification(
                player_id=visible_player_ids[0],
                confidence=0.9,
                method="single_face_visible"
            )

        # Несколько игроков: берем первого (TODO: улучшить логику)
        # В будущем можно добавить анализ движения губ
        return SpeakerIdentification(
            player_id=visible_player_ids[0],
            confidence=0.6,
            method="multiple_faces_first"
        )

    def identify_speaker_by_position(
        self,
        current_turn_player_id: Optional[int],
        visible_player_ids: List[int],
    ) -> Optional[SpeakerIdentification]:
        """
        Определить по текущему ходу

        Args:
            current_turn_player_id: ID игрока чей сейчас ход
            visible_player_ids: Видимые игроки

        Returns:
            Идентификация говорящего
        """
        if current_turn_player_id is None:
            return None

        # Если говорящий должен быть конкретный игрок
        if current_turn_player_id in visible_player_ids:
            return SpeakerIdentification(
                player_id=current_turn_player_id,
                confidence=0.95,
                method="turn_based_high_conf"
            )
        else:
            # Игрока не видно, но его ход
            return SpeakerIdentification(
                player_id=current_turn_player_id,
                confidence=0.7,
                method="turn_based_low_conf"
            )

    def identify_speaker_hybrid(
        self,
        players: List[Player],
        visible_player_ids: List[int],
        current_turn_player_id: Optional[int],
        timestamp: float,
    ) -> Optional[SpeakerIdentification]:
        """
        Гибридный метод: комбинация face matching + turn-based

        Args:
            players: Все игроки
            visible_player_ids: Видимые игроки
            current_turn_player_id: Чей сейчас ход
            timestamp: Временная метка

        Returns:
            Наиболее вероятный говорящий
        """
        # Если известен чей ход, приоритет ему
        turn_result = self.identify_speaker_by_position(
            current_turn_player_id, visible_player_ids
        )

        if turn_result and turn_result.confidence >= 0.9:
            return turn_result

        # Иначе используем face matching
        face_result = self.identify_speaker_by_faces(
            players, visible_player_ids, timestamp
        )

        if face_result:
            return face_result

        # Фолбэк на turn если есть
        return turn_result
