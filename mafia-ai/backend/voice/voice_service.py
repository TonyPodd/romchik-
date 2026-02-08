# backend/voice/voice_service.py
"""
Сервис регистрации и распознавания голосов игроков.

Хранит профили на диске, чтобы распознавание работало после перезапуска backend.
Использует устойчивые спектральные признаки (MFCC + deltas + spectral stats).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np

TARGET_SR = 16000
FEATURE_DIM = 166


@dataclass
class VoiceProfile:
    """Голосовой профиль игрока."""

    player_id: int
    player_name: str
    embeddings: List[np.ndarray]
    created_at: float


class VoiceService:
    """Сервис для регистрации и распознавания голоса."""

    def __init__(
        self,
        similarity_threshold: float = 0.72,
        min_margin: float = 0.0,
        storage_path: str = "storage/voice_profiles.json",
    ) -> None:
        self.profiles: Dict[int, VoiceProfile] = {}
        self.similarity_threshold = similarity_threshold
        self.min_margin = min_margin
        self.storage_path = storage_path
        self._load_profiles()

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-8:
            return np.zeros_like(vec)
        return (vec / norm).astype(np.float32)

    def _prepare_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Нормализация, ресемплинг и trim тишины."""
        data = np.asarray(audio, dtype=np.float32).flatten()
        if data.size == 0:
            return data

        data = np.nan_to_num(data)
        peak = float(np.max(np.abs(data)))
        if peak > 0:
            data = data / peak

        if sr != TARGET_SR and sr > 0:
            data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)

        trimmed, _ = librosa.effects.trim(data, top_db=28)
        if trimmed.size > 0:
            data = trimmed

        return data.astype(np.float32)

    def detect_voice_activity(
        self,
        audio: np.ndarray,
        sr: int = TARGET_SR,
        energy_threshold: float = 0.012,
        min_duration_sec: float = 0.45,
    ) -> bool:
        """Быстрая проверка, что в сэмпле есть речь."""
        data = self._prepare_audio(audio, sr)
        if data.size < int(TARGET_SR * min_duration_sec):
            return False

        rms = float(np.sqrt(np.mean(data**2)))
        peak = float(np.max(np.abs(data)))
        voiced_ratio = float(np.mean(np.abs(data) > max(0.05, 0.12 * peak)))
        return rms >= energy_threshold and voiced_ratio >= 0.05

    def extract_features(self, audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
        """Извлекает фиксированный вектор признаков для speaker matching."""
        try:
            data = self._prepare_audio(audio, sr)
            if data.size < int(TARGET_SR * 0.3):
                return np.zeros(FEATURE_DIM, dtype=np.float32)

            # Pre-emphasis делает форму спектра стабильнее для speaker-ID.
            data = librosa.effects.preemphasis(data)

            mfcc = librosa.feature.mfcc(y=data, sr=TARGET_SR, n_mfcc=13)
            mfcc_delta = librosa.feature.delta(mfcc, order=1)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

            centroid = librosa.feature.spectral_centroid(y=data, sr=TARGET_SR)
            bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=TARGET_SR)
            rolloff = librosa.feature.spectral_rolloff(y=data, sr=TARGET_SR)
            flatness = librosa.feature.spectral_flatness(y=data)
            zcr = librosa.feature.zero_crossing_rate(data)

            def stats(x: np.ndarray) -> np.ndarray:
                return np.concatenate(
                    [
                        np.mean(x, axis=1),
                        np.std(x, axis=1),
                        np.percentile(x, 10, axis=1),
                        np.percentile(x, 90, axis=1),
                    ]
                )

            feature_vector = np.concatenate(
                [
                    stats(mfcc),
                    stats(mfcc_delta),
                    stats(mfcc_delta2),
                    [
                        float(np.mean(centroid)),
                        float(np.std(centroid)),
                        float(np.mean(bandwidth)),
                        float(np.std(bandwidth)),
                        float(np.mean(rolloff)),
                        float(np.std(rolloff)),
                        float(np.mean(flatness)),
                        float(np.std(flatness)),
                        float(np.mean(zcr)),
                        float(np.std(zcr)),
                    ],
                ],
                axis=0,
            ).astype(np.float32)

            if feature_vector.shape[0] != FEATURE_DIM:
                return np.zeros(FEATURE_DIM, dtype=np.float32)

            return self._normalize(feature_vector)
        except Exception as exc:
            print(f"[voice] Feature extraction failed: {exc}")
            return np.zeros(FEATURE_DIM, dtype=np.float32)

    def _profile_similarity(self, query: np.ndarray, embeddings: List[np.ndarray]) -> float:
        if not embeddings:
            return -1.0
        sims = [float(np.dot(query, emb)) for emb in embeddings]
        sims.sort(reverse=True)
        top = sims[: min(3, len(sims))]
        return float(0.7 * np.mean(top) + 0.3 * sims[0])

    def register_voice(
        self,
        player_id: int,
        player_name: str,
        audio_samples: List[np.ndarray],
        sr: int = TARGET_SR,
    ) -> bool:
        """Регистрирует или обновляет голосовой профиль игрока."""
        try:
            name = (player_name or "").strip() or f"Игрок {player_id}"
            valid_embeddings: List[np.ndarray] = []

            for sample in audio_samples:
                if self.detect_voice_activity(sample, sr):
                    emb = self.extract_features(sample, sr)
                    if np.linalg.norm(emb) > 0:
                        valid_embeddings.append(emb)

            if len(valid_embeddings) < 2:
                print(f"[voice] Not enough quality samples for {name}")
                return False

            self.profiles[int(player_id)] = VoiceProfile(
                player_id=int(player_id),
                player_name=name,
                embeddings=valid_embeddings,
                created_at=time.time(),
            )
            self._save_profiles()

            print(f"[voice] Registered {name} with {len(valid_embeddings)} samples")
            return True
        except Exception as exc:
            print(f"[voice] Registration failed: {exc}")
            return False

    def identify_top_k(
        self,
        audio: np.ndarray,
        sr: int = TARGET_SR,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Возвращает top-k кандидатов для диагностики качества распознавания."""
        if not self.profiles or len(audio) == 0:
            return []
        if not self.detect_voice_activity(audio, sr):
            return []

        query = self.extract_features(audio, sr)
        if np.linalg.norm(query) <= 0:
            return []

        ranked: List[Tuple[int, str, float]] = []
        for profile in self.profiles.values():
            score = self._profile_similarity(query, profile.embeddings)
            ranked.append((profile.player_id, profile.player_name, score))

        ranked.sort(key=lambda item: item[2], reverse=True)
        return [
            {
                "player_id": pid,
                "player_name": name,
                "score": float(score),
            }
            for pid, name, score in ranked[: max(1, k)]
        ]

    def identify_speaker(
        self,
        audio: np.ndarray,
        sr: int = TARGET_SR,
    ) -> Optional[Tuple[int, str, float]]:
        """Идентифицирует говорящего по аудио-сэмплу."""
        ranked = self.identify_top_k(audio, sr, k=2)
        if not ranked:
            return None

        best = ranked[0]
        if best["score"] < self.similarity_threshold:
            return None

        if self.min_margin > 0 and len(ranked) > 1:
            margin = float(best["score"] - ranked[1]["score"])
            if margin < self.min_margin:
                return None

        return (
            int(best["player_id"]),
            str(best["player_name"]),
            float(best["score"]),
        )

    def list_profiles(self) -> List[Dict[str, Any]]:
        """Возвращает список зарегистрированных профилей."""
        return [
            {
                "player_id": profile.player_id,
                "player_name": profile.player_name,
                "samples_count": len(profile.embeddings),
                "created_at": profile.created_at,
            }
            for profile in sorted(self.profiles.values(), key=lambda item: item.player_id)
        ]

    def clear_all(self) -> None:
        self.profiles.clear()
        self._save_profiles()
        print("[voice] All profiles cleared")

    def delete_profile(self, player_id: int) -> bool:
        pid = int(player_id)
        if pid not in self.profiles:
            return False
        player_name = self.profiles[pid].player_name
        del self.profiles[pid]
        self._save_profiles()
        print(f"[voice] Deleted profile for {player_name}")
        return True

    def _save_profiles(self) -> None:
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        payload = {
            "profiles": [
                {
                    "player_id": profile.player_id,
                    "player_name": profile.player_name,
                    "created_at": profile.created_at,
                    "embeddings": [emb.tolist() for emb in profile.embeddings],
                }
                for profile in self.profiles.values()
            ]
        }
        with open(self.storage_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)

    def _load_profiles(self) -> None:
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception as exc:
            print(f"[voice] Failed to load profiles: {exc}")
            return

        loaded_profiles: Dict[int, VoiceProfile] = {}
        for item in payload.get("profiles", []):
            try:
                embeddings = [
                    self._normalize(np.asarray(emb, dtype=np.float32))
                    for emb in item.get("embeddings", [])
                ]
                embeddings = [emb for emb in embeddings if np.linalg.norm(emb) > 0]
                if not embeddings:
                    continue

                pid = int(item["player_id"])
                loaded_profiles[pid] = VoiceProfile(
                    player_id=pid,
                    player_name=str(item.get("player_name") or f"Игрок {pid}"),
                    embeddings=embeddings,
                    created_at=float(item.get("created_at") or time.time()),
                )
            except Exception:
                continue

        self.profiles = loaded_profiles
        if loaded_profiles:
            print(f"[voice] Loaded {len(loaded_profiles)} profiles from disk")


_voice_service: Optional[VoiceService] = None


def get_voice_service() -> VoiceService:
    global _voice_service
    if _voice_service is None:
        _voice_service = VoiceService()
    return _voice_service
