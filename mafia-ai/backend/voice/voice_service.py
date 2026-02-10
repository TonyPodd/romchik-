# backend/voice/voice_service.py
"""
Сервис регистрации и распознавания голосов игроков.

Поддерживает два движка эмбеддингов:
- resemblyzer (по умолчанию, заметно точнее для speaker-ID)
- mfcc (fallback для совместимости)
"""

from __future__ import annotations

import json
import os
import platform
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np

TARGET_SR = 16000
FEATURE_DIM_MFCC = 166
FEATURE_DIM_RESEMBLYZER = 256

EMBEDDER_RESEMBLYZER = "resemblyzer_v1"
EMBEDDER_MFCC = "mfcc_v1"


@dataclass
class VoiceProfile:
    """Голосовой профиль игрока."""

    player_id: int
    player_name: str
    embeddings: List[np.ndarray]
    created_at: float
    embedder: str = EMBEDDER_MFCC


class VoiceService:
    """Сервис для регистрации и распознавания голоса."""

    def __init__(
        self,
        similarity_threshold: float = 0.66,
        min_margin: float = 0.012,
        storage_path: str = "storage/voice_profiles.json",
    ) -> None:
        self.profiles: Dict[int, VoiceProfile] = {}
        self.similarity_threshold = float(
            os.getenv("VOICE_SIMILARITY_THRESHOLD", str(similarity_threshold))
        )
        self.min_margin = float(os.getenv("VOICE_MIN_MARGIN", str(min_margin)))
        self.storage_path = storage_path

        # resemblyzer runtime
        self._resemblyzer_encoder: Optional[Any] = None
        self._resemblyzer_device: str = "cpu"
        self._resemblyzer_lock = threading.Lock()

        # Requested mode: resemblyzer | mfcc | auto
        self.preferred_embedder = (
            os.getenv("VOICE_EMBEDDER", "resemblyzer").strip().lower()
        )

        self._init_resemblyzer()
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

        data = np.nan_to_num(data, copy=False)
        peak = float(np.max(np.abs(data)))
        if peak > 1e-8:
            data = data / peak

        if sr != TARGET_SR and sr > 0:
            data = librosa.resample(
                data,
                orig_sr=sr,
                target_sr=TARGET_SR,
                res_type=os.getenv("VOICE_RESAMPLE_TYPE", "polyphase"),
            )

        trimmed, _ = librosa.effects.trim(data, top_db=28)
        if trimmed.size > 0:
            data = trimmed

        return data.astype(np.float32)

    def _resolve_resemblyzer_device(self) -> str:
        requested = os.getenv("VOICE_EMBEDDER_DEVICE", "auto").strip().lower()
        if requested in {"cpu", "cuda", "mps"}:
            return requested

        try:
            import torch

            if platform.system() == "Darwin" and torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _init_resemblyzer(self) -> None:
        if self.preferred_embedder not in {"resemblyzer", "auto", EMBEDDER_RESEMBLYZER}:
            return
        try:
            from resemblyzer import VoiceEncoder

            self._resemblyzer_device = self._resolve_resemblyzer_device()
            self._resemblyzer_encoder = VoiceEncoder(device=self._resemblyzer_device)
            print(f"[voice] Resemblyzer enabled (device={self._resemblyzer_device})")
        except Exception as exc:
            self._resemblyzer_encoder = None
            print(f"[voice] Resemblyzer unavailable, fallback to MFCC: {exc}")

    def _active_embedder_for_registration(self) -> str:
        if self.profiles and os.getenv("VOICE_FORCE_EMBEDDER", "0").strip().lower() not in {"1", "true", "yes"}:
            counts = Counter((p.embedder or EMBEDDER_MFCC) for p in self.profiles.values())
            dominant = counts.most_common(1)[0][0]
            if dominant == EMBEDDER_RESEMBLYZER and self._resemblyzer_encoder is not None:
                return EMBEDDER_RESEMBLYZER
            return EMBEDDER_MFCC

        pref = self.preferred_embedder
        if pref in {"mfcc", EMBEDDER_MFCC}:
            return EMBEDDER_MFCC
        if pref in {"resemblyzer", EMBEDDER_RESEMBLYZER}:
            return EMBEDDER_RESEMBLYZER if self._resemblyzer_encoder is not None else EMBEDDER_MFCC

        # auto
        if self.profiles:
            counts = Counter((p.embedder or EMBEDDER_MFCC) for p in self.profiles.values())
            if counts.get(EMBEDDER_RESEMBLYZER, 0) > 0 and self._resemblyzer_encoder is not None:
                return EMBEDDER_RESEMBLYZER
            if counts.get(EMBEDDER_MFCC, 0) > 0:
                return EMBEDDER_MFCC
        return EMBEDDER_RESEMBLYZER if self._resemblyzer_encoder is not None else EMBEDDER_MFCC

    def _active_embedder_for_identification(self) -> str:
        counts = Counter((p.embedder or EMBEDDER_MFCC) for p in self.profiles.values())
        pref = self.preferred_embedder

        if pref in {"resemblyzer", EMBEDDER_RESEMBLYZER}:
            if counts.get(EMBEDDER_RESEMBLYZER, 0) > 0 and self._resemblyzer_encoder is not None:
                return EMBEDDER_RESEMBLYZER
            if counts.get(EMBEDDER_MFCC, 0) > 0:
                return EMBEDDER_MFCC
            return EMBEDDER_RESEMBLYZER if self._resemblyzer_encoder is not None else EMBEDDER_MFCC

        if pref in {"mfcc", EMBEDDER_MFCC}:
            if counts.get(EMBEDDER_MFCC, 0) > 0:
                return EMBEDDER_MFCC
            if counts.get(EMBEDDER_RESEMBLYZER, 0) > 0 and self._resemblyzer_encoder is not None:
                return EMBEDDER_RESEMBLYZER
            return EMBEDDER_MFCC

        # auto: prefer dominant embedder in storage
        if counts:
            dominant = counts.most_common(1)[0][0]
            if dominant == EMBEDDER_RESEMBLYZER and self._resemblyzer_encoder is not None:
                return EMBEDDER_RESEMBLYZER
            return EMBEDDER_MFCC

        return EMBEDDER_RESEMBLYZER if self._resemblyzer_encoder is not None else EMBEDDER_MFCC

    def detect_voice_activity(
        self,
        audio: np.ndarray,
        sr: int = TARGET_SR,
        energy_threshold: float = 0.0075,
        min_duration_sec: float = 0.28,
    ) -> bool:
        """Быстрая проверка, что в сэмпле есть речь."""
        energy_threshold = float(os.getenv("VOICE_VAD_ENERGY_THRESHOLD", str(energy_threshold)))
        min_duration_sec = float(os.getenv("VOICE_VAD_MIN_DURATION_SEC", str(min_duration_sec)))
        voiced_ratio_threshold = float(os.getenv("VOICE_VAD_MIN_VOICED_RATIO", "0.035"))
        min_peak = float(os.getenv("VOICE_VAD_MIN_PEAK", "0.03"))

        data = self._prepare_audio(audio, sr)
        if data.size < int(TARGET_SR * min_duration_sec):
            return False

        rms = float(np.sqrt(np.mean(data**2)))
        peak = float(np.max(np.abs(data)))
        voiced_ratio = float(np.mean(np.abs(data) > max(0.05, 0.12 * peak)))
        if rms >= energy_threshold and voiced_ratio >= voiced_ratio_threshold:
            return True

        # Fallback for quiet mics / distant speaker: keep false positives low with peak gate.
        if peak >= min_peak and rms >= (energy_threshold * 0.55):
            return True
        return False

    def _extract_features_mfcc(self, audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
        """Fallback-эмбеддинг: MFCC + deltas + spectral stats."""
        try:
            data = self._prepare_audio(audio, sr)
            if data.size < int(TARGET_SR * 0.3):
                return np.zeros(FEATURE_DIM_MFCC, dtype=np.float32)

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

            if feature_vector.shape[0] != FEATURE_DIM_MFCC:
                return np.zeros(FEATURE_DIM_MFCC, dtype=np.float32)

            return self._normalize(feature_vector)
        except Exception as exc:
            print(f"[voice] MFCC feature extraction failed: {exc}")
            return np.zeros(FEATURE_DIM_MFCC, dtype=np.float32)

    def _extract_features_resemblyzer(self, audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
        """Primary-эмбеддинг: Resemblyzer d-vector."""
        if self._resemblyzer_encoder is None:
            return np.zeros(FEATURE_DIM_RESEMBLYZER, dtype=np.float32)
        try:
            data = self._prepare_audio(audio, sr)
            min_sec = float(os.getenv("VOICE_RESEMBLYZER_MIN_SEC", "0.45"))
            if data.size < int(TARGET_SR * min_sec):
                return np.zeros(FEATURE_DIM_RESEMBLYZER, dtype=np.float32)

            with self._resemblyzer_lock:
                vec = self._resemblyzer_encoder.embed_utterance(data)
            out = np.asarray(vec, dtype=np.float32).flatten()
            if out.size != FEATURE_DIM_RESEMBLYZER:
                return np.zeros(FEATURE_DIM_RESEMBLYZER, dtype=np.float32)
            return self._normalize(out)
        except Exception as exc:
            print(f"[voice] Resemblyzer feature extraction failed: {exc}")
            return np.zeros(FEATURE_DIM_RESEMBLYZER, dtype=np.float32)

    def extract_features(
        self,
        audio: np.ndarray,
        sr: int = TARGET_SR,
        embedder: Optional[str] = None,
    ) -> np.ndarray:
        kind = (embedder or EMBEDDER_MFCC).strip().lower()
        if kind in {"resemblyzer", EMBEDDER_RESEMBLYZER}:
            return self._extract_features_resemblyzer(audio, sr)
        return self._extract_features_mfcc(audio, sr)

    def _profile_similarity(self, query: np.ndarray, embeddings: List[np.ndarray]) -> float:
        if query.ndim != 1 or query.size == 0:
            return -1.0
        if not embeddings:
            return -1.0

        valid_sims: List[float] = []
        for emb in embeddings:
            if emb.shape != query.shape:
                continue
            valid_sims.append(float(np.dot(query, emb)))
        if not valid_sims:
            return -1.0

        valid_sims.sort(reverse=True)
        top = valid_sims[: min(3, len(valid_sims))]
        return float(0.7 * np.mean(top) + 0.3 * valid_sims[0])

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
            embedder = self._active_embedder_for_registration()
            valid_embeddings: List[np.ndarray] = []

            for sample in audio_samples:
                if self.detect_voice_activity(sample, sr):
                    emb = self.extract_features(sample, sr, embedder=embedder)
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
                embedder=embedder,
            )
            self._save_profiles()

            print(
                f"[voice] Registered {name} with {len(valid_embeddings)} samples "
                f"(embedder={embedder})"
            )
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
            prepared = self._prepare_audio(audio, sr)
            if prepared.size == 0:
                return []
            rms = float(np.sqrt(np.mean(prepared**2)))
            peak = float(np.max(np.abs(prepared)))
            fallback_rms = float(os.getenv("VOICE_IDENTIFY_FALLBACK_RMS", "0.005"))
            fallback_peak = float(os.getenv("VOICE_IDENTIFY_FALLBACK_PEAK", "0.03"))
            if rms < fallback_rms and peak < fallback_peak:
                return []

        target_embedder = self._active_embedder_for_identification()
        query = self.extract_features(audio, sr, embedder=target_embedder)
        if np.linalg.norm(query) <= 0:
            return []

        ranked: List[Tuple[int, str, float, str]] = []
        for profile in self.profiles.values():
            profile_embedder = (profile.embedder or EMBEDDER_MFCC).strip().lower()
            if profile_embedder != target_embedder:
                continue
            score = self._profile_similarity(query, profile.embeddings)
            if score <= -0.5:
                continue
            ranked.append((profile.player_id, profile.player_name, score, profile_embedder))

        ranked.sort(key=lambda item: item[2], reverse=True)
        return [
            {
                "player_id": pid,
                "player_name": name,
                "score": float(score),
                "embedder": embedder,
            }
            for pid, name, score, embedder in ranked[: max(1, k)]
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
        best_score = float(best["score"])
        active_threshold = float(self.similarity_threshold)
        if len(ranked) == 1:
            active_threshold -= 0.08
        if str(best.get("embedder") or "").strip().lower() == EMBEDDER_MFCC:
            active_threshold = min(active_threshold, 0.58)
        active_threshold = max(0.52, active_threshold)

        if best_score < active_threshold:
            return None

        if len(ranked) > 1:
            margin = best_score - float(ranked[1]["score"])
            # Защита от ложного выбора "первого профиля" при почти равных score.
            hard_ambiguity_margin = float(os.getenv("VOICE_HARD_AMBIGUITY_MARGIN", "0.02"))
            if best_score >= 0.90 and margin < hard_ambiguity_margin:
                return None
            if self.min_margin > 0 and margin < self.min_margin:
                return None

        return (
            int(best["player_id"]),
            str(best["player_name"]),
            best_score,
        )

    def list_profiles(self) -> List[Dict[str, Any]]:
        """Возвращает список зарегистрированных профилей."""
        return [
            {
                "player_id": profile.player_id,
                "player_name": profile.player_name,
                "samples_count": len(profile.embeddings),
                "created_at": profile.created_at,
                "embedder": profile.embedder,
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
                    "embedder": profile.embedder,
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
                embedder = str(item.get("embedder") or EMBEDDER_MFCC).strip().lower()
                loaded_profiles[pid] = VoiceProfile(
                    player_id=pid,
                    player_name=str(item.get("player_name") or f"Игрок {pid}"),
                    embeddings=embeddings,
                    created_at=float(item.get("created_at") or time.time()),
                    embedder=embedder,
                )
            except Exception:
                continue

        self.profiles = loaded_profiles
        if loaded_profiles:
            counts = Counter((p.embedder or EMBEDDER_MFCC) for p in loaded_profiles.values())
            print(
                f"[voice] Loaded {len(loaded_profiles)} profiles from disk "
                f"(embedder split: {dict(counts)})"
            )


_voice_service: Optional[VoiceService] = None


def get_voice_service() -> VoiceService:
    global _voice_service
    if _voice_service is None:
        _voice_service = VoiceService()
    return _voice_service
