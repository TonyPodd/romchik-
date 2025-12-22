"""
Voice Enrollment Service
Регистрация голосов игроков и создание speaker embeddings
"""

import numpy as np
import torch
import tempfile
import os
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from pyannote.audio import Inference, Model
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("[VoiceEnrollment] ⚠️  pyannote.audio not available")


class VoiceEnrollmentService:
    """Сервис для регистрации голосов игроков"""
    
    def __init__(self, model_name: str = "pyannote/wespeaker-voxceleb-resnet34-LM"):
        """
        Args:
            model_name: Название модели для speaker embeddings
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[Any] = None
        self.inference: Optional[Any] = None
        self._initialized = False
        
        if PYANNOTE_AVAILABLE:
            self._initialize_model()
    
    def _initialize_model(self):
        """Инициализация модели pyannote.audio"""
        try:
            print(f"[VoiceEnrollment] Loading model: {self.model_name}...")
            self.model = Model.from_pretrained(self.model_name)
            self.inference = Inference(self.model, window="whole")
            self._initialized = True
            print(f"[VoiceEnrollment] ✅ Model loaded successfully")
        except Exception as e:
            print(f"[VoiceEnrollment] ⚠️  Failed to load model: {e}")
            self._initialized = False
    
    def create_embedding(self, audio_data: bytes, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """
        Создает speaker embedding из аудио данных
        
        Args:
            audio_data: Аудио данные в формате WAV (bytes)
            sample_rate: Частота дискретизации
        
        Returns:
            numpy array с embedding или None при ошибке
        """
        if not self._initialized:
            print("[VoiceEnrollment] ⚠️  Model not initialized")
            return None
        
        try:
            # Сохраняем во временный файл
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            
            # Получаем embedding
            embedding = self.inference(tmp_path)
            
            # Усредняем по времени и нормализуем
            emb_array = np.array(embedding)
            emb_mean = np.mean(emb_array, axis=0).flatten()
            emb_normalized = self._normalize(emb_mean)
            
            # Удаляем временный файл
            os.unlink(tmp_path)
            
            return emb_normalized.astype(np.float32)
            
        except Exception as e:
            print(f"[VoiceEnrollment] ⚠️  Error creating embedding: {e}")
            return None
    
    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """Нормализация вектора"""
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            return vec / norm
        return vec
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Вычисление косинусного сходства между двумя embeddings
        
        Args:
            emb1: Первый embedding
            emb2: Второй embedding
        
        Returns:
            Косинусное сходство [-1, 1]
        """
        from scipy.spatial.distance import cosine
        return 1 - cosine(emb1, emb2)
    
    def save_voice_sample(
        self,
        player_id: int,
        audio_data: bytes,
        storage_dir: str = "storage/voice_samples"
    ) -> Optional[str]:
        """
        Сохраняет голосовой образец игрока
        
        Args:
            player_id: ID игрока
            audio_data: Аудио данные (WAV bytes)
            storage_dir: Директория для хранения
        
        Returns:
            Путь к сохраненному файлу или None
        """
        try:
            voice_dir = Path(storage_dir)
            voice_dir.mkdir(parents=True, exist_ok=True)
            
            voice_path = voice_dir / f"{player_id}.wav"
            
            with open(voice_path, "wb") as f:
                f.write(audio_data)
            
            print(f"[VoiceEnrollment] ✅ Saved voice sample: {voice_path}")
            return str(voice_path)
            
        except Exception as e:
            print(f"[VoiceEnrollment] ⚠️  Error saving voice sample: {e}")
            return None
    
    def create_enrollment_embedding(
        self,
        audio_samples: list[bytes],
        sample_rate: int = 16000
    ) -> Optional[Dict[str, Any]]:
        """
        Создает усредненный embedding из нескольких образцов
        
        Args:
            audio_samples: Список аудио данных (WAV bytes)
            sample_rate: Частота дискретизации
        
        Returns:
            Dict с результатами или None
        """
        if not audio_samples:
            return None
        
        embeddings = []
        for audio_data in audio_samples:
            emb = self.create_embedding(audio_data, sample_rate)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return None
        
        # Усредняем embeddings
        mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
        mean_emb = self._normalize(mean_emb)
        
        return {
            "embedding": mean_emb.tolist(),
            "samples_count": len(embeddings),
            "embedding_dim": len(mean_emb),
        }


# Глобальный экземпляр сервиса
_voice_service: Optional[VoiceEnrollmentService] = None


def get_voice_service() -> VoiceEnrollmentService:
    """Получить глобальный экземпляр VoiceEnrollmentService"""
    global _voice_service
    if _voice_service is None:
        _voice_service = VoiceEnrollmentService()
    return _voice_service

