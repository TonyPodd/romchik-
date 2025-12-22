"""Application settings - конфигурация через переменные окружения"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """
    Настройки приложения
    Можно переопределить через переменные окружения
    """

    # API Settings
    api_title: str = "Mafia AI Backend"
    api_version: str = "2.0.0"
    debug: bool = Field(default=False, validation_alias="DEBUG")

    # Server
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")

    # Camera Settings
    camera_index: int = Field(default=0, validation_alias="CAMERA_INDEX")
    camera_width: int = Field(default=1280, validation_alias="CAMERA_WIDTH")
    camera_height: int = Field(default=720, validation_alias="CAMERA_HEIGHT")
    camera_fps: int = Field(default=30, validation_alias="CAMERA_FPS")

    # Face Detection
    face_detector_type: str = Field(default="compreface", validation_alias="FACE_DETECTOR")  # compreface, yolov8, mediapipe
    face_recognition_threshold: float = Field(default=0.85, validation_alias="FACE_THRESHOLD")
    face_model_path: Optional[str] = Field(default=None, validation_alias="FACE_MODEL_PATH")
    
    # CompreFace Settings
    compreface_api_url: str = Field(default="http://localhost:8080", validation_alias="COMPREFACE_API_URL")
    compreface_api_key: Optional[str] = Field(default=None, validation_alias="COMPREFACE_API_KEY")
    compreface_recognition_key: Optional[str] = Field(default=None, validation_alias="COMPREFACE_RECOGNITION_KEY")
    compreface_det_threshold: float = Field(default=0.8, validation_alias="COMPREFACE_DET_THRESHOLD")
    compreface_limit: int = Field(default=10, validation_alias="COMPREFACE_LIMIT")
    compreface_min_enrollment: int = Field(default=5, validation_alias="COMPREFACE_MIN_ENROLLMENT")

    # Gesture Detection
    gesture_detector_type: str = Field(default="hybrid", validation_alias="GESTURE_DETECTOR")  # hybrid, yolo, mediapipe
    hand_detection_confidence: float = Field(default=0.5, validation_alias="HAND_CONFIDENCE")

    # Table Detection
    table_y_ratio: float = Field(default=0.8, validation_alias="TABLE_Y_RATIO")

    # Audio Settings
    audio_sample_rate: int = Field(default=16000, validation_alias="AUDIO_SAMPLE_RATE")
    vad_threshold: float = Field(default=0.5, validation_alias="VAD_THRESHOLD")
    asr_model: str = Field(default="base", validation_alias="ASR_MODEL")  # tiny, base, small, medium, large
    asr_language: str = Field(default="ru", validation_alias="ASR_LANGUAGE")

    # Game Settings
    speech_duration: int = Field(default=60, validation_alias="SPEECH_DURATION")  # seconds
    last_word_duration: int = Field(default=60, validation_alias="LAST_WORD_DURATION")
    tie_speech_duration: int = Field(default=30, validation_alias="TIE_SPEECH_DURATION")

    # Storage
    storage_dir: str = Field(default="storage", validation_alias="STORAGE_DIR")
    players_db_path: str = Field(default="storage/players.json", validation_alias="PLAYERS_DB")
    games_db_path: str = Field(default="storage/games.json", validation_alias="GAMES_DB")

    # Models Directory
    models_dir: str = Field(default="models", validation_alias="MODELS_DIR")

    # Performance
    max_concurrent_detections: int = Field(default=4, validation_alias="MAX_CONCURRENT_DETECTIONS")
    detection_batch_size: int = Field(default=1, validation_alias="DETECTION_BATCH_SIZE")

    # CORS
    cors_origins: list[str] = Field(
        default=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        validation_alias="CORS_ORIGINS"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Глобальный экземпляр настроек
settings = Settings()
