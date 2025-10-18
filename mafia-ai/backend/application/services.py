"""Application Services - Dependency Injection Container"""

from typing import Optional
from pathlib import Path

from core.interfaces.repositories import IPlayerRepository, IGameRepository
from core.interfaces.detectors import IFaceDetector, IGestureDetector, ITableDetector
from infrastructure.storage.repositories import JsonPlayerRepository, JsonGameRepository
from infrastructure.detection.face import HybridFaceDetector, LegacyFaceDetector
from infrastructure.detection.gesture import LegacyGestureDetector
from infrastructure.detection.table import LegacyTableDetector
from config.settings import Settings


class ServiceContainer:
    """
    Service Container для dependency injection

    Централизованное управление зависимостями:
    - Repositories
    - Detectors
    - Configuration
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Args:
            settings: Настройки приложения
        """
        self.settings = settings or Settings()

        # Repositories
        self._player_repo: Optional[IPlayerRepository] = None
        self._game_repo: Optional[IGameRepository] = None

        # Detectors
        self._face_detector: Optional[IFaceDetector] = None
        self._gesture_detector: Optional[IGestureDetector] = None
        self._table_detector: Optional[ITableDetector] = None

    # ========== Repositories ==========

    @property
    def player_repository(self) -> IPlayerRepository:
        """Получить repository игроков"""
        if self._player_repo is None:
            storage_path = Path(self.settings.storage_path) / "players.json"
            self._player_repo = JsonPlayerRepository(str(storage_path))
        return self._player_repo

    @property
    def game_repository(self) -> IGameRepository:
        """Получить repository игр"""
        if self._game_repo is None:
            storage_path = Path(self.settings.storage_path) / "games.json"
            self._game_repo = JsonGameRepository(str(storage_path))
        return self._game_repo

    # ========== Detectors ==========

    @property
    def face_detector(self) -> IFaceDetector:
        """Получить face detector"""
        if self._face_detector is None:
            # Пытаемся использовать Hybrid (YOLOv8 + InsightFace)
            # Если не работает, fallback на Legacy (MediaPipe + ArcFace)
            try:
                self._face_detector = HybridFaceDetector(
                    yolo_model_size="n",  # nano для скорости
                    insightface_model="buffalo_s",  # small для стабильности
                    device="cpu",
                    confidence_threshold=self.settings.face_recognition_threshold
                )
                print("[ServiceContainer] ✅ Using Hybrid face detector")
            except Exception as e:
                print(f"[ServiceContainer] ⚠️  Hybrid detector failed: {e}")
                print("[ServiceContainer] Falling back to Legacy face detector")
                self._face_detector = LegacyFaceDetector(
                    threshold=self.settings.face_recognition_threshold
                )

        return self._face_detector

    @property
    def gesture_detector(self) -> IGestureDetector:
        """Получить gesture detector"""
        if self._gesture_detector is None:
            self._gesture_detector = LegacyGestureDetector()
            print("[ServiceContainer] ✅ Using Legacy gesture detector")
        return self._gesture_detector

    @property
    def table_detector(self) -> ITableDetector:
        """Получить table detector"""
        if self._table_detector is None:
            self._table_detector = LegacyTableDetector()
            print("[ServiceContainer] ✅ Using Legacy table detector")
        return self._table_detector

    # ========== Cleanup ==========

    def cleanup(self):
        """Очистка ресурсов"""
        # Здесь можно добавить cleanup logic для detectors если нужно
        pass


# Глобальный singleton container
_container: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    """
    Получить глобальный service container

    Returns:
        ServiceContainer instance
    """
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


def reset_container():
    """Сбросить container (для тестов)"""
    global _container
    if _container:
        _container.cleanup()
    _container = None
