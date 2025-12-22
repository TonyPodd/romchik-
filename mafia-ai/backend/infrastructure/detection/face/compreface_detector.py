"""CompreFace face detector - интеграция с CompreFace API для распознавания лиц"""

from typing import List, Optional, Dict, Any
import numpy as np
import cv2
import base64
from io import BytesIO
import asyncio
from functools import partial

from core.interfaces.detectors import IFaceDetector, Face

try:
    from compreface import CompreFace
    from compreface.service import RecognitionService
    COMPREFACE_AVAILABLE = True
except ImportError:
    COMPREFACE_AVAILABLE = False
    print("[CompreFace] ⚠️  compreface-sdk not installed. Install with: pip install compreface-sdk")


class CompreFaceDetector(IFaceDetector):
    """
    CompreFace-based face detector
    
    Использует CompreFace REST API для:
    - Детекции лиц
    - Получения эмбеддингов (512D)
    - Распознавания лиц по базе
    
    Преимущества:
    - Высокая точность распознавания (state-of-the-art)
    - Простая интеграция через REST API
    - Управление базой лиц через UI
    - Поддержка нескольких моделей
    - Хорошая документация
    """
    
    def __init__(
        self,
        api_url: str = "http://compreface-api:8080",
        api_key: Optional[str] = None,
        recognition_service_key: Optional[str] = None,
        detection_plugin_status: bool = True,
        age_gender_plugin_status: bool = False,
        face_plugins: str = "landmarks,calculator",
        det_prob_threshold: float = 0.8,
        limit: int = 10,
    ):
        """
        Args:
            api_url: URL CompreFace API (например http://localhost:8000)
            api_key: API ключ (можно задать позже)
            recognition_service_key: API ключ сервиса распознавания
            detection_plugin_status: Включить детекцию лиц
            age_gender_plugin_status: Включить определение возраста/пола
            face_plugins: Плагины (landmarks - ключевые точки, calculator - эмбеддинги)
            det_prob_threshold: Порог уверенности детекции (0-1)
            limit: Максимальное количество лиц на кадре
        """
        if not COMPREFACE_AVAILABLE:
            raise ImportError(
                "compreface-sdk not installed. Install with: pip install compreface-sdk"
            )
        
        self.api_url = api_url
        self.api_key = api_key
        self.recognition_service_key = recognition_service_key
        self.detection_plugin_status = detection_plugin_status
        self.face_plugins = face_plugins
        self.det_prob_threshold = det_prob_threshold
        self.limit = limit
        self.age_gender_plugin_status = age_gender_plugin_status
        
        self.compre_face: Optional[CompreFace] = None
        self.recognition_service: Optional[RecognitionService] = None
        self._initialized = False
        
        # Инициализация CompreFace (если есть ключи)
        if api_key and recognition_service_key:
            self._initialize_services()
    
    def _initialize_services(self):
        """Инициализация CompreFace сервисов"""
        try:
            if not self.recognition_service_key:
                print("[CompreFace] ⚠️  API keys not provided, skipping initialization")
                return
            
            # Парсим URL чтобы получить протокол, хост и порт
            # URL формат: http://localhost:8080
            from urllib.parse import urlparse
            parsed = urlparse(self.api_url)
            
            protocol = parsed.scheme or 'http'
            host = parsed.hostname or 'localhost'
            port_str = str(parsed.port) if parsed.port else "8080"
            
            # Формируем domain с протоколом для SDK
            # SDK CompreFace ожидает domain в формате: http://host
            domain_with_protocol = f"{protocol}://{host}"
            
            # Создаем клиент CompreFace
            self.compre_face = CompreFace(domain_with_protocol, port_str)
            
            # Создаем сервис распознавания с API ключом сервиса
            self.recognition_service = self.compre_face.init_face_recognition(
                self.recognition_service_key
            )
            
            self._initialized = True
            print(f"[CompreFace] ✅ Initialized successfully")
            print(f"[CompreFace] API URL: {protocol}://{host}:{port_str}")
            print(f"[CompreFace] Detection threshold: {self.det_prob_threshold}")
            
        except Exception as e:
            import traceback
            print(f"[CompreFace] ⚠️  Initialization failed: {e}")
            print(f"[CompreFace] Full traceback:")
            traceback.print_exc()
            print(f"[CompreFace] Will continue without face recognition")
            self._initialized = False
    
    def set_api_keys(self, api_key: str, recognition_service_key: str):
        """
        Установить API ключи после создания объекта
        
        Args:
            api_key: Основной API ключ
            recognition_service_key: API ключ сервиса распознавания
        """
        self.api_key = api_key
        self.recognition_service_key = recognition_service_key
        self._initialize_services()
    
    def _frame_to_bytes(self, frame: np.ndarray, format: str = '.jpg') -> bytes:
        """Конвертация numpy array в bytes"""
        success, encoded = cv2.imencode(format, frame)
        if not success:
            raise ValueError("Failed to encode frame")
        return encoded.tobytes()
    
    async def detect(self, frame: np.ndarray) -> List[Face]:
        """
        Обнаружить и распознать лица на кадре
        
        Args:
            frame: BGR изображение (numpy array)
        
        Returns:
            Список обнаруженных лиц с эмбеддингами
        """
        if not self._initialized or self.recognition_service is None:
            # Если CompreFace не инициализирован, возвращаем пустой список
            return []
        
        try:
            # Конвертируем кадр в bytes
            image_bytes = self._frame_to_bytes(frame)
            
            # Вызываем CompreFace API в отдельном потоке
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                partial(
                    self.recognition_service.recognize,
                    image_path=image_bytes,
                    det_prob_threshold=self.det_prob_threshold,
                    limit=self.limit,
                    face_plugins=self.face_plugins
                )
            )
            
            # Парсим результат
            faces = self._parse_compreface_result(result)
            
            return faces
            
        except Exception as e:
            print(f"[CompreFace] ⚠️  Detection error: {e}")
            return []
    
    def _parse_compreface_result(self, result: Dict[str, Any]) -> List[Face]:
        """
        Парсинг результата от CompreFace API
        
        Args:
            result: Ответ от CompreFace API
        
        Returns:
            Список объектов Face
        """
        faces = []
        
        # CompreFace возвращает список лиц в поле "result"
        face_results = result.get("result", [])
        
        for face_data in face_results:
            # Извлекаем bbox
            box = face_data.get("box", {})
            x_min = box.get("x_min", 0)
            y_min = box.get("y_min", 0)
            x_max = box.get("x_max", 0)
            y_max = box.get("y_max", 0)
            
            # Извлекаем уверенность детекции
            confidence = face_data.get("box", {}).get("probability", 0.0)
            
            # Извлекаем эмбеддинг (из плагина calculator)
            embedding = None
            if "embedding" in face_data:
                # Если есть эмбеддинг напрямую
                embedding = np.array(face_data["embedding"], dtype=np.float32)
            elif "execution_time" in face_data:
                # Если эмбеддинг в execution_time (некоторые версии API)
                calculator_data = face_data.get("execution_time", {}).get("calculator", None)
                if calculator_data:
                    embedding = np.array(calculator_data, dtype=np.float32)
            
            # Извлекаем landmarks (ключевые точки)
            landmarks = None
            if "landmarks" in face_data:
                landmarks_list = face_data["landmarks"]
                # Конвертируем в numpy array
                landmarks = np.array(
                    [[pt[0], pt[1]] for pt in landmarks_list],
                    dtype=np.float32
                )
            
            # Если эмбеддинг не найден, создаем пустой вектор
            if embedding is None:
                embedding = np.zeros(512, dtype=np.float32)
            
            # Создаем объект Face
            face = Face(
                bbox=(int(x_min), int(y_min), int(x_max), int(y_max)),
                embedding=embedding,
                confidence=float(confidence),
                landmarks=landmarks
            )
            
            faces.append(face)
        
        return faces
    
    async def add_subject(
        self,
        subject_name: str,
        frame: np.ndarray,
        det_prob_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Добавить лицо в базу CompreFace
        
        Args:
            subject_name: Имя субъекта (игрока)
            frame: BGR изображение с лицом
            det_prob_threshold: Порог детекции (если None - используется self.det_prob_threshold)
        
        Returns:
            Результат добавления от CompreFace API
        """
        if not self._initialized or self.recognition_service is None:
            raise RuntimeError("CompreFace not initialized")
        
        try:
            # Конвертируем кадр в bytes
            image_bytes = self._frame_to_bytes(frame)
            
            # Порог детекции
            threshold = det_prob_threshold if det_prob_threshold is not None else self.det_prob_threshold
            
            # Получаем face_collection для добавления лиц
            face_collection = self.recognition_service.get_face_collection()
            
            # Опции для детекции
            options = {"det_prob_threshold": threshold}
            
            # Добавляем в базу
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                partial(
                    face_collection.add,
                    image_path=image_bytes,
                    subject=subject_name,
                    options=options
                )
            )
            
            print(f"[CompreFace] ✅ Added subject '{subject_name}': {result}")
            return result
            
        except Exception as e:
            print(f"[CompreFace] ⚠️  Error adding subject '{subject_name}': {e}")
            raise
    
    async def delete_subject(self, subject_name: str) -> Dict[str, Any]:
        """
        Удалить субъекта из базы CompreFace
        
        Args:
            subject_name: Имя субъекта для удаления
        
        Returns:
            Результат удаления
        """
        if not self._initialized or self.recognition_service is None:
            raise RuntimeError("CompreFace not initialized")
        
        try:
            face_collection = self.recognition_service.get_face_collection()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                partial(face_collection.delete, subject=subject_name)
            )
            
            print(f"[CompreFace] ✅ Deleted subject '{subject_name}'")
            return result
            
        except Exception as e:
            print(f"[CompreFace] ⚠️  Error deleting subject '{subject_name}': {e}")
            raise
    
    async def list_subjects(self) -> List[str]:
        """
        Получить список всех субъектов в базе
        
        Returns:
            Список имен субъектов
        """
        if not self._initialized or self.recognition_service is None:
            return []
        
        try:
            face_collection = self.recognition_service.get_face_collection()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                face_collection.list
            )
            
            # CompreFace возвращает {"subjects": ["name1", "name2", ...]}
            subjects = result.get("subjects", [])
            return subjects
            
        except Exception as e:
            print(f"[CompreFace] ⚠️  Error listing subjects: {e}")
            return []
    
    def get_embedding_dim(self) -> int:
        """Размерность эмбеддинга CompreFace"""
        return 512
    
    @property
    def is_initialized(self) -> bool:
        """Проверка инициализации"""
        return self._initialized

