"""CompreFace Face Manager - высокоуровневый сервис для управления лицами"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2
from dataclasses import dataclass
import asyncio
from collections import defaultdict

from core.interfaces.detectors import Face
from .compreface_detector import CompreFaceDetector


@dataclass
class RecognitionResult:
    """Результат распознавания лица"""
    face: Face
    person_name: Optional[str] = None
    person_id: Optional[str] = None
    similarity: float = 0.0
    is_recognized: bool = False


class CompreFaceManager:
    """
    Менеджер для работы с CompreFace
    
    Функции:
    - Регистрация новых игроков (enrollment)
    - Распознавание лиц в реальном времени
    - Управление базой лиц
    - Трекинг качества регистрации
    """
    
    def __init__(
        self,
        detector: CompreFaceDetector,
        recognition_threshold: float = 0.85,
        min_enrollment_samples: int = 5,
    ):
        """
        Args:
            detector: CompreFaceDetector instance
            recognition_threshold: Порог схожести для распознавания (0-1)
            min_enrollment_samples: Минимальное количество фото для регистрации
        """
        self.detector = detector
        self.recognition_threshold = recognition_threshold
        self.min_enrollment_samples = min_enrollment_samples
        
        # Статистика регистрации (для UI)
        self.enrollment_progress: Dict[str, int] = defaultdict(int)
        self.enrollment_quality: Dict[str, List[float]] = defaultdict(list)
    
    async def enroll_person(
        self,
        person_name: str,
        frames: List[np.ndarray],
        min_face_size: int = 80,
        max_faces_per_frame: int = 1
    ) -> Dict[str, Any]:
        """
        Зарегистрировать нового человека в базе
        
        Args:
            person_name: Имя игрока
            frames: Список кадров с лицом
            min_face_size: Минимальный размер лица в пикселях
            max_faces_per_frame: Максимальное количество лиц на кадре
        
        Returns:
            Результат регистрации с статистикой
        """
        if not self.detector.is_initialized:
            return {
                "success": False,
                "error": "CompreFace not initialized",
                "person_name": person_name
            }
        
        added_count = 0
        failed_count = 0
        quality_scores = []
        
        for i, frame in enumerate(frames):
            try:
                # Проверяем наличие лица
                detected_faces = await self.detector.detect(frame)
                
                # Фильтруем по количеству лиц
                if len(detected_faces) != max_faces_per_frame:
                    print(f"[Enroll] Frame {i}: Expected {max_faces_per_frame} face(s), found {len(detected_faces)}")
                    failed_count += 1
                    continue
                
                # Проверяем размер лица
                face = detected_faces[0]
                x1, y1, x2, y2 = face.bbox
                face_width = x2 - x1
                face_height = y2 - y1
                
                if face_width < min_face_size or face_height < min_face_size:
                    print(f"[Enroll] Frame {i}: Face too small ({face_width}x{face_height})")
                    failed_count += 1
                    continue
                
                # Добавляем в базу CompreFace
                result = await self.detector.add_subject(
                    subject_name=person_name,
                    frame=frame
                )
                
                # Сохраняем качество
                if "result" in result and len(result["result"]) > 0:
                    box_data = result["result"][0].get("box", {})
                    probability = box_data.get("probability", 0.0)
                    quality_scores.append(probability)
                
                added_count += 1
                print(f"[Enroll] Frame {i}: ✅ Added successfully")
                
            except Exception as e:
                print(f"[Enroll] Frame {i}: ⚠️  Error: {e}")
                failed_count += 1
        
        # Обновляем статистику
        self.enrollment_progress[person_name] = added_count
        self.enrollment_quality[person_name] = quality_scores
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        return {
            "success": added_count >= self.min_enrollment_samples,
            "person_name": person_name,
            "added_count": added_count,
            "failed_count": failed_count,
            "total_frames": len(frames),
            "average_quality": float(avg_quality),
            "quality_scores": quality_scores,
            "meets_minimum": added_count >= self.min_enrollment_samples
        }
    
    async def enroll_person_single(
        self,
        person_name: str,
        frame: np.ndarray
    ) -> Dict[str, Any]:
        """
        Зарегистрировать одно фото человека
        
        Args:
            person_name: Имя игрока
            frame: Кадр с лицом
        
        Returns:
            Результат регистрации
        """
        try:
            result = await self.detector.add_subject(
                subject_name=person_name,
                frame=frame
            )
            
            # Обновляем прогресс
            self.enrollment_progress[person_name] += 1
            
            return {
                "success": True,
                "person_name": person_name,
                "result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "person_name": person_name,
                "error": str(e)
            }
    
    async def recognize_faces(
        self,
        frame: np.ndarray
    ) -> List[RecognitionResult]:
        """
        Распознать лица на кадре
        
        Args:
            frame: BGR изображение
        
        Returns:
            Список результатов распознавания
        """
        if not self.detector.is_initialized:
            return []
        
        try:
            # Получаем результат от CompreFace
            raw_result = await self._recognize_with_compreface(frame)
            
            # Парсим результат
            recognition_results = self._parse_recognition_result(raw_result)
            
            return recognition_results
            
        except Exception as e:
            print(f"[CompreFaceManager] Recognition error: {e}")
            return []
    
    async def _recognize_with_compreface(self, frame: np.ndarray) -> Dict[str, Any]:
        """Вызов CompreFace API для распознавания"""
        from functools import partial
        
        image_bytes = self.detector._frame_to_bytes(frame)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                self.detector.recognition_service.recognize,
                image_path=image_bytes,
                det_prob_threshold=self.detector.det_prob_threshold,
                limit=self.detector.limit,
                face_plugins=self.detector.face_plugins
            )
        )
        
        return result
    
    def _parse_recognition_result(self, result: Dict[str, Any]) -> List[RecognitionResult]:
        """
        Парсинг результата распознавания от CompreFace
        
        Args:
            result: Ответ от CompreFace API
        
        Returns:
            Список RecognitionResult
        """
        recognition_results = []
        
        face_results = result.get("result", [])
        
        for face_data in face_results:
            # Извлекаем bbox
            box = face_data.get("box", {})
            x_min = box.get("x_min", 0)
            y_min = box.get("y_min", 0)
            x_max = box.get("x_max", 0)
            y_max = box.get("y_max", 0)
            confidence = box.get("probability", 0.0)
            
            # Извлекаем эмбеддинг
            embedding = np.zeros(512, dtype=np.float32)
            if "embedding" in face_data:
                embedding = np.array(face_data["embedding"], dtype=np.float32)
            
            # Извлекаем landmarks
            landmarks = None
            if "landmarks" in face_data:
                landmarks_list = face_data["landmarks"]
                landmarks = np.array(
                    [[pt[0], pt[1]] for pt in landmarks_list],
                    dtype=np.float32
                )
            
            # Создаем объект Face
            face = Face(
                bbox=(int(x_min), int(y_min), int(x_max), int(y_max)),
                embedding=embedding,
                confidence=float(confidence),
                landmarks=landmarks
            )
            
            # Извлекаем информацию о распознавании
            person_name = None
            person_id = None
            similarity = 0.0
            is_recognized = False
            
            # CompreFace возвращает subjects (список похожих лиц)
            subjects = face_data.get("subjects", [])
            
            if subjects and len(subjects) > 0:
                # Берем самое похожее лицо
                best_match = subjects[0]
                similarity = best_match.get("similarity", 0.0)
                
                # Проверяем порог
                if similarity >= self.recognition_threshold:
                    person_name = best_match.get("subject", None)
                    person_id = person_name  # В CompreFace subject и есть ID
                    is_recognized = True
            
            recognition_result = RecognitionResult(
                face=face,
                person_name=person_name,
                person_id=person_id,
                similarity=float(similarity),
                is_recognized=is_recognized
            )
            
            recognition_results.append(recognition_result)
        
        return recognition_results
    
    async def delete_person(self, person_name: str) -> Dict[str, Any]:
        """
        Удалить человека из базы
        
        Args:
            person_name: Имя игрока
        
        Returns:
            Результат удаления
        """
        try:
            result = await self.detector.delete_subject(person_name)
            
            # Очищаем статистику
            if person_name in self.enrollment_progress:
                del self.enrollment_progress[person_name]
            if person_name in self.enrollment_quality:
                del self.enrollment_quality[person_name]
            
            return {
                "success": True,
                "person_name": person_name,
                "result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "person_name": person_name,
                "error": str(e)
            }
    
    async def list_persons(self) -> List[str]:
        """
        Получить список всех зарегистрированных людей
        
        Returns:
            Список имен
        """
        return await self.detector.list_subjects()
    
    def get_enrollment_progress(self, person_name: str) -> Tuple[int, float]:
        """
        Получить прогресс регистрации
        
        Args:
            person_name: Имя игрока
        
        Returns:
            (количество добавленных фото, средняя оценка качества)
        """
        count = self.enrollment_progress.get(person_name, 0)
        quality_scores = self.enrollment_quality.get(person_name, [])
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        return count, float(avg_quality)
    
    def reset_enrollment_progress(self, person_name: str):
        """Сбросить прогресс регистрации"""
        if person_name in self.enrollment_progress:
            del self.enrollment_progress[person_name]
        if person_name in self.enrollment_quality:
            del self.enrollment_quality[person_name]

