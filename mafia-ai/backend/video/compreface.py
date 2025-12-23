# backend/video/compreface.py
"""CompreFace integration for face recognition"""

import os
import base64
from typing import Optional, Dict, Any, List
import httpx
import numpy as np
import cv2


class CompreFaceClient:
    """Client for CompreFace face recognition API"""

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        api_key: Optional[str] = None,
        recognition_api_key: Optional[str] = None
    ):
        """
        Initialize CompreFace client.

        Args:
            base_url: CompreFace server URL
            api_key: API key for face collection (deprecated, use recognition_api_key)
            recognition_api_key: API key for recognition service
        """
        self.base_url = base_url.rstrip("/")
        self.recognition_api_key = recognition_api_key or api_key or os.getenv("COMPREFACE_API_KEY", "")
        self.timeout = 30.0

    def _encode_image(self, image_bgr: np.ndarray) -> bytes:
        """Encode BGR image to JPEG bytes"""
        success, buffer = cv2.imencode('.jpg', image_bgr)
        if not success:
            raise ValueError("Failed to encode image")
        return buffer.tobytes()

    async def add_face(
        self,
        subject: str,
        image_bgr: np.ndarray,
        det_prob_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Add a face to the recognition database.

        Args:
            subject: Subject name (player name)
            image_bgr: Face image in BGR format (OpenCV)
            det_prob_threshold: Detection probability threshold (0.0-1.0)

        Returns:
            Response from CompreFace API
        """
        url = f"{self.base_url}/api/v1/recognition/faces"
        params = {
            "subject": subject,
            "det_prob_threshold": det_prob_threshold
        }
        headers = {
            "x-api-key": self.recognition_api_key
        }

        image_bytes = self._encode_image(image_bgr)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
            response = await client.post(url, params=params, headers=headers, files=files)
            response.raise_for_status()
            return response.json()

    async def recognize_face(
        self,
        image_bgr: np.ndarray,
        det_prob_threshold: float = 0.8,
        limit: int = 1,
        prediction_count: int = 1,
        face_plugins: str = ""
    ) -> Dict[str, Any]:
        """
        Recognize faces in an image.

        Args:
            image_bgr: Image in BGR format (OpenCV)
            det_prob_threshold: Detection probability threshold
            limit: Maximum number of faces to return
            prediction_count: Number of predictions per face
            face_plugins: Comma-separated list of plugins (age,gender,pose,etc)

        Returns:
            Response from CompreFace API with detected faces and matches
        """
        url = f"{self.base_url}/api/v1/recognition/recognize"
        params = {
            "det_prob_threshold": det_prob_threshold,
            "limit": limit,
            "prediction_count": prediction_count
        }
        if face_plugins:
            params["face_plugins"] = face_plugins

        headers = {
            "x-api-key": self.recognition_api_key
        }

        image_bytes = self._encode_image(image_bgr)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
            response = await client.post(url, params=params, headers=headers, files=files)
            response.raise_for_status()
            return response.json()

    async def delete_subject(self, subject: str) -> Dict[str, Any]:
        """
        Delete all faces of a subject from the database.

        Args:
            subject: Subject name to delete

        Returns:
            Response from CompreFace API
        """
        url = f"{self.base_url}/api/v1/recognition/faces"
        params = {"subject": subject}
        headers = {"x-api-key": self.recognition_api_key}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.delete(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()

    async def list_subjects(self) -> Dict[str, Any]:
        """
        List all subjects in the database.

        Returns:
            Response from CompreFace API with list of subjects
        """
        url = f"{self.base_url}/api/v1/recognition/subjects"
        headers = {"x-api-key": self.recognition_api_key}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    async def verify_face(
        self,
        image_bgr: np.ndarray,
        subject: str,
        det_prob_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Verify if a face belongs to a specific subject.

        Args:
            image_bgr: Image in BGR format
            subject: Subject name to verify against
            det_prob_threshold: Detection probability threshold

        Returns:
            Response from CompreFace API with verification result
        """
        url = f"{self.base_url}/api/v1/recognition/faces/{subject}/verify"
        params = {"det_prob_threshold": det_prob_threshold}
        headers = {"x-api-key": self.recognition_api_key}

        image_bytes = self._encode_image(image_bgr)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
            response = await client.post(url, params=params, headers=headers, files=files)
            response.raise_for_status()
            return response.json()


def parse_compreface_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse CompreFace recognition result to our internal format.

    Args:
        result: CompreFace API response

    Returns:
        List of detected faces with bboxes and player info
    """
    from storage import players as P
    
    faces = []
    
    # Получаем список всех игроков для сопоставления имени с ID
    registered_players = P.list_players()
    name_to_id: Dict[str, int] = {p.get("name", ""): p["id"] for p in registered_players if p.get("name")}

    for detection in result.get("result", []):
        box = detection.get("box", {})
        x_min = box.get("x_min", 0)
        y_min = box.get("y_min", 0)
        x_max = box.get("x_max", 0)
        y_max = box.get("y_max", 0)

        # Get best match
        subjects = detection.get("subjects", [])
        player_id = None
        player_name = None
        similarity = 0.0

        if subjects:
            best_match = subjects[0]
            player_name = best_match.get("subject")
            similarity = best_match.get("similarity", 0.0)

            # Сопоставляем имя (subject) с player_id из локальной БД
            if player_name and player_name in name_to_id:
                player_id = name_to_id[player_name]
            elif player_name:
                # Если имя не найдено, пытаемся найти по частичному совпадению
                for name, pid in name_to_id.items():
                    if name and player_name and (name.lower() == player_name.lower() or name.lower() in player_name.lower() or player_name.lower() in name.lower()):
                        player_id = pid
                        player_name = name  # Используем правильное имя из БД
                        break

        faces.append({
            "bbox": (x_min, y_min, x_max, y_max),
            "id": player_id,
            "name": player_name,
            "sim": similarity,
            "compreface_recognized": player_id is not None  # Флаг что лицо распознано CompreFace
        })

    return faces
