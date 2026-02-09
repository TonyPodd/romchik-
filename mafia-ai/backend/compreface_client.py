from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class CompreFaceClient:
    def __init__(self) -> None:
        provider = (os.getenv("FACE_PROVIDER") or "").strip().upper()
        self.enabled = _env_bool("COMPREFACE_ENABLED", provider == "COMPREFACE")
        self.api_key = (os.getenv("COMPREFACE_API_KEY") or "").strip()
        self.timeout_sec = float(os.getenv("COMPREFACE_TIMEOUT_SEC", "4.0"))
        self.similarity_threshold = float(os.getenv("COMPREFACE_SIMILARITY_THRESHOLD", "0.82"))
        self.det_prob_threshold = float(os.getenv("COMPREFACE_DET_PROB_THRESHOLD", "0.8"))
        self.prediction_count = int(os.getenv("COMPREFACE_PREDICTION_COUNT", "3"))
        self.subject_prefix = (os.getenv("COMPREFACE_SUBJECT_PREFIX") or "player_").strip() or "player_"

        base_url = (os.getenv("COMPREFACE_URL") or "http://127.0.0.1:8002").rstrip("/")
        if base_url.endswith("/api/v1/recognition"):
            recognition_root = base_url
        elif base_url.endswith("/api/v1"):
            recognition_root = f"{base_url}/recognition"
        else:
            recognition_root = f"{base_url}/api/v1/recognition"

        self.recognition_root = recognition_root
        self.recognize_url = f"{self.recognition_root}/recognize"
        self.faces_url = f"{self.recognition_root}/faces"
        self.subjects_url = f"{self.recognition_root}/subjects"
        self._session = requests.Session()
        self._logged_disabled = False

    def is_active(self) -> bool:
        return self.enabled and bool(self.api_key)

    def subject_for_player(self, player_id: int) -> str:
        return f"{self.subject_prefix}{int(player_id)}"

    def _headers(self) -> Dict[str, str]:
        return {"x-api-key": self.api_key}

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.is_active():
            if not self._logged_disabled:
                print("[compreface] disabled (set COMPREFACE_ENABLED=1 and COMPREFACE_API_KEY)")
                self._logged_disabled = True
            raise RuntimeError("CompreFace is not configured")
        resp = self._session.request(
            method=method,
            url=url,
            params=params,
            files=files,
            headers=self._headers(),
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError("CompreFace response is not JSON object")
        return data

    def health(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "error": "disabled"}
        if not self.api_key:
            return {"ok": False, "error": "missing_api_key"}
        try:
            data = self._request_json("GET", self.subjects_url)
            return {"ok": True, "subjects": len(data.get("subjects") or [])}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def recognize_best(self, image_bytes: bytes) -> Tuple[Optional[str], float]:
        if not image_bytes:
            return None, 0.0
        try:
            data = self._request_json(
                "POST",
                self.recognize_url,
                params={
                    "prediction_count": self.prediction_count,
                    "det_prob_threshold": self.det_prob_threshold,
                },
                files={"file": ("frame.jpg", image_bytes, "image/jpeg")},
            )
        except Exception:
            return None, 0.0

        best_subject: Optional[str] = None
        best_similarity = 0.0
        for face in data.get("result") or []:
            subjects = face.get("subjects") or []
            for subj in subjects:
                subject = subj.get("subject")
                if not isinstance(subject, str) or not subject:
                    continue
                similarity = float(subj.get("similarity", 0.0))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_subject = subject

        if best_subject is None:
            return None, 0.0
        if best_similarity < self.similarity_threshold:
            return None, best_similarity
        return best_subject, best_similarity

    def add_face_to_subject(self, subject: str, image_bytes: bytes) -> bool:
        if not subject or not image_bytes:
            return False
        try:
            data = self._request_json(
                "POST",
                self.faces_url,
                params={
                    "subject": subject,
                    "det_prob_threshold": self.det_prob_threshold,
                },
                files={"file": ("face.jpg", image_bytes, "image/jpeg")},
            )
        except Exception:
            return False
        return bool(data.get("result"))

    def register_subject_samples(self, subject: str, samples: List[bytes]) -> Dict[str, Any]:
        if not self.is_active():
            return {"ok": False, "error": "compreface_not_configured", "added": 0, "total": len(samples)}
        if not subject:
            return {"ok": False, "error": "empty_subject", "added": 0, "total": len(samples)}
        cleaned = [s for s in samples if isinstance(s, (bytes, bytearray)) and len(s) > 0]
        if not cleaned:
            return {"ok": False, "error": "no_samples", "added": 0, "total": 0}

        # Re-enrolling must replace stale descriptors for the same slot/player.
        self.delete_subject(subject)
        added = 0
        for sample in cleaned:
            if self.add_face_to_subject(subject, bytes(sample)):
                added += 1
        return {"ok": added > 0, "added": added, "total": len(cleaned)}

    def list_subjects(self) -> List[str]:
        try:
            data = self._request_json("GET", self.subjects_url)
        except Exception:
            return []
        subjects = data.get("subjects")
        if not isinstance(subjects, list):
            return []
        return [s for s in subjects if isinstance(s, str)]

    def delete_subject(self, subject: str) -> bool:
        if not self.is_active() or not subject:
            return False
        url = f"{self.subjects_url}/{quote(subject, safe='')}"
        try:
            self._request_json("DELETE", url)
            return True
        except Exception:
            return False

    def delete_all_player_subjects(self) -> int:
        deleted = 0
        for subject in self.list_subjects():
            if subject.startswith(self.subject_prefix) and self.delete_subject(subject):
                deleted += 1
        return deleted


_client: Optional[CompreFaceClient] = None


def get_compreface_client() -> CompreFaceClient:
    global _client
    if _client is None:
        _client = CompreFaceClient()
    return _client
