from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import cv2
import numpy as np
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
        self.similarity_threshold = float(os.getenv("COMPREFACE_SIMILARITY_THRESHOLD", "0.64"))
        self.det_prob_threshold = float(os.getenv("COMPREFACE_DET_PROB_THRESHOLD", "0.55"))
        self.enroll_det_prob_threshold = float(os.getenv("COMPREFACE_ENROLL_DET_PROB_THRESHOLD", "0.25"))
        self.enroll_max_variants = max(1, int(os.getenv("COMPREFACE_ENROLL_MAX_VARIANTS", "6")))
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
        resp = None
        try:
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
        except requests.RequestException as e:
            status = getattr(resp, "status_code", None)
            body = ""
            if resp is not None:
                try:
                    body = (resp.text or "").strip()
                except Exception:
                    body = ""
            if status is not None and body:
                raise RuntimeError(f"HTTP {status}: {body[:300]}") from e
            raise RuntimeError(str(e)) from e
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
        candidates = self.recognize_candidates(image_bytes)
        if not candidates:
            return None, 0.0
        best_subject, best_similarity = candidates[0]
        if best_similarity < self.similarity_threshold:
            return None, best_similarity
        return best_subject, best_similarity

    @staticmethod
    def _extract_subject_candidates(data: Dict[str, Any]) -> List[Tuple[str, float]]:
        scored: Dict[str, float] = {}
        for face in data.get("result") or []:
            subjects = face.get("subjects") or []
            for subj in subjects:
                subject = subj.get("subject")
                if not isinstance(subject, str) or not subject:
                    continue
                similarity = float(subj.get("similarity", 0.0))
                prev = scored.get(subject)
                if prev is None or similarity > prev:
                    scored[subject] = similarity
        ordered = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        return [(str(subj), float(sim)) for subj, sim in ordered]

    def recognize_candidates(
        self,
        image_bytes: bytes,
        *,
        det_prob_threshold: Optional[float] = None,
        prediction_count: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        if not image_bytes:
            return []
        threshold = self.det_prob_threshold if det_prob_threshold is None else float(det_prob_threshold)
        pred_count = self.prediction_count if prediction_count is None else max(1, int(prediction_count))
        try:
            data = self._request_json(
                "POST",
                self.recognize_url,
                params={
                    "prediction_count": pred_count,
                    "det_prob_threshold": threshold,
                },
                files={"file": ("frame.jpg", image_bytes, "image/jpeg")},
            )
        except Exception:
            return []
        return self._extract_subject_candidates(data)

    @staticmethod
    def _extract_compreface_error(data: Dict[str, Any]) -> Optional[str]:
        direct = data.get("message")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()

        result = data.get("result")
        if isinstance(result, list):
            for row in result:
                if not isinstance(row, dict):
                    continue
                msg = row.get("message") or row.get("error")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
        if isinstance(result, dict):
            msg = result.get("message") or result.get("error")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
        return None

    def add_face_to_subject(
        self,
        subject: str,
        image_bytes: bytes,
        *,
        det_prob_threshold: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        if not subject or not image_bytes:
            return False, "empty_subject_or_image"
        threshold = self.det_prob_threshold if det_prob_threshold is None else float(det_prob_threshold)
        try:
            data = self._request_json(
                "POST",
                self.faces_url,
                params={
                    "subject": subject,
                    "det_prob_threshold": threshold,
                },
                files={"file": ("face.jpg", image_bytes, "image/jpeg")},
            )
        except Exception as e:
            return False, str(e)

        # CompreFace 1.2 for POST /recognition/faces returns:
        # {"image_id": "...", "subject": "..."}
        image_id = data.get("image_id")
        resp_subject = data.get("subject")
        if isinstance(image_id, str) and image_id.strip():
            if not resp_subject or str(resp_subject).strip() == subject:
                return True, None

        result = data.get("result")
        ok = bool(result)
        if ok:
            return True, None
        return False, self._extract_compreface_error(data) or "no_face_detected"

    @staticmethod
    def _encode_image(img_bgr: np.ndarray, quality: int = 95) -> bytes:
        if img_bgr is None or img_bgr.size == 0:
            return b""
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else b""

    def _sample_variants(self, image_bytes: bytes) -> List[bytes]:
        variants: List[bytes] = []
        if not image_bytes:
            return variants
        variants.append(bytes(image_bytes))

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return variants

        h, w = img.shape[:2]
        min_side = max(1, min(h, w))

        # Вариант 2: апскейл маленького лица (CompreFace хуже видит слишком маленькие кропы).
        if min_side < 260:
            scale = 260.0 / float(min_side)
            up = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            up_jpg = self._encode_image(up, quality=95)
            if up_jpg:
                variants.append(up_jpg)

        # Вариант 3: добавляем поля вокруг лица, чтобы детектор не "резался" по краям.
        border = max(8, int(0.18 * float(min_side)))
        with_border = cv2.copyMakeBorder(
            img,
            border,
            border,
            border,
            border,
            cv2.BORDER_REPLICATE,
        )
        with_border_jpg = self._encode_image(with_border, quality=95)
        if with_border_jpg:
            variants.append(with_border_jpg)

        # Вариант 4: мягкое улучшение контраста/яркости.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.06, beta=6)
        enhanced_jpg = self._encode_image(enhanced, quality=95)
        if enhanced_jpg:
            variants.append(enhanced_jpg)

        # Вариант 5: небольшой поворот (некоторые кадры приходят с наклоном головы/камеры).
        hh, ww = img.shape[:2]
        center = (ww * 0.5, hh * 0.5)
        for angle in (-7.0, 7.0):
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rot = cv2.warpAffine(
                img,
                M,
                (ww, hh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            rot_jpg = self._encode_image(rot, quality=95)
            if rot_jpg:
                variants.append(rot_jpg)

        # Дедупликация по содержимому.
        dedup: List[bytes] = []
        seen = set()
        for blob in variants:
            key = hash(blob)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(blob)
        return dedup

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
        errors: List[str] = []
        for sample in cleaned:
            sample_ok = False
            last_error: Optional[str] = None
            for variant in self._sample_variants(bytes(sample))[: self.enroll_max_variants]:
                ok, error = self.add_face_to_subject(
                    subject,
                    variant,
                    det_prob_threshold=self.enroll_det_prob_threshold,
                )
                if ok:
                    added += 1
                    sample_ok = True
                    break
                if error:
                    last_error = error
            if not sample_ok and last_error and len(errors) < 3:
                errors.append(last_error)

        rolled_back = False
        if added == 0:
            # Не оставляем "пустой" subject в коллекции, если лицо не удалось добавить ни разу.
            rolled_back = self.delete_subject(subject)

        response: Dict[str, Any] = {
            "ok": added > 0,
            "added": added,
            "failed": max(0, len(cleaned) - added),
            "total": len(cleaned),
            "rolled_back": rolled_back,
        }
        if errors:
            response["sample_errors"] = errors
        if added == 0 and errors:
            response["error"] = errors[0]
        return response

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
