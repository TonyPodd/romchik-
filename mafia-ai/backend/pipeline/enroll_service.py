# backend/pipeline/enroll_service.py
from __future__ import annotations
import asyncio, time, json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np

# берём из нашего продвинутого модуля лица
from video.face_recog_adv import (
    MPFaceDetector, align_by_5pts, face_quality_score,
    make_embedder
)

# хранилище игроков (в проекте уже используется в stream_worker)
from storage import players as P


def _angle_from_pts5(pts5: np.ndarray) -> Tuple[float, float]:
    """
    Грубая оценка поворота головы по 5-точкам:
    вернём yaw (влево/вправо) и pitch (вверх/вниз) в градусах примерно.
    """
    # глаза: 0=left,1=right,2=nose,3=mouthL,4=mouthR (как в face_recog_adv)
    L, R = pts5[0], pts5[1]
    eye_dx = R[0] - L[0]
    eye_dy = R[1] - L[1]
    yaw = np.degrees(np.arctan2(eye_dy, max(1e-6, eye_dx)))  # наклон линии глаз
    # вертикальная: нос относительно середины глаз
    mid = (L + R) / 2.0
    pitch = np.degrees(np.arctan2(mid[1] - pts5[2][1], max(1e-6, abs(eye_dx))))
    return float(yaw), float(pitch)


@dataclass
class EnrollState:
    active: bool = False
    name: str = ""
    need: int = 12
    collected: int = 0
    gallery: List[np.ndarray] = field(default_factory=list)
    thumb: Optional[np.ndarray] = None
    last_hint: str = "Смотрите прямо"
    last_quality: float = 0.0
    # покрытие ракурсов
    yaw_left: int = 0
    yaw_right: int = 0
    pitch_up: int = 0
    pitch_down: int = 0


class EnrollService:
    """
    Тянет кадр из stream.get_last_jpeg(raw=True),
    детектит лицо, нормализует, фильтрует по качеству/разнообразию,
    копит эмбеддинги и сохраняет в players-хранилище.
    """
    def __init__(self, stream):
        self.stream = stream
        self.det = MPFaceDetector()
        # по умолчанию r100 если лежит, иначе mbf
        self.embedder = make_embedder(kind="r100", model_path=None)
        self.state = EnrollState()

    async def _get_last_frame(self) -> Optional[np.ndarray]:
        jpeg = await self.stream.get_last_jpeg(raw=True)
        if not jpeg:
            return None
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    def _diversity_ok(self, e: np.ndarray, S: List[np.ndarray], thr: float = 0.90) -> bool:
        if not S:
            return True
        S_ = np.stack(S, axis=0)
        S_ = S_ / (np.linalg.norm(S_, axis=1, keepdims=True) + 1e-6)
        e_ = e / (np.linalg.norm(e) + 1e-6)
        sim = float(np.max(S_ @ e_))
        return sim < thr

    async def start(self, name: str, need: int = 12) -> Dict[str, Any]:
        self.state = EnrollState(active=True, name=name.strip() or f"user_{int(time.time())}", need=max(6, need))
        return {"ok": True, "name": self.state.name, "need": self.state.need}

    async def cancel(self) -> Dict[str, Any]:
        self.state = EnrollState()
        return {"ok": True}

    async def step(self) -> Dict[str, Any]:
        if not self.state.active:
            return {"ok": False, "error": "no_session"}

        frame = await self._get_last_frame()
        if frame is None:
            return {"ok": False, "error": "no_frame"}

        faces = self.det.detect_and_landmarks(frame)
        if not faces:
            self.state.last_hint = "Подойдите ближе и смотрите на камеру"
            return self._status()

        # берём самое крупное лицо
        f = max(faces, key=lambda it: (it["bbox"][2]-it["bbox"][0])*(it["bbox"][3]-it["bbox"][1]))
        pts5 = f["pts5"]
        if pts5 is None:
            x1,y1,x2,y2 = f["bbox"]
            crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            aligned = cv2.resize(crop, (112, 112)) if crop.size else None
        else:
            aligned = align_by_5pts(frame, pts5, (112,112))

        if aligned is None:
            self.state.last_hint = "Не вижу лицо целиком"
            return self._status()

        q = face_quality_score(aligned)
        self.state.last_quality = q
        if q < 0.35:
            self.state.last_hint = "Недостаточно резкости/света"
            return self._status()

        # эмбеддинг
        e = self.embedder.embed(aligned)

        # разнообразие
        if not self._diversity_ok(e, self.state.gallery, thr=0.90):
            self.state.last_hint = "Смените ракурс (поверните голову)"
            return self._status()

        # обновим покрытие ракурсов
        if pts5 is not None:
            yaw, pitch = _angle_from_pts5(pts5)
            if yaw > 6: self.state.yaw_right += 1
            if yaw < -6: self.state.yaw_left += 1
            if pitch > 4: self.state.pitch_up += 1
            if pitch < -4: self.state.pitch_down += 1

        if self.state.thumb is None:
            self.state.thumb = aligned.copy()

        self.state.gallery.append(e)
        self.state.collected = len(self.state.gallery)

        # подсказка как в Face ID
        if self.state.yaw_left < 2:
            self.state.last_hint = "Поверните голову влево"
        elif self.state.yaw_right < 2:
            self.state.last_hint = "Поверните голову вправо"
        elif self.state.pitch_up < 2:
            self.state.last_hint = "Поднимите голову чуть выше"
        elif self.state.pitch_down < 2:
            self.state.last_hint = "Опустите голову чуть ниже"
        else:
            self.state.last_hint = "Отлично! Продолжайте"

        return self._status()

    def _status(self) -> Dict[str, Any]:
        s = self.state
        done = s.collected >= s.need
        return {
            "ok": True,
            "active": s.active,
            "name": s.name,
            "collected": s.collected,
            "need": s.need,
            "done": done,
            "hint": s.last_hint,
            "quality": round(s.last_quality, 3),
            "coverage": {
                "yaw_left": s.yaw_left,
                "yaw_right": s.yaw_right,
                "pitch_up": s.pitch_up,
                "pitch_down": s.pitch_down,
            }
        }

    async def commit(self) -> Dict[str, Any]:
        if not self.state.active or self.state.collected < 3:
            return {"ok": False, "error": "not_enough_samples"}

        mean = np.mean(np.stack(self.state.gallery, axis=0), axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-6)
        embedding = mean.astype(np.float32)

        # сохраняем в players-хранилище
        try:
            # ожидаем, что есть upsert/add API. Подстроиться под твой storage.players:
            # запишем средний эмбеддинг и галерею в meta
            pid = P.upsert_player(
                name=self.state.name,
                embedding=embedding.tolist(),
                meta={"gallery": [g.tolist() for g in self.state.gallery]}
            )
        except Exception:
            # запасной вариант — add_player
            pid = P.add_player(name=self.state.name, embedding=embedding.tolist(), meta={"gallery": []})

        # миниатюра
        try:
            from video.face_recog_adv import THUMBS_DIR
            THUMBS_DIR.mkdir(parents=True, exist_ok=True)
            if self.state.thumb is not None:
                cv2.imwrite(str(THUMBS_DIR / f"{pid}.jpg"), self.state.thumb)
        except Exception:
            pass

        name = self.state.name
        total = self.state.collected
        self.state = EnrollState()  # reset
        return {"ok": True, "id": pid, "name": name, "samples": total}


# pkill -9 -f "uvicorn app:app" && pkill -9 -f "vite" && pkill -9 -f "npm run dev" && sleep 1 && echo "✅ Все процессы завершены"
