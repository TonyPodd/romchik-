# backend/video/gestures.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    raise RuntimeError("Install mediapipe first: pip install mediapipe") from e


@dataclass
class HandInfo:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    handedness: str                   # 'Left' or 'Right'
    extended: Dict[str, bool]         # thumb, index, middle, ring, pinky
    count: int                        # number of extended fingers
    center: Tuple[int, int]           # cx, cy (pixels)


@dataclass
class GestureResult:
    # High-level classification per frame
    digit: Optional[int]              # 0..10 if confidently inferred
    fist_on_table: bool               # closed fist near bottom of frame (table area)
    pistol: bool                      # thumb+index extended, others folded
    hands: List[HandInfo]             # raw per-hand info for debugging


class GestureDetector:
    def __init__(
        self,
        table_y_ratio: float = 0.80,              # нижняя часть кадра считается столом (80% высоты и ниже)
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 2,
    ):
        self.table_y_ratio = table_y_ratio
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,
        )

    def _norm_to_px(self, lm, w, h):
        return int(lm.x * w), int(lm.y * h)

    def _bbox_from_landmarks(self, landmarks_px: List[Tuple[int,int]]) -> Tuple[int,int,int,int]:
        xs = [p[0] for p in landmarks_px]
        ys = [p[1] for p in landmarks_px]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return x1, y1, x2 - x1, y2 - y1

    def _finger_extended(self, lm_px, finger: str) -> bool:
        """
        Простые эвристики в пикселях:
        - Для указ., ср., безым., мизинца: кончик выше (меньше y) одного-двух суставов
        - Для большого: сравнение по x в зависимости от ладонности плюс угол между фалангами
        """
        # lm indices (MediaPipe Hands):
        # 0:wrist,
        # thumb: 1,2,3,4
        # index: 5,6,7,8
        # middle: 9,10,11,12
        # ring: 13,14,15,16
        # pinky: 17,18,19,20
        def up(i_tip, i_pip):  # "tip above pip" in image coordinates (smaller y)
            return lm_px[i_tip][1] < lm_px[i_pip][1] - 4  # небольшой зазор

        if finger == "index":
            return up(8, 6)
        if finger == "middle":
            return up(12, 10)
        if finger == "ring":
            return up(16, 14)
        if finger == "pinky":
            return up(20, 18)
        if finger == "thumb":
            # thumb: сравним направление фаланги (2->4) и «раскрытие» относительно запястья
            # простая эвристика: расстояние tip (4) до запястья (0) должно быть больше, чем 3 до запястья
            wrist = lm_px[0]
            d_tip = np.hypot(lm_px[4][0]-wrist[0], lm_px[4][1]-wrist[1])
            d_3   = np.hypot(lm_px[3][0]-wrist[0], lm_px[3][1]-wrist[1])
            return d_tip > d_3 + 6
        return False

    def _pistol(self, ext: Dict[str,bool]) -> bool:
        # «пистолет»: большой и указательный — да; остальные — нет
        return ext["thumb"] and ext["index"] and not (ext["middle"] or ext["ring"] or ext["pinky"])

    def _closed_fist(self, ext: Dict[str,bool]) -> bool:
        return not (ext["thumb"] or ext["index"] or ext["middle"] or ext["ring"] or ext["pinky"])

    def process_frame(self, frame_bgr: np.ndarray) -> GestureResult:
        """
        Возвращает агрегированную оценку кадра и подробности по каждой руке.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(frame_rgb)

        hands_out: List[HandInfo] = []
        if res.multi_hand_landmarks and res.multi_handedness:
            for lms, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                lm_px = [self._norm_to_px(lm, w, h) for lm in lms.landmark]
                bbox = self._bbox_from_landmarks(lm_px)
                cx = int(np.mean([p[0] for p in lm_px]))
                cy = int(np.mean([p[1] for p in lm_px]))
                label = handedness.classification[0].label  # 'Left' / 'Right'

                ext = {
                    "thumb":  self._finger_extended(lm_px, "thumb"),
                    "index":  self._finger_extended(lm_px, "index"),
                    "middle": self._finger_extended(lm_px, "middle"),
                    "ring":   self._finger_extended(lm_px, "ring"),
                    "pinky":  self._finger_extended(lm_px, "pinky"),
                }
                count = sum(ext.values())

                hands_out.append(HandInfo(bbox=bbox, handedness=label, extended=ext, count=count, center=(cx, cy)))

        # Аггрегации на уровне кадра
        total_fingers = sum(h.count for h in hands_out)
        digit: Optional[int] = None
        pistol = any(self._pistol(h.extended) for h in hands_out)

        # цифры: 1–5 одной рукой, 6–10 — сумма двух рук
        if len(hands_out) == 0:
            digit = None
        elif len(hands_out) == 1:
            # стабилизируем в диапазоне 0..5
            c = max(0, min(5, hands_out[0].count))
            digit = c if c > 0 else 0  # 0 трактуем как «ничего не показывает»
        else:
            # две руки: сумма 0..10
            s = max(0, min(10, total_fingers))
            digit = s if s > 0 else 0

        # кулак «на столе»: закрыт и в нижней части кадра
        table_y = int(self.table_y_ratio * h)
        fist_on_table = any(
            (self._closed_fist(hh.extended) and hh.center[1] >= table_y)
            for hh in hands_out
        )

        return GestureResult(
            digit=digit,
            fist_on_table=fist_on_table,
            pistol=pistol,
            hands=hands_out
        )
