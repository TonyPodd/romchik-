# backend/video/hand_pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import mediapipe as mp

from .yolo_hand import YoloHandDetector, YoloDet
from .hand_gestures import classify_gesture, finger_state

@dataclass
class HandOut:
    bbox: Tuple[int,int,int,int]   # x,y,w,h
    center: Tuple[int,int]
    handedness: str
    count: int
    extended: Dict[str,bool]
    gesture: str
    conf: float

class HandPipeline:
    def __init__(self, max_hands: int = 20, yolo_conf: float = 0.25, yolo_iou: float = 0.45):
        self.det = YoloHandDetector(conf_thr=yolo_conf, iou_thr=yolo_iou, input_size=640)
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True, model_complexity=1,
            max_num_hands=1, min_detection_confidence=0.35, min_tracking_confidence=0.35
        )
        self.max_hands = max_hands

    def _handedness_text(self, res) -> str:
        try:
            return res.multi_handedness[0].classification[0].label
        except Exception:
            return "Unknown"

    def process(self, frame_bgr) -> List[HandOut]:
        H,W = frame_bgr.shape[:2]
        out: List[HandOut] = []

        dets: List[YoloDet] = []
        if self.det.available():
            try:
                dets = self.det.detect(frame_bgr)
            except Exception as e:
                print(f"[hand_pipeline] YOLO failure: {e} -> fallback MP")
                try: self.det._broken = True
                except: pass
                dets = []

        if not dets:
            res = self.mp_hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                    xs = [pt.x for pt in lm.landmark]; ys = [pt.y for pt in lm.landmark]
                    x1 = int(max(0, min(xs) * W)); x2 = int(min(1.0, max(xs)) * W)
                    y1 = int(max(0, min(ys) * H)); y2 = int(min(1.0, max(ys)) * H)
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    fs = finger_state(lm); gest = classify_gesture(lm)
                    out.append(HandOut(bbox=(x1,y1,x2-x1,y2-y1), center=(cx,cy),
                                       handedness=handed.classification[0].label if handed else "Unknown",
                                       count=fs.count, extended=fs.extended, gesture=gest, conf=0.5))
            return out[: self.max_hands]

        dets = sorted(dets, key=lambda d: d.conf, reverse=True)[: self.max_hands]
        for d in dets:
            x1,y1,x2,y2 = d.xyxy
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0: continue
            res = self.mp_hands.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if not res.multi_hand_landmarks: continue
            lm = res.multi_hand_landmarks[0].landmark
            fs = finger_state(lm); gest = classify_gesture(lm)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            out.append(HandOut(
                bbox=(x1,y1,x2-x1,y2-y1), center=(cx,cy),
                handedness=self._handedness_text(res),
                count=fs.count, extended=fs.extended, gesture=gest, conf=d.conf
            ))
        return out
