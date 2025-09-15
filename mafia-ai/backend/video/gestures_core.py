# backend/video/gestures_core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum, auto

Point = Tuple[float, float]
BBox  = Tuple[int, int, int, int]

class GestureLabel(Enum):
    UNKNOWN      = auto()
    FIST         = auto()
    OPEN_PALM    = auto()
    ONE          = auto()
    TWO          = auto()
    THREE        = auto()
    FOUR         = auto()
    FIVE         = auto()
    POINT        = auto()
    THUMB_UP     = auto()
    THUMB_DOWN   = auto()
    OK_SIGN      = auto()
    PEACE_V      = auto()   # «V», инд. и средний разведены
    ROCK         = auto()   # «\m/», индекс + мизинец
    CALL_ME      = auto()   # «🤙», большой + мизинец
    PINCH        = auto()   # щепотка (большой + указательный)


@dataclass
class HandState:
    track_id: int
    bbox: BBox
    center: Tuple[int,int]
    handedness: str                  # "Left" / "Right"
    handedness_score: float
    landmarks: List[Point]           # 21 точка в пикселях
    extended: List[bool]             # [thumb,index,middle,ring,pinky]
    raw_label: GestureLabel          # быстрая классификация на кадре
    smooth_label: GestureLabel       # результат сглаживания по окну
    confidence: float                # 0..1 (доля голосов за итоговую метку)
    score: float = 0.0               # базовый score детектора

# --------- Геометрия ---------
def bbox_from_landmarks(lms: List[Point]) -> BBox:
    xs = [p[0] for p in lms]; ys = [p[1] for p in lms]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    return (x1, y1, x2, y2)

def center_of(b: BBox) -> Tuple[int,int]:
    x1,y1,x2,y2 = b
    return ((x1+x2)//2, (y1+y2)//2)

def iou(a: BBox, b: BBox) -> float:
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ra = max(1, (ax2-ax1)*(ay2-ay1))
    rb = max(1, (bx2-bx1)*(by2-by1))
    return inter / (ra + rb - inter + 1e-6)
