# backend/video/gestures_engine.py
from __future__ import annotations
import time, math
from collections import deque
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import cv2
import numpy as np

from .gestures_core import HandState, GestureLabel, bbox_from_landmarks, center_of, iou
from .gestures_hagrid import HaGRIDClassifier
from .gestures_yolo import YOLOHandDetector

# --- MediaPipe индексы ---
WRIST=0
THUMB_CMC,THUMB_MCP,THUMB_IP,THUMB_TIP = 1,2,3,4
INDEX_MCP,INDEX_PIP,INDEX_DIP,INDEX_TIP = 5,6,7,8
MIDDLE_MCP,MIDDLE_PIP,MIDDLE_DIP,MIDDLE_TIP = 9,10,11,12
RING_MCP,RING_PIP,RING_DIP,RING_TIP = 13,14,15,16
PINKY_MCP,PINKY_PIP,PINKY_DIP,PINKY_TIP = 17,18,19,20

FINGERS = [
    (THUMB_MCP, THUMB_IP, THUMB_TIP),
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
]

def _v(a,b): 
    a=np.array(a); b=np.array(b); return b-a
def _len(v): 
    v=np.array(v); return float(np.linalg.norm(v)+1e-9)
def _angle(a,b,c)->float:
    ba = _v(b,a); bc = _v(b,c)
    cosang = np.clip(ba.dot(bc)/(_len(ba)*_len(bc)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

# --- метрики пальцев ---
def finger_curl(lms: List[Tuple[float,float]], mcp:int, pip:int, tip:int) -> float:
    """
    Возвращает степень сгиба [0..1]: 0 — палец прямой, 1 — сильно согнут.
    Используем углы и нормированные длины.
    """
    ang = _angle(lms[mcp], lms[pip], lms[tip])  # ~180 при прямом
    # нормированная длина MCP->TIP относительно MCP->PIP
    d_mt = _len(_v(lms[mcp], lms[tip]))
    d_mp = _len(_v(lms[mcp], lms[pip])) + 1e-6
    ratio = d_mt / d_mp  # меньше при согнутом
    # переводим в сгиб: чем меньше угол к 180 и меньше ratio — тем больше curl
    c_ang = max(0.0, min(1.0, (180.0 - ang)/100.0))      # 0..~1
    c_len = max(0.0, min(1.0, (1.2 - ratio)/0.8))        # ~0 при прямом, ~1 при сильном сгибе
    curl = 0.65*c_ang + 0.35*c_len
    return float(max(0.0, min(1.0, curl)))

def thumb_curl(lms: List[Tuple[float,float]]) -> float:
    # большой палец: берём угол MCP-IP-TIP и относим TIP к ладони
    ang = _angle(lms[THUMB_MCP], lms[THUMB_IP], lms[THUMB_TIP])
    c_ang = max(0.0, min(1.0, (180.0 - ang)/90.0))
    palm = _len(_v(lms[WRIST], lms[MIDDLE_MCP])) + 1e-6
    d = _len(_v(lms[THUMB_TIP], lms[WRIST]))/palm
    c_len = max(0.0, min(1.0, (0.9 - d)/0.6))
    return float(max(0.0, min(1.0, 0.7*c_ang + 0.3*c_len)))

def finger_splay(lms: List[Tuple[float,float]], base_a:int, tip_a:int, base_b:int, tip_b:int) -> float:
    """
    Развод между лучами (MCP->TIP) двух пальцев в градусах (0..~60).
    """
    ra = _v(lms[base_a], lms[tip_a]); rb = _v(lms[base_b], lms[tip_b])
    cosang = np.clip(ra.dot(rb)/(_len(ra)*_len(rb)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def extended_from_curl(c: float, th: float=0.35) -> bool:
    return c <= th

class _Track:
    def __init__(self, tid:int):
        self.id = tid
        self.history = deque(maxlen=8)     # (center, bbox)
        self.labels  = deque(maxlen=8)     # GestureLabel
        self.curl_ema = np.zeros(5, dtype=np.float32)  # EMA по curl для [thumb..pinky]
        self.last_ts = time.time()
        self._init = False

    def update_ema(self, curl_vec: np.ndarray, alpha: float = 0.35):
        if not self._init:
            self.curl_ema[:] = curl_vec
            self._init = True
        else:
            self.curl_ema = alpha*curl_vec + (1.0-alpha)*self.curl_ema

class GestureEngine:
    """
    Детектор + лэндмарки (MediaPipe/YOLO) -> метрики пальцев -> шаблоны жестов -> сглаживание.
    Опц. HaGRID (CNN) с фьюжном.
    """
    def __init__(
        self,
        max_hands:int=6,
        det_conf:float=0.6,
        track_conf:float=0.6,
        detector: str = "mp",
        yolo_model_path: Optional[str] = None,
        yolo_conf: float = 0.25,
        yolo_iou: float = 0.45,
        yolo_imgsz: int = 640,
        use_mp_landmarks: bool = True,
        use_hagrid: bool = False,
        hagrid_model_path: Optional[str] = None,
    ):
        self.detector = detector.lower()
        self._tracks: Dict[int,_Track] = {}
        self._next_id = 1

        self._hagrid = HaGRIDClassifier(model_path=Path(hagrid_model_path)) if use_hagrid else None

        if self.detector == "mp":
            import mediapipe as mp
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=det_conf,
                min_tracking_confidence=track_conf,
                model_complexity=1
            )
            self.hands_crop = None
            self.yolo = None
        elif self.detector == "yolo":
            if not yolo_model_path:
                raise ValueError("detector='yolo' требует yolo_model_path (.pt)")
            self.yolo = YOLOHandDetector(yolo_model_path, conf=yolo_conf, iou=yolo_iou, imgsz=yolo_imgsz, device="cpu")
            self.hands = None
            if use_mp_landmarks:
                import mediapipe as mp
                self.hands_crop = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=det_conf,
                    min_tracking_confidence=track_conf,
                    model_complexity=1
                )
            else:
                self.hands_crop = None
        else:
            raise ValueError("detector must be 'mp' or 'yolo'")

    # ----- трекинг -----
    def _assign_tracks(self, boxes: List[Tuple[int,int,int,int]]) -> List[int]:
        tids = [-1]*len(boxes)
        used = set()
        for ti, tr in list(self._tracks.items()):
            last_bbox = tr.history[-1][1] if tr.history else None
            if last_bbox is None: continue
            lc = center_of(last_bbox)
            best_j, best_cost = -1, 1e9
            for j,b in enumerate(boxes):
                if j in used: continue
                c = center_of(b)
                d = (c[0]-lc[0])**2 + (c[1]-lc[1])**2
                o = 1.0 - iou(last_bbox, b)
                cost = 0.7*o + 0.3*(d/40000.0)
                if cost < best_cost:
                    best_cost, best_j = cost, j
            if best_j >= 0 and best_cost < 0.9:
                tids[best_j] = ti; used.add(best_j)
                tr.last_ts = time.time()
        for j,b in enumerate(boxes):
            if tids[j] == -1:
                tid = self._next_id; self._next_id+=1
                self._tracks[tid] = _Track(tid)
                tids[j] = tid
        # уборка
        now = time.time()
        dead = [ti for ti,tr in self._tracks.items() if (now - tr.last_ts) > 2.0 and (not tr.history)]
        for ti in dead: self._tracks.pop(ti, None)
        return tids

    # ----- ядро: метрики -> метка -----
    def _classify_from_metrics(self, curls: np.ndarray, lms: List[Tuple[float,float]]) -> Tuple[GestureLabel, float]:
        """
        На основе curl[5] (0 прямой, 1 согнут) и геометрии (splay, дистанции) выдаём (label, rule_score 0..1).
        """
        t,i,m,r,p = curls.tolist()

        # extended по порогам (адаптивным)
        ext = np.array([t,i,m,r,p]) <= np.array([0.35, 0.30, 0.32, 0.35, 0.38])

        # счётчик
        k = int(ext.sum())

        # дополнительные геометрии
        score = 0.0
        if lms and len(lms) >= 21:
            # «V»: большой не обязателен, важно что index & middle выпрямлены и РАЗВЕДЕНЫ
            s_im = finger_splay(lms, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP)
            # CALL/ROCK требуют специфических пар
            d_ok   = _len(_v(lms[THUMB_TIP], lms[INDEX_TIP]))
            d_palm = _len(_v(lms[WRIST], lms[MIDDLE_MCP])) + 1e-9
            ratio_ok = d_ok / d_palm

        # БАЗОВЫЕ ЖЕСТЫ
        if k == 0:
            return GestureLabel.FIST, 0.9
        if k == 5 and (t<0.45 and i<0.45 and m<0.45 and r<0.45 and p<0.45):
            # «ладонь вперёд» против «пять» — считаем одним классом FIVE,
            # OPEN_PALM оставим как синоним при необходимости
            return GestureLabel.FIVE, 0.9

        # ONE/TWO/THREE/FOUR чётче, чтобы меньше путать 2/3 и 4/5
        if ext[1] and not ext[2] and not ext[3] and not ext[4]:
            return GestureLabel.ONE, 0.9
        if ext[1] and ext[2] and not ext[3] and not ext[4]:
            # отличаем PEACE (большой может быть любым) — по разводу
            if lms and s_im >= 20.0:
                return GestureLabel.PEACE_V, 0.85
            return GestureLabel.TWO, 0.85
        if ext[1] and ext[2] and ext[3] and not ext[4]:
            return GestureLabel.THREE, 0.85
        if ext[1] and ext[2] and ext[3] and ext[4] and not ext[0]:
            return GestureLabel.FOUR, 0.85

        # OK / PINCH
        if lms and ratio_ok < 0.35:
            # если ещё и средний/безымянный согнуты — скорее PINCH, иначе OK
            if not ext[2] and not ext[3]:
                return GestureLabel.PINCH, 0.8
            return GestureLabel.OK_SIGN, 0.8

        # LIKE/DISLIKE (только большой выпрямлен)
        if ext[0] and not any([ext[1],ext[2],ext[3],ext[4]]):
            # ориентируемся по вертикали
            thumb_y = lms[THUMB_TIP][1] if lms else 0.0
            wrist_y = lms[WRIST][1] if lms else thumb_y+100.0
            if thumb_y < wrist_y - 10:
                return GestureLabel.THUMB_UP, 0.85
            if thumb_y > wrist_y + 10:
                return GestureLabel.THUMB_DOWN, 0.85

        # ROCK: индекс и мизинец прямые, средний/безымянный согнуты (большой любой)
        if ext[1] and not ext[2] and not ext[3] and ext[4]:
            return GestureLabel.ROCK, 0.8

        # CALL_ME: большой и мизинец прямые, ост. согнуты
        if ext[0] and not ext[1] and not ext[2] and not ext[3] and ext[4]:
            return GestureLabel.CALL_ME, 0.8

        # OPEN_PALM (если не сработала «пять» из-за большого)
        if k>=4 and (i<0.45 and m<0.45 and r<0.45 and p<0.48):
            return GestureLabel.OPEN_PALM, 0.7

        return GestureLabel.UNKNOWN, 0.3

    # ----- основной проход -----
    def process(self, frame_bgr) -> List[HandState]:
        H, W = frame_bgr.shape[:2]
        hands_out: List[HandState] = []

        # 1) боксы
        boxes: List[Tuple[int,int,int,int]] = []
        base_infos: List[Dict] = []
        if self.detector == "mp":
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)
            if res.multi_hand_landmarks:
                for lmset, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                    lms = [(min(max(0.0,l.x),1.0)*W, min(max(0.0,l.y),1.0)*H) for l in lmset.landmark]
                    bbox = bbox_from_landmarks(lms)
                    boxes.append(bbox)
                    base_infos.append({
                        "lms": lms,
                        "handed": handedness.classification[0].label,
                        "score": float(handedness.classification[0].score)
                    })
        else:
            dets = self.yolo.detect(frame_bgr)
            boxes = [d["bbox"] for d in dets]
            base_infos = dets

        if not boxes:
            for tr in self._tracks.values():
                tr.history.clear(); tr.labels.clear()
            return []

        # 2) трекинг
        tids = self._assign_tracks(boxes)

        # 3) расчёт метрик + классификация
        for j, bbox in enumerate(boxes):
            x1,y1,x2,y2 = bbox
            tid = tids[j]
            handed = "Right"; score = 0.5
            lms: List[Tuple[float,float]] = []

            if self.detector == "mp":
                lms = base_infos[j]["lms"]
                handed = base_infos[j]["handed"]
                score = base_infos[j]["score"]
            else:
                if self.hands_crop is not None:
                    crop = frame_bgr[y1:y2, x1:x2]
                    if crop.size > 0:
                        rgbc = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        res = self.hands_crop.process(rgbc)
                        if res.multi_hand_landmarks:
                            lmset = res.multi_hand_landmarks[0]
                            Hc, Wc = crop.shape[:2]
                            lms = [(x1 + min(max(0.0,l.x),1.0)*Wc, y1 + min(max(0.0,l.y),1.0)*Hc) for l in lmset.landmark]
                            if res.multi_handedness:
                                handed = res.multi_handedness[0].classification[0].label
                                score  = float(res.multi_handedness[0].classification[0].score)

            # метрики curl
            if lms:
                curls = np.array([
                    thumb_curl(lms),
                    finger_curl(lms, *FINGERS[1]),
                    finger_curl(lms, *FINGERS[2]),
                    finger_curl(lms, *FINGERS[3]),
                    finger_curl(lms, *FINGERS[4]),
                ], dtype=np.float32)
            else:
                curls = np.array([0.5,0.5,0.5,0.5,0.5], dtype=np.float32)

            # трек и EMA по curl
            tr = self._tracks[tid]
            tr.history.append((center_of(bbox), bbox))
            tr.update_ema(curls)

            # rule-based классификация
            raw_label, rule_score = self._classify_from_metrics(tr.curl_ema, lms)

            # HaGRID фьюжн (если доступен)
            if self._hagrid is not None and self._hagrid.available:
                padx = int(0.12*(x2-x1)); pady = int(0.12*(y2-y1))
                xx1,yy1 = max(0, x1-padx), max(0, y1-pady)
                xx2,yy2 = min(W, x2+padx), min(H, y2+pady)
                crop = frame_bgr[yy1:yy2, xx1:xx2]
                if crop.size > 0:
                    proba = self._hagrid.predict_proba(crop)
                    if proba is not None:
                        # берём только знакомые классы и переводим в наши метки
                        name_idx = int(np.argmax(proba))
                        name = self._hagrid.class_names[name_idx] if name_idx < len(self._hagrid.class_names) else None
                        if name:
                            # перевод названия в GestureLabel
                            from .gestures_hagrid import CLASS2LABEL
                            cnn_label = CLASS2LABEL.get(name, GestureLabel.UNKNOWN)
                            cnn_conf  = float(proba[name_idx])
                            # простая стратегия: если cnn уверена сильно — берём её метку, иначе rule
                            if cnn_conf >= max(0.60, rule_score+0.05):
                                raw_label = cnn_label
                                rule_score = cnn_conf

            # голосование по меткам (как было)
            tr.labels.append(raw_label)
            values = list(tr.labels)
            smooth = max(set(values), key=values.count)
            conf = values.count(smooth) / max(1, len(values))

            # extended как бинаризация EMA-кёрлов для вывода
            ext_bin = [bool(v) for v in (tr.curl_ema <= np.array([0.35,0.30,0.32,0.35,0.38]))]

            hs = HandState(
                track_id=tid,
                bbox=bbox,
                center=center_of(bbox),
                handedness=handed,
                handedness_score=score,
                landmarks=lms,
                extended=ext_bin,
                raw_label=raw_label,
                smooth_label=smooth,
                confidence=float(conf*0.6 + 0.4*rule_score),
                score=score
            )
            hands_out.append(hs)

        return hands_out

# --- отрисовка (без изменений по сути) ---
def draw_hands(frame, hands: List[HandState]) -> None:
    conns = [
        (WRIST, THUMB_CMC),(THUMB_CMC,THUMB_MCP),(THUMB_MCP,THUMB_IP),(THUMB_IP,THUMB_TIP),
        (WRIST, INDEX_MCP),(INDEX_MCP,INDEX_PIP),(INDEX_PIP,INDEX_DIP),(INDEX_DIP,INDEX_TIP),
        (WRIST, MIDDLE_MCP),(MIDDLE_MCP,MIDDLE_PIP),(MIDDLE_PIP,MIDDLE_DIP),(MIDDLE_DIP,MIDDLE_TIP),
        (WRIST, RING_MCP),(RING_MCP,RING_PIP),(RING_PIP,RING_DIP),(RING_DIP,RING_TIP),
        (WRIST, PINKY_MCP),(PINKY_MCP,PINKY_PIP),(PINKY_PIP,PINKY_DIP),(PINKY_DIP,PINKY_TIP),
    ]
    for h in hands:
        x1,y1,x2,y2 = h.bbox
        color = (102,182,255)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

        if len(h.landmarks) == 21:
            for a,b in conns:
                ax,ay = map(int, h.landmarks[a]); bx,by = map(int, h.landmarks[b])
                cv2.line(frame, (ax,ay), (bx,by), (180,180,180), 1, cv2.LINE_AA)
            for (x,y) in h.landmarks:
                cv2.circle(frame, (int(x),int(y)), 2, (255,255,255), -1, cv2.LINE_AA)

        label = f"{h.smooth_label.name}  id:{h.track_id}  conf:{h.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, h.handedness, (x1, y1-28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)
