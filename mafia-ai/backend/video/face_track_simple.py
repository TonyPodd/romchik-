# backend/video/face_track_simple.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import time
from collections import deque

import numpy as np
import cv2

# ---- утилиты ----
BBox = Tuple[int,int,int,int]
Point = Tuple[float,float]

def center_of(b: BBox) -> Tuple[int,int]:
    x1,y1,x2,y2 = b
    return ((x1+x2)//2, (y1+y2)//2)

def iou(a: BBox, b: BBox) -> float:
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    ra = max(1,(ax2-ax1)*(ay2-ay1))
    rb = max(1,(bx2-bx1)*(by2-by1))
    return inter/(ra+rb-inter+1e-6)

# Подмножество индексов FaceMesh (ключевые точки)
MESH_KEYS = [
    # глаза (углы)
    33, 133, 362, 263,
    # брови (центры)
    105, 334,
    # нос
    1, 2, 98, 327, 94, 331,
    # рот (углы и центр верх/низ)
    61, 291, 13, 14,
    # подбородок
    152
]

@dataclass
class FaceObs:
    track_id: int
    bbox: BBox
    center: Tuple[int,int]
    person_id: Optional[int]
    sim: float
    score: float

class _Track:
    def __init__(self, tid:int):
        self.id = tid
        self.history = deque(maxlen=8)  # (center, bbox)
        self.last_ts = time.time()
        self.embema = None  # EMA эмбеддинга

class FaceTracker:
    """
    Лёгкий face-tracking + регистрация персон на основе FaceMesh-эмбеддинга.
    """
    def __init__(self, max_faces:int=8, det_conf:float=0.6, mesh_refine:bool=False):
        import mediapipe as mp
        self.det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=det_conf)
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=max_faces, refine_landmarks=mesh_refine,
            min_detection_confidence=det_conf, min_tracking_confidence=0.6
        )
        self._tracks: Dict[int,_Track] = {}
        self._next_id = 1

        # простая БД персон: person_id -> embedding(np.float32)
        self.registry: Dict[int, np.ndarray] = {}
        self._next_person_id = 1
        self.sim_threshold = 0.85  # под медиапайповый эмбеддинг (подкрутишь при желании)

    # --- эмбеддинг по mesh ---
    def _embed_from_mesh(self, lms: List[Point]) -> Optional[np.ndarray]:
        if not lms or len(lms) < 468:
            return None
        pts = np.array([lms[i] for i in MESH_KEYS], dtype=np.float32)  # (K,2)
        # нормализация в bbox
        minxy = pts.min(axis=0); maxxy = pts.max(axis=0)
        size = (maxxy - minxy).max() + 1e-6
        norm = (pts - minxy) / size  # (K,2) ∈ [0..1]
        vec = norm.reshape(-1)       # (2K,)
        vec = vec - vec.mean()
        n = np.linalg.norm(vec) + 1e-6
        return (vec / n).astype(np.float32)

    def _assign_tracks(self, boxes: List[BBox]) -> List[int]:
        tids = [-1]*len(boxes)
        used=set()
        for tid,tr in list(self._tracks.items()):
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
                if cost < best_cost: best_cost, best_j = cost, j
            if best_j>=0 and best_cost<0.9:
                tids[best_j]=tid; used.add(best_j); tr.last_ts=time.time()
        for j,b in enumerate(boxes):
            if tids[j]==-1:
                tid=self._next_id; self._next_id+=1
                self._tracks[tid]=_Track(tid)
                tids[j]=tid
        now=time.time()
        dead=[tid for tid,tr in self._tracks.items() if (now-tr.last_ts)>2.0 and (not tr.history)]
        for tid in dead: self._tracks.pop(tid, None)
        return tids

    def _detect_faces(self, frame_bgr) -> Tuple[List[BBox], List[float]]:
        H,W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.det.process(rgb)
        boxes: List[BBox] = []
        scores: List[float] = []
        if res.detections:
            for d in res.detections:
                r = d.location_data.relative_bounding_box
                x1 = int(max(0, r.xmin) * W); y1 = int(max(0, r.ymin) * H)
                x2 = int(min(1.0, r.xmin + r.width) * W); y2 = int(min(1.0, r.ymin + r.height) * H)
                if x2>x1 and y2>y1:
                    boxes.append((x1,y1,x2,y2))
                    scores.append(float(d.score[0] if d.score else 0.0))
        return boxes, scores

    def _mesh_landmarks(self, frame_bgr, boxes: List[BBox]) -> List[Optional[List[Point]]]:
        # вычисляем mesh на полном кадре (быстрее и устойчивее), а не по кропам
        H,W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        out=[None]*len(boxes)
        if not res.multi_face_landmarks: return out
        # mediaPipe не даёт прямого сопоставления с детектором → сматчим по IoU
        # сначала соберём bbox по mesh
        mesh_boxes=[]
        mesh_pts=[]
        for lmset in res.multi_face_landmarks:
            xs=[lm.x for lm in lmset.landmark]; ys=[lm.y for lm in lmset.landmark]
            minx,maxx = max(0.0,min(xs)), min(1.0,max(xs))
            miny,maxy = max(0.0,min(ys)), min(1.0,max(ys))
            x1,y1 = int(minx*W), int(miny*H); x2,y2 = int(maxx*W), int(maxy*H)
            lms=[(lm.x*W, lm.y*H) for lm in lmset.landmark]
            mesh_boxes.append((x1,y1,x2,y2)); mesh_pts.append(lms)
        for i,b in enumerate(boxes):
            # найти mesh с max IoU
            best=-1; bestv=0.0
            for j,mb in enumerate(mesh_boxes):
                v = iou(b, mb)
                if v>bestv: best, bestv = j, v
            if best>=0 and bestv>0.2:
                out[i] = mesh_pts[best]
        return out

    def process(self, frame_bgr) -> List[FaceObs]:
        boxes, scores = self._detect_faces(frame_bgr)
        if not boxes:
            for tr in self._tracks.values(): tr.history.clear()
            return []

        tids = self._assign_tracks(boxes)
        lms_list = self._mesh_landmarks(frame_bgr, boxes)

        obs: List[FaceObs] = []
        for (bbox,score,tid,lms) in zip(boxes, scores, tids, lms_list):
            # эмбеддинг и EMA
            emb = self._embed_from_mesh(lms) if lms is not None else None
            tr = self._tracks[tid]
            tr.history.append((center_of(bbox), bbox))
            if emb is not None:
                if tr.embema is None: tr.embema = emb
                else: tr.embema = 0.3*emb + 0.7*tr.embema

            # сопоставление с БД
            person_id=None; sim=0.0
            if tr.embema is not None and self.registry:
                regs = np.stack(list(self.registry.values()), axis=0)  # N x D
                q = tr.embema / (np.linalg.norm(tr.embema)+1e-6)
                regs_n = regs / (np.linalg.norm(regs,axis=1,keepdims=True)+1e-6)
                sims = regs_n @ q
                j = int(np.argmax(sims))
                sim = float(sims[j])
                pid = list(self.registry.keys())[j]
                if sim >= self.sim_threshold:
                    person_id = pid

            obs.append(FaceObs(
                track_id=tid, bbox=bbox, center=center_of(bbox),
                person_id=person_id, sim=sim, score=score
            ))
        return obs

    # --- публичные операции с БД персон ---
    def enroll_track(self, track_id:int) -> Optional[int]:
        """Сохранить текущий эмбеддинг трека как нового человека; вернёт person_id или None."""
        tr = self._tracks.get(track_id)
        if tr is None or tr.embema is None:
            return None
        pid = self._next_person_id; self._next_person_id += 1
        self.registry[pid] = tr.embema.astype(np.float32)
        return pid

    def clear_registry(self):
        self.registry.clear()
        self._next_person_id = 1
