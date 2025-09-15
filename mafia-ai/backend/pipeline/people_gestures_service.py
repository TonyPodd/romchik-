# backend/pipeline/people_gestures_service.py
from __future__ import annotations
import time
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from video.face_recog_adv import (
    MPFaceDetector, align_by_5pts, make_embedder, FaceBankV2, face_quality_score
)
from video.gestures_engine import GestureEngine, draw_hands
from storage.gestures_history import GestureEvent, append_event, summary_window

BBox = Tuple[int,int,int,int]

def center_of(b: BBox) -> Tuple[int,int]:
    x1,y1,x2,y2 = b
    return ((x1+x2)//2, (y1+y2)//2)

def iou(a: BBox, b: BBox) -> float:
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1)
    inter=iw*ih; ra=max(1,(ax2-ax1)*(ay2-ay1)); rb=max(1,(bx2-bx1)*(by2-by1))
    return inter/(ra+rb-inter+1e-6)

class _FaceTrack:
    def __init__(self, tid:int):
        self.id = tid
        self.last_ts = time.time()
        self.history: List[Tuple[Tuple[int,int], BBox]] = []

class PeopleGesturesService:
    """
    Единый сервис «лица+жесты» с историей событий.
    - Лица: MP+5pts -> align -> embed (mbf/r100) -> FaceBankV2
    - Жесты: GestureEngine (MediaPipe/YOLO/HaGRID)
    - Сшивка «рука→лицо» по ближайшему центру (или без лица, если далеко)
    - Логгер: в storage/gestures/gestures.jsonl (дебаунс + min_conf)
    - Энролл: start_enroll(name) / stop_enroll(save_min)
    """
    def __init__(
        self,
        # gestures
        hand_detector: str = "mp",
        yolo_path: Optional[str] = None,
        use_mp_landmarks: bool = True,
        use_hagrid: bool = False,
        hagrid_model_path: Optional[str] = None,
        max_hands: int = 20,
        det_conf: float = 0.6,
        track_conf: float = 0.6,
        # faces
        embedder_kind: str = "r100",      # "mbf"|"r100"
        embedder_model_path: Optional[str] = None,
        match_thr_mean: float = 0.55,
        match_thr_gallery: float = 0.60,
        # events
        min_conf: float = 0.65,
        fire_every_ms: int = 400,
        # linking
        max_link_dist2: int = 300*300     # макс. кв-раст. рука→лицо для связи
    ):
        # gestures
        self.gest = GestureEngine(
            detector=hand_detector, yolo_model_path=yolo_path,
            use_mp_landmarks=use_mp_landmarks, use_hagrid=use_hagrid, hagrid_model_path=hagrid_model_path,
            max_hands=max_hands, det_conf=det_conf, track_conf=track_conf
        )
        # faces
        self.face_det = MPFaceDetector(det_conf=0.6, mesh_track_conf=0.6)
        self.embedder = make_embedder(embedder_kind, embedder_model_path)
        self.bank = FaceBankV2()
        self.match_thr_mean = match_thr_mean
        self.match_thr_gallery = match_thr_gallery

        # tracks for faces
        self._tracks: Dict[int,_FaceTrack] = {}
        self._next_tid = 1

        # events
        self.min_conf = min_conf
        self.fire_every_ms = fire_every_ms
        self.last_fire: Dict[int, Tuple[str,float]] = {}  # hand_track_id -> (label_name, ts)

        # enroll
        self._enrolling = False
        self._enroll_name = ""
        self._samples: List[np.ndarray] = []
        self._enroll_thumb: Optional[np.ndarray] = None
        self._last_capture = 0.0

        self.max_link_dist2 = max_link_dist2

    # ------------- enroll control -------------
    def start_enroll(self, name: str):
        self._enrolling = True
        self._enroll_name = (name or f"player_{int(time.time())}").strip()
        self._samples.clear()
        self._enroll_thumb = None
        self._last_capture = 0.0

    def stop_enroll(self, save_min_samples: int = 8) -> Optional[Dict[str,Any]]:
        self._enrolling = False
        if len(self._samples) >= max(6, save_min_samples):
            rec = self.bank.add_or_update(self._enroll_name, self._samples,
                                          self._enroll_thumb if self._enroll_thumb is not None else np.zeros((112,112,3),np.uint8))
            self._samples.clear()
            return rec
        self._samples.clear()
        return None

    # ------------- internals -------------
    def _assign_face_tracks(self, boxes: List[BBox]) -> List[int]:
        tids = [-1]*len(boxes)
        used = set()
        for tid,tr in list(self._tracks.items()):
            lb = tr.history[-1][1] if tr.history else None
            if lb is None: continue
            lc = center_of(lb)
            bestj, bestcost = -1, 1e9
            for j,b in enumerate(boxes):
                if j in used: continue
                c = center_of(b)
                d = (c[0]-lc[0])**2 + (c[1]-lc[1])**2
                o = 1.0 - iou(lb, b)
                cost = 0.7*o + 0.3*(d/40000.0)
                if cost < bestcost:
                    bestcost, bestj = cost, j
            if bestj>=0 and bestcost<0.9:
                tids[bestj]=tid; used.add(bestj); tr.last_ts=time.time()
        for j,b in enumerate(boxes):
            if tids[j]==-1:
                tid=self._next_tid; self._next_tid+=1
                self._tracks[tid]=_FaceTrack(tid); tids[j]=tid
        # gc
        now=time.time()
        for tid in [t for t,tr in self._tracks.items() if (now-tr.last_ts)>3.0 and (not tr.history)]:
            self._tracks.pop(tid, None)
        return tids

    def _try_capture_enroll(self, aligned: np.ndarray) -> bool:
        if not self._enrolling: return False
        now=time.time()
        if now - self._last_capture < 0.25:  # 4 Гц
            return False
        q = face_quality_score(aligned)
        if q < 0.35:
            return False
        import numpy as np
        e = self.embedder.embed(aligned)
        if len(self._samples)==0:
            self._samples.append(e); self._enroll_thumb = aligned.copy(); self._last_capture=now; return True
        S = np.array(self._samples, dtype=np.float32)
        S = S/(np.linalg.norm(S,axis=1,keepdims=True)+1e-6)
        en = e/(np.linalg.norm(e)+1e-6)
        sim = float(np.max(S @ en))
        if sim < 0.90:
            self._samples.append(e); self._last_capture=now; return True
        return False

    # ------------- public process -------------
    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str,Any]:
        H, W = frame_bgr.shape[:2]

        # faces
        dets = self.face_det.detect_and_landmarks(frame_bgr)
        boxes = [d["bbox"] for d in dets]
        tids = self._assign_face_tracks(boxes)

        faces_out = []
        for d,tid in zip(dets, tids):
            x1,y1,x2,y2 = d["bbox"]; pts5 = d["pts5"]
            if pts5 is not None:
                aligned = align_by_5pts(frame_bgr, pts5, (112,112))
            else:
                crop = frame_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                aligned = cv2.resize(crop, (112,112)) if crop.size>0 else np.zeros((112,112,3),np.uint8)

            # enroll sampling
            self._try_capture_enroll(aligned)

            emb = self.embedder.embed(aligned)
            pid,name,score = self.bank.match(
                emb, threshold_mean=self.match_thr_mean, threshold_gallery=self.match_thr_gallery
            )
            tr = self._tracks[tid]
            tr.history.append((center_of(d["bbox"]), d["bbox"]))
            faces_out.append({
                "track_id": tid, "bbox": d["bbox"], "person_id": pid, "person_name": name, "sim": float(score),
                "aligned": aligned  # опционально, удобно для отладки
            })

        # gestures
        hands = self.gest.process(frame_bgr)

        # link hands→faces
        mapping: Dict[int, List[Dict[str,Any]]] = {f["track_id"]: [] for f in faces_out}
        for h in hands:
            hx, hy = h.center
            best: Optional[Dict[str, Any]] = None
            bestd: float = 1e12
            for f in faces_out:
                fx, fy = center_of(f["bbox"])
                d2 = (hx - fx) ** 2 + (hy - fy) ** 2
                if d2 < bestd:
                    bestd = d2          # <-- правильный порядок присваивания
                    best = f
            if best is not None and bestd <= float(self.max_link_dist2):
                # гарантируем, что ключ есть
                if best["track_id"] not in mapping:
                    mapping[best["track_id"]] = []
                mapping[best["track_id"]].append({
                    "hand_track_id": h.track_id,
                    "label": h.smooth_label.name,
                    "confidence": float(h.confidence),
                    "bbox": h.bbox
                })


        # events (debounce)
        now=time.time()
        for f in faces_out:
            lst = mapping.get(f["track_id"], [])
            if not lst: continue
            lst.sort(key=lambda x: x["confidence"], reverse=True)
            top = lst[0]
            if top["confidence"] >= self.min_conf:
                last = self.last_fire.get(top["hand_track_id"], (None,0.0))
                if (now-last[1])*1000.0 >= self.fire_every_ms and top["label"] != (last[0] or ""):
                    self.last_fire[top["hand_track_id"]] = (top["label"], now)
                    ev = GestureEvent(
                        ts=now,
                        person_id=f["person_id"],
                        face_track_id=f["track_id"],
                        hand_track_id=top["hand_track_id"],
                        label=top["label"],
                        confidence=top["confidence"],
                        frame_w=W, frame_h=H,
                        hand_bbox=tuple(map(int, top["bbox"])),
                        face_bbox=tuple(map(int, f["bbox"]))
                    )
                    append_event(ev)

        return {
            "faces": faces_out,
            "hands": hands,
            "links": mapping
        }

    # helpers
    def history_summary(self, window_sec: float = 10.0, person_id: int | None = None):
        return summary_window(window_sec, person_id=person_id)
