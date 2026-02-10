# backend/video/gestures.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    raise RuntimeError("Install mediapipe first: pip install mediapipe") from e


@dataclass
class HandInfo:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    handedness: str                  # "Left" / "Right"
    extended: Dict[str, bool]        # thumb/index/middle/ring/pinky
    count: int                       # number of extended fingers
    center: Tuple[int, int]          # cx, cy (pixels)
    track_id: int                    # lightweight hand track id
    gesture: str                     # canonical label: 1..5, thumb_up, thumb_down, ok, jambo, shot, ...


@dataclass
class GestureResult:
    digit: Optional[int]
    fist_on_table: bool
    pistol: bool
    hands: List[HandInfo]


@dataclass
class _HandTrackState:
    track_id: int
    handedness: str
    center: Tuple[int, int]
    last_seen: float
    two_up_ts: float = 0.0
    shot_until_ts: float = 0.0


class GestureDetector:
    def __init__(
        self,
        table_y_ratio: float = 0.80,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 10,
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
        self._tracks: Dict[int, _HandTrackState] = {}
        self._next_track_id = 1
        self._track_ttl_sec = 1.2
        self._shot_window_sec = 0.9
        self._shot_hold_sec = 0.45

    @staticmethod
    def _norm_to_px(lm, w: int, h: int) -> Tuple[int, int]:
        return int(lm.x * w), int(lm.y * h)

    @staticmethod
    def _bbox_from_landmarks(landmarks_px: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in landmarks_px]
        ys = [p[1] for p in landmarks_px]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return x1, y1, x2 - x1, y2 - y1

    @staticmethod
    def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(np.hypot(float(a[0] - b[0]), float(a[1] - b[1])))

    @staticmethod
    def _angle(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> float:
        ba = np.array([float(a[0] - b[0]), float(a[1] - b[1])], dtype=np.float32)
        bc = np.array([float(c[0] - b[0]), float(c[1] - b[1])], dtype=np.float32)
        na = float(np.linalg.norm(ba))
        nb = float(np.linalg.norm(bc))
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        cosang = float(np.clip(float(np.dot(ba, bc) / (na * nb)), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosang)))

    def _finger_extended(self, lm_px: List[Tuple[int, int]], finger: str) -> bool:
        # MediaPipe landmark indices:
        # 0:wrist; thumb:1,2,3,4; index:5,6,7,8; middle:9,10,11,12; ring:13..16; pinky:17..20
        def up(i_tip: int, i_pip: int) -> bool:
            return lm_px[i_tip][1] < (lm_px[i_pip][1] - 4)

        if finger == "index":
            return up(8, 6)
        if finger == "middle":
            return up(12, 10)
        if finger == "ring":
            return up(16, 14)
        if finger == "pinky":
            return up(20, 18)
        if finger == "thumb":
            # More conservative thumb check to avoid +1 finger bias.
            wrist = lm_px[0]
            thumb_mcp = lm_px[2]
            thumb_ip = lm_px[3]
            thumb_tip = lm_px[4]
            index_mcp = lm_px[5]
            middle_mcp = lm_px[9]

            palm = max(12.0, self._dist(wrist, middle_mcp))
            d_tip_wrist = self._dist(wrist, thumb_tip)
            d_ip_wrist = self._dist(wrist, thumb_ip)
            d_tip_index = self._dist(thumb_tip, index_mcp)
            ang = self._angle(thumb_mcp, thumb_ip, thumb_tip)

            straight = ang >= 140.0
            radial = d_tip_wrist > (d_ip_wrist + 0.10 * palm)
            away_from_index = d_tip_index > (0.34 * palm)
            return bool(straight and radial and away_from_index)
        return False

    @staticmethod
    def _pistol(ext: Dict[str, bool]) -> bool:
        return bool(ext["thumb"] and ext["index"] and not (ext["middle"] or ext["ring"] or ext["pinky"]))

    @staticmethod
    def _closed_fist(ext: Dict[str, bool]) -> bool:
        return not (ext["thumb"] or ext["index"] or ext["middle"] or ext["ring"] or ext["pinky"])

    def _assign_track(
        self,
        center: Tuple[int, int],
        handedness: str,
        bbox: Tuple[int, int, int, int],
        now: float,
        used_track_ids: set[int],
    ) -> _HandTrackState:
        bw = max(1.0, float(bbox[2]))
        bh = max(1.0, float(bbox[3]))
        gate = max(90.0, 1.4 * max(bw, bh))
        gate2 = gate * gate

        best_id: Optional[int] = None
        best_d2 = float("inf")

        for tid, track in self._tracks.items():
            if tid in used_track_ids:
                continue
            if track.handedness != handedness:
                continue
            dx = float(center[0] - track.center[0])
            dy = float(center[1] - track.center[1])
            d2 = dx * dx + dy * dy
            if d2 < best_d2 and d2 <= gate2:
                best_d2 = d2
                best_id = tid

        if best_id is None:
            best_id = self._next_track_id
            self._next_track_id += 1
            track = _HandTrackState(
                track_id=best_id,
                handedness=handedness,
                center=center,
                last_seen=now,
            )
            self._tracks[best_id] = track
            return track

        track = self._tracks[best_id]
        track.center = center
        track.last_seen = now
        return track

    def _cleanup_tracks(self, now: float) -> None:
        dead = [tid for tid, tr in self._tracks.items() if (now - tr.last_seen) > self._track_ttl_sec]
        for tid in dead:
            self._tracks.pop(tid, None)

    def _non_thumb_folded_count(self, lm_px: List[Tuple[int, int]]) -> int:
        wrist = lm_px[0]
        palm = max(12.0, self._dist(wrist, lm_px[9]))
        folded = 0
        # index, middle, ring, pinky
        for mcp_i, pip_i, tip_i in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
            mcp = lm_px[mcp_i]
            pip = lm_px[pip_i]
            tip = lm_px[tip_i]
            ang = self._angle(mcp, pip, tip)
            d_tip_wrist = self._dist(tip, wrist)
            d_mcp_wrist = self._dist(mcp, wrist)
            folded_angle = ang < 158.0
            folded_dist = d_tip_wrist < (d_mcp_wrist + 0.22 * palm)
            if folded_angle and folded_dist:
                folded += 1
        return folded

    def _is_thumb_up(self, ext: Dict[str, bool], lm_px: List[Tuple[int, int]]) -> bool:
        palm = max(12.0, self._dist(lm_px[0], lm_px[9]))
        thumb_mcp = lm_px[2]
        thumb_ip = lm_px[3]
        tip = lm_px[4]
        wrist = lm_px[0]
        vx = float(tip[0] - thumb_ip[0])
        vy = float(tip[1] - thumb_ip[1])
        thumb_len = self._dist(thumb_ip, tip)
        thumb_ang = self._angle(thumb_mcp, thumb_ip, tip)
        vertical_axis = abs(vy) > (abs(vx) * 1.15)
        folded_non_thumb = self._non_thumb_folded_count(lm_px)

        if not vertical_axis:
            return False
        if thumb_len < (0.20 * palm):
            return False
        if thumb_ang < 135.0:
            return False
        if tip[1] >= (wrist[1] - 0.20 * palm):
            return False
        # Allow small landmark noise: at least 3 of 4 non-thumb fingers folded.
        return folded_non_thumb >= 3

    def _is_thumb_down(self, ext: Dict[str, bool], lm_px: List[Tuple[int, int]]) -> bool:
        palm = max(12.0, self._dist(lm_px[0], lm_px[9]))
        thumb_mcp = lm_px[2]
        thumb_ip = lm_px[3]
        tip = lm_px[4]
        wrist = lm_px[0]
        vx = float(tip[0] - thumb_ip[0])
        vy = float(tip[1] - thumb_ip[1])
        thumb_len = self._dist(thumb_ip, tip)
        thumb_ang = self._angle(thumb_mcp, thumb_ip, tip)
        vertical_axis = abs(vy) > (abs(vx) * 1.15)
        folded_non_thumb = self._non_thumb_folded_count(lm_px)

        if not vertical_axis:
            return False
        if thumb_len < (0.20 * palm):
            return False
        if thumb_ang < 135.0:
            return False
        if tip[1] <= (wrist[1] + 0.20 * palm):
            return False
        if vy <= 0:
            return False
        # More tolerant than strict "all folded", but still avoids confusion with numeric gestures.
        return folded_non_thumb >= 3

    def _is_ok_sign(self, ext: Dict[str, bool], lm_px: List[Tuple[int, int]]) -> bool:
        palm = max(12.0, self._dist(lm_px[0], lm_px[9]))
        thumb_index_tip = self._dist(lm_px[4], lm_px[8])
        close_enough = thumb_index_tip < (0.34 * palm)
        support_fingers = ext["middle"] or ext["ring"] or ext["pinky"]
        return bool(close_enough and support_fingers)

    @staticmethod
    def _is_jambo(ext: Dict[str, bool]) -> bool:
        return bool(ext["thumb"] and ext["pinky"] and not (ext["index"] or ext["middle"] or ext["ring"]))

    @staticmethod
    def _is_index_pointing(ext: Dict[str, bool]) -> bool:
        # Allow thumb to be slightly open: in practice pointing is often shown as "1" with relaxed thumb.
        return bool(ext["index"] and not (ext["middle"] or ext["ring"] or ext["pinky"]))

    def _is_self_point(self, ext: Dict[str, bool], lm_px: List[Tuple[int, int]]) -> bool:
        if not self._is_index_pointing(ext):
            return False
        wrist = lm_px[0]
        idx_pip = lm_px[6]
        idx_tip = lm_px[8]
        palm = max(12.0, self._dist(wrist, lm_px[9]))
        # Finger directed down toward chest and close to vertical body axis.
        points_down = idx_tip[1] > (idx_pip[1] + 0.16 * palm)
        near_axis = abs(float(idx_tip[0] - wrist[0])) < (0.55 * palm)
        return bool(points_down and near_axis)

    @staticmethod
    def _is_open_palm(ext: Dict[str, bool], count: int) -> bool:
        if count < 4:
            return False
        return bool(ext["index"] and ext["middle"] and ext["ring"] and ext["pinky"])

    def _numeric_count(self, ext: Dict[str, bool], lm_px: List[Tuple[int, int]]) -> int:
        # For numeric gestures we primarily count non-thumb fingers to avoid +1 bias.
        # Thumb contributes only for a clear open palm ("5").
        base = int(bool(ext["index"])) + int(bool(ext["middle"])) + int(bool(ext["ring"])) + int(bool(ext["pinky"]))
        if base < 4 or not bool(ext["thumb"]):
            return base

        palm = max(12.0, self._dist(lm_px[0], lm_px[9]))
        thumb_tip = lm_px[4]
        thumb_ip = lm_px[3]
        index_mcp = lm_px[5]
        wrist = lm_px[0]

        thumb_span = self._dist(thumb_tip, index_mcp)
        thumb_len = self._dist(thumb_ip, thumb_tip)
        d_tip_wrist = self._dist(thumb_tip, wrist)
        d_ip_wrist = self._dist(thumb_ip, wrist)

        if (
            thumb_len >= (0.18 * palm)
            and thumb_span >= (0.34 * palm)
            and d_tip_wrist > (d_ip_wrist + 0.08 * palm)
        ):
            return 5
        return 4

    @staticmethod
    def _is_two_up_pose(ext: Dict[str, bool]) -> bool:
        return bool(ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"])

    @staticmethod
    def _is_two_folded(ext: Dict[str, bool]) -> bool:
        return bool((not ext["index"]) and (not ext["middle"]))

    def _classify_hand(
        self,
        ext: Dict[str, bool],
        count: int,
        lm_px: List[Tuple[int, int]],
        track: _HandTrackState,
        now: float,
    ) -> str:
        two_up = self._is_two_up_pose(ext)
        if two_up:
            track.two_up_ts = now

        if self._is_two_folded(ext) and track.two_up_ts > 0.0 and (now - track.two_up_ts) <= self._shot_window_sec:
            track.shot_until_ts = now + self._shot_hold_sec
            track.two_up_ts = 0.0

        if track.shot_until_ts > now:
            return "shot"

        if self._is_ok_sign(ext, lm_px):
            # По правилам этого проекта "OK" = шериф.
            return "sheriff"
        if self._is_self_point(ext, lm_px):
            return "self"
        if self._is_thumb_up(ext, lm_px):
            return "thumb_up"
        if self._is_thumb_down(ext, lm_px):
            return "thumb_down"
        if self._is_jambo(ext):
            return "jambo"

        if 1 <= count <= 5:
            return str(count)
        if count <= 0:
            return "unknown"
        return "unknown"

    def process_frame(self, frame_bgr: np.ndarray) -> GestureResult:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(frame_rgb)

        now = time.time()
        used_track_ids: set[int] = set()
        hands_out: List[HandInfo] = []
        hands_meta: List[Dict[str, Any]] = []

        if res.multi_hand_landmarks and res.multi_handedness:
            for lms, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                lm_px = [self._norm_to_px(lm, w, h) for lm in lms.landmark]
                bbox = self._bbox_from_landmarks(lm_px)
                cx = int(np.mean([p[0] for p in lm_px]))
                cy = int(np.mean([p[1] for p in lm_px]))
                label = handedness.classification[0].label  # "Left" / "Right"

                ext = {
                    "thumb": self._finger_extended(lm_px, "thumb"),
                    "index": self._finger_extended(lm_px, "index"),
                    "middle": self._finger_extended(lm_px, "middle"),
                    "ring": self._finger_extended(lm_px, "ring"),
                    "pinky": self._finger_extended(lm_px, "pinky"),
                }
                count = self._numeric_count(ext, lm_px)

                track = self._assign_track((cx, cy), label, bbox, now, used_track_ids)
                used_track_ids.add(track.track_id)
                gesture = self._classify_hand(ext, count, lm_px, track, now)

                hands_out.append(
                    HandInfo(
                        bbox=bbox,
                        handedness=label,
                        extended=ext,
                        count=count,
                        center=(cx, cy),
                        track_id=track.track_id,
                        gesture=gesture,
                    )
                )
                hands_meta.append(
                    {
                        "index": len(hands_out) - 1,
                        "lm_px": lm_px,
                        "ext": ext,
                        "count": count,
                    }
                )

            # "Дон": открытая ладонь + указательный палец второй руки на безымянный палец ладони.
            for pointer in hands_meta:
                p_ext = pointer.get("ext")
                p_lm = pointer.get("lm_px")
                if not isinstance(p_ext, dict) or not isinstance(p_lm, list):
                    continue
                if not self._is_index_pointing(p_ext):
                    continue
                p_tip = p_lm[8]

                for palm in hands_meta:
                    if palm["index"] == pointer["index"]:
                        continue
                    b_ext = palm.get("ext")
                    b_lm = palm.get("lm_px")
                    b_count = int(palm.get("count", 0))
                    if not isinstance(b_ext, dict) or not isinstance(b_lm, list):
                        continue
                    if not self._is_open_palm(b_ext, b_count):
                        continue

                    base_palm = max(12.0, self._dist(b_lm[0], b_lm[9]))
                    ring_mcp = b_lm[13]
                    ring_pip = b_lm[14]
                    ring_tip = b_lm[16]
                    near_ring = (
                        self._dist(p_tip, ring_mcp) <= (0.78 * base_palm)
                        or self._dist(p_tip, ring_pip) <= (0.74 * base_palm)
                        or self._dist(p_tip, ring_tip) <= (0.68 * base_palm)
                    )
                    if not near_ring:
                        continue

                    pointer_center = hands_out[pointer["index"]].center
                    palm_center = hands_out[palm["index"]].center
                    if self._dist(pointer_center, palm_center) > (2.7 * base_palm):
                        continue

                    hands_out[pointer["index"]].gesture = "don"
                    break

        self._cleanup_tracks(now)

        total_fingers = sum(hh.count for hh in hands_out)
        digit: Optional[int] = None
        pistol = any(self._pistol(hh.extended) for hh in hands_out)

        if len(hands_out) == 0:
            digit = None
        elif len(hands_out) == 1:
            c = max(0, min(5, hands_out[0].count))
            digit = c if c > 0 else 0
        else:
            s = max(0, min(10, total_fingers))
            digit = s if s > 0 else 0

        table_y = int(self.table_y_ratio * h)
        fist_on_table = any(self._closed_fist(hh.extended) and hh.center[1] >= table_y for hh in hands_out)

        return GestureResult(digit=digit, fist_on_table=fist_on_table, pistol=pistol, hands=hands_out)
