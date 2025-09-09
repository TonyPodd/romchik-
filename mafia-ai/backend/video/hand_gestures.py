# backend/video/hand_gestures.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np

TIPS = dict(thumb=4, index=8, middle=12, ring=16, pinky=20)
PIPS = dict(thumb=3, index=6, middle=10, ring=14, pinky=18)
MCPS = dict(thumb=2, index=5, middle=9,  ring=13, pinky=17)

@dataclass
class FingerState:
    extended: Dict[str, bool]
    count: int

def _is_extended(lm, tip, pip, mcp) -> bool:
    wrist = np.array([lm[0].x, lm[0].y])
    tipv  = np.array([lm[tip].x, lm[tip].y])
    pipv  = np.array([lm[pip].x, lm[pip].y])
    mcpv  = np.array([lm[mcp].x, lm[mcp].y])
    v1 = tipv - mcpv; v2 = pipv - mcpv
    cos = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
    straight = cos > 0.6
    dist_ok = np.linalg.norm(tipv-wrist) > np.linalg.norm(pipv-wrist)
    return bool(straight and dist_ok)

def finger_state(lm) -> FingerState:
    ext = {k: _is_extended(lm, TIPS[k], PIPS[k], MCPS[k]) for k in TIPS.keys()}
    return FingerState(extended=ext, count=sum(ext.values()))

def classify_gesture(lm) -> str:
    fs = finger_state(lm)
    e = fs.extended; c = fs.count
    if c == 0: return "fist"
    if c == 5: return "open"
    if c == 1 and e["index"]: return "one"
    if c == 2 and e["index"] and e["middle"] and not (e["thumb"] or e["ring"] or e["pinky"]): return "two"
    if e["thumb"] and e["index"] and not(e["middle"] or e["ring"] or e["pinky"]): return "pistol"
    if e["index"] and e["pinky"] and not(e["middle"] or e["ring"]): return "rock"
    if e["thumb"] and e["pinky"] and not(e["index"] or e["middle"] or e["ring"]): return "call"
    th = np.array([lm[TIPS["thumb"]].x,  lm[TIPS["thumb"]].y])
    ix = np.array([lm[TIPS["index"]].x,  lm[TIPS["index"]].y])
    if np.linalg.norm(th - ix) < 0.03: return "ok"
    return {1:"one",2:"two",3:"three",4:"four",5:"open"}.get(c,"unknown")
