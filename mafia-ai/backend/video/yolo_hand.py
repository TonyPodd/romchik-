# backend/video/yolo_hand.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np, cv2, os

@dataclass
class YoloDet:
    xyxy: Tuple[int,int,int,int]
    conf: float
    cls: int

def _letterbox(img, new_shape=(640,640), color=(114,114,114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def _nms(boxes, scores, iou_thr=0.45):
    if len(boxes) == 0: return []
    boxes = boxes.astype(np.float32)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        w = np.maximum(0, xx2-xx1+1)
        h = np.maximum(0, yy2-yy1+1)
        inter = w*h
        iou = inter / (areas[i] + areas[idxs[1:]] - inter + 1e-8)
        idxs = idxs[1:][iou < iou_thr]
    return keep

class YoloHandDetector:
    def __init__(self, path: Optional[str] = None, conf_thr: float = 0.25, iou_thr: float = 0.45, input_size: int = 640):
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.input_size = input_size
        self.session = None
        self.iname = None; self.oname = None
        self._broken = False
        if os.getenv("DISABLE_YOLO_HAND", "0") == "1":
            self._broken = True
            print("[yolo_hand] disabled by env, using MediaPipe only")

        if path is None:
            path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "hand_yolo.onnx"))
        if os.path.exists(path):
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                self.iname = self.session.get_inputs()[0].name
                self.oname = self.session.get_outputs()[0].name
                print(f"[yolo_hand] loaded: {path}")
            except Exception as e:
                print(f"[yolo_hand] failed to load {path}: {e}")
        else:
            print(f"[yolo_hand] model not found at {path} — fallback to MediaPipe")

    def available(self) -> bool:
        return self.session is not None and not self._broken

    def _prepare(self, img_bgr):
        img, r, (dw, dh) = _letterbox(img_bgr, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))[None,...]
        return img, r, (dw, dh)

    def _xywh2xyxy(self, x):
        y = x.copy()
        y[:,0] = x[:,0] - x[:,2] / 2
        y[:,1] = x[:,1] - x[:,3] / 2
        y[:,2] = x[:,0] + x[:,2] / 2
        y[:,3] = x[:,1] + x[:,3] / 2
        return y

    def detect(self, frame_bgr) -> List[YoloDet]:
        if not self.available(): return []
        H, W = frame_bgr.shape[:2]
        inp, r, (dw, dh) = self._prepare(frame_bgr)
        try:
            out = self.session.run([self.oname], {self.iname: inp})[0]
        except Exception as e:
            print(f"[yolo_hand] inference disabled due to error: {e}")
            self._broken = True
            return []

        if out.ndim == 3 and out.shape[1] == 6:
            pred = out[0]
        elif out.ndim == 3 and out.shape[2] >= 6:
            pred = out[0][:, :6]
        elif out.ndim == 4:
            pred = out[0].reshape(out.shape[1], -1).T
            pred = pred[:, :6]
        else:
            pred = out.reshape(-1, out.shape[-1])
            pred = pred[:, :6]

        xyxy = pred[:, :4].copy()
        if (xyxy.max() <= 1.5):
            xyxy[:, 0:4] *= self.input_size
        w_h = (xyxy[:,2] - xyxy[:,0]).mean()
        if w_h < 1:
            xyxy = self._xywh2xyxy(xyxy)

        xyxy[:, [0,2]] -= dw; xyxy[:, [1,3]] -= dh
        xyxy /= r

        conf = pred[:, 4]
        cls  = pred[:, 5].astype(np.int32) if pred.shape[1] >= 6 else np.zeros(len(pred),dtype=np.int32)

        keep = conf >= self.conf_thr
        xyxy = xyxy[keep]; conf = conf[keep]; cls = cls[keep]
        if len(xyxy) == 0: return []

        xyxy[:,0::2] = np.clip(xyxy[:,0::2], 0, W-1)
        xyxy[:,1::2] = np.clip(xyxy[:,1::2], 0, H-1)

        inds = _nms(xyxy, conf, self.iou_thr)
        outd: List[YoloDet] = []
        for i in inds:
            x1,y1,x2,y2 = xyxy[i].astype(int)
            outd.append(YoloDet((x1,y1,x2,y2), float(conf[i]), int(cls[i]) if pred.shape[1]>=6 else 0))
        return outd
