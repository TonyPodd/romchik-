# backend/tests/gestures_test.py
import time, argparse
import cv2
import numpy as np
from video.gestures_engine import GestureEngine, draw_hands

def main(
    cam_idx:int=0,
    detector:str="mp",
    yolo_path:str|None=None,
    use_mp_landmarks:bool=True,
    use_hagrid:bool=False,
    hagrid_model_path:str|None=None
):
    cap=None
    for api in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
        try:
            cap=cv2.VideoCapture(cam_idx, api)
            if cap.isOpened():
                print(f"[camera] opened index={cam_idx} api={api}"); break
        except: pass
    if cap is None or not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    eng = GestureEngine(
        detector=detector,
        yolo_model_path=yolo_path,
        use_mp_landmarks=use_mp_landmarks,
        use_hagrid=use_hagrid,
        hagrid_model_path=hagrid_model_path,
        max_hands=8, det_conf=0.6, track_conf=0.6
    )

    last=time.time(); fps=0.0
    while True:
        ok, frame = cap.read()
        if not ok: time.sleep(0.01); continue

        hands = eng.process(frame)
        vis = frame.copy()
        draw_hands(vis, hands)

        now=time.time(); dt=now-last; last=now
        fps = 0.9*fps + 0.1*(1.0/dt if dt>0 else fps)
        mode = f"YOLO+{'MP' if use_mp_landmarks else 'noMP'}" if detector=="yolo" else "MediaPipe"
        hdr = f"hands:{len(hands)} FPS:{fps:.1f}  [{mode}]  {'HaGRID ON' if use_hagrid else 'rules'}"
        cv2.putText(vis, hdr, (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Gestures Test (Q - quit)", vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), ord('Q')):
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--detector", choices=["mp","yolo"], default="mp")
    ap.add_argument("--yolo", type=str, default=None, help="path to hand_yolov8n.pt")
    ap.add_argument("--no-mp-landmarks", action="store_true", help="disable MP landmarks on YOLO crops")
    ap.add_argument("--hagrid", action="store_true", help="enable HaGRID CNN classifier")
    ap.add_argument("--hagrid-model", type=str, default=None, help="path to HaGRID .onnx (optional)")
    args = ap.parse_args()
    main(
        cam_idx=args.cam,
        detector=args.detector,
        yolo_path=args.yolo,
        use_mp_landmarks=not args.no_mp_landmarks,
        use_hagrid=args.hagrid,
        hagrid_model_path=args.hagrid_model
    )

# python -m tests.gestures_test --detector yolo --yolo "C:\Users\a_nto\Code\romchik\mafia-ai\backend\models\hand_yolov8n.pt"