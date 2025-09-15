# backend/tests/people_gestures_run.py
import cv2, time, argparse
from pipeline.people_gestures_service import PeopleGesturesService

def draw_overlay(frame, result):
    vis = frame.copy()
    # руки (встроенный draw_hands уже внутри сервиса недоступен — выводим компактно)
    for h in result["hands"]:
        x1,y1,x2,y2 = h.bbox
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,180),2)
        cv2.putText(vis, f'{h.smooth_label.name} {h.confidence:.2f}',
                    (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,(255,255,255),2,cv2.LINE_AA)
    # лица
    for f in result["faces"]:
        x1,y1,x2,y2 = f["bbox"]
        color = (102,182,255) if f["person_id"] else (140,140,140)
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
        label = f'F#{f["track_id"]}'
        if f["person_id"]:
            label += f' P#{f["person_id"]} {f["person_name"]} {f["sim"]:.2f}'
        cv2.putText(vis, label, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62,(255,255,255),2,cv2.LINE_AA)
        # подпись лучшего жеста над лицом
        lst = result["links"].get(f["track_id"], [])
        if lst:
            lst.sort(key=lambda x: x["confidence"], reverse=True)
            g = lst[0]
            cv2.putText(vis, f'{g["label"]} ({g["confidence"]:.2f})',
                        (x1, y2+22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2,cv2.LINE_AA)
    return vis

def main(args):
    # камера
    cap=None
    for api in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
        try:
            cap=cv2.VideoCapture(args.cam, api)
            if cap.isOpened():
                print(f"[camera] opened index={args.cam} api={api}"); break
        except: pass
    if cap is None or not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    svc = PeopleGesturesService(
        hand_detector=args.detector,
        yolo_path=args.yolo,
        use_mp_landmarks=not args.no_mp_landmarks,
        use_hagrid=not args.no_hagrid,
        hagrid_model_path=args.hagrid_model,
        embedder_kind=args.embedder,
        embedder_model_path=args.model,
        min_conf=args.min_conf,
        fire_every_ms=args.fire_ms
    )

    print("[keys] E — start/stop enroll, C — clear bank, R — reload bank.json, H — summary(10s), Q — quit")
    enrolling=False; fps=0.0; last=time.time()

    while True:
        ok, frame = cap.read()
        if not ok: time.sleep(0.01); continue

        res = svc.process_frame(frame)

        vis = draw_overlay(frame, res)

        now=time.time(); dt=now-last; last=now
        fps = 0.9*fps + 0.1*(1.0/dt if dt>0 else fps)
        cv2.putText(vis, f"faces:{len(res['faces'])} hands:{len(res['hands'])} FPS:{fps:.1f}",
                    (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2,cv2.LINE_AA)
        if enrolling:
            cv2.putText(vis, f"ENROLLING… {args.enroll_name}",
                        (12,52), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2,cv2.LINE_AA)

        cv2.imshow("People+Gestures Service (Q quit)", vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'),ord('Q')): break
        elif k in (ord('h'),ord('H')):
            print("— summary(10s) —")
            for k,v in sorted(svc.history_summary(10.0).items(), key=lambda kv:(-kv[1], kv[0])):
                print(f"  {k}: {v}")
        elif k in (ord('c'),ord('C')):
            svc.bank.items.clear(); svc.bank.save()
            print("[bank] cleared")
        elif k in (ord('r'),ord('R')):
            svc.bank._load(); print("[bank] reloaded")
        elif k in (ord('e'),ord('E')):
            if not enrolling:
                svc.start_enroll(args.enroll_name or f"player_{int(time.time())}")
                enrolling=True
            else:
                rec = svc.stop_enroll(save_min_samples=args.save_min)
                if rec:
                    print(f'[bank] saved id={rec["id"]} name={rec["name"]}')
                else:
                    print("[bank] not enough diverse samples")
                enrolling=False

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--detector", choices=["mp","yolo"], default="mp")
    ap.add_argument("--yolo", type=str, default=None)
    ap.add_argument("--no-mp-landmarks", action="store_true")
    ap.add_argument("--no-hagrid", action="store_true")
    ap.add_argument("--hagrid-model", type=str, default=None)
    ap.add_argument("--embedder", choices=["mbf","r100"], default="r100")
    ap.add_argument("--model", type=str, default=None, help="path to r100 onnx (auto-discovery если не задано)")
    ap.add_argument("--min-conf", type=float, default=0.65)
    ap.add_argument("--fire-ms", type=int, default=400)
    ap.add_argument("--enroll-name", type=str, default="player")
    ap.add_argument("--save-min", type=int, default=8)
    args = ap.parse_args()
    main(args)

# 10 игроков / 20 рук, YOLO-детектор рук
# python -m tests.people_gestures_run --detector yolo --yolo "C:\Users\a_nto\Code\romchik\mafia-ai\backend\models\hand_yolov8n.pt" --no-hagrid

# python -m tests.people_gestures_run --detector mp --no-hagrid