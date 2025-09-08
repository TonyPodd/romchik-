# backend/video/gestures_demo.py
import cv2
from gestures import GestureDetector

def main():
    det = GestureDetector(table_y_ratio=0.80)
    cap = cv2.VideoCapture(0)  # при необходимости поменяйте индекс камеры

    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру (index=0)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = det.process_frame(frame)
        h, w = frame.shape[:2]

        # отрисовка «линии стола» для наглядности
        table_y = int(0.80 * h)
        cv2.line(frame, (0, table_y), (w, table_y), (128, 128, 128), 1)

        # рамки рук и подписи
        for hand in res.hands:
            x, y, ww, hh = hand.bbox
            cv2.rectangle(frame, (x, y), (x+ww, y+hh), (0, 255, 0), 2)
            txt = f"{hand.handedness} count={hand.count} ext={''.join(k[0] for k,v in hand.extended.items() if v)}"
            cv2.putText(frame, txt, (x, max(15, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        # сводка по кадру
        summary = f"digit={res.digit} | fist_on_table={int(res.fist_on_table)} | pistol={int(res.pistol)}"
        cv2.putText(frame, summary, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Gestures Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
