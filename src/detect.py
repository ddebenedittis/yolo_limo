from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO


colors = [
    '#0072BD',
    '#D95319',
    '#EDB120',
    '#7E2F8E',
    '#77AC30',
    '#4DBEEE',
    '#A2142F',
    '#FF6F00',
    '#8DFF33',
    '#33FFF7',
]

traj_length = 100


model = YOLO("/home/ws/detect/yolov8n-custom/weights/best.pt")
video_path = "/home/ws/vid/C0953.mp4"
cap = cv2.VideoCapture(video_path)
track_history = defaultdict(lambda: [])

cv2.namedWindow("YOLO26 Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO26 Tracking", 960, 540)
while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True)
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
        annotated_frame = results[0].plot(labels=False, conf=False)
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > traj_length:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            c = colors[track_id % len(colors)]
            c = tuple(int(c.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=c, thickness=5)
            
        cv2.imshow("YOLO26 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
