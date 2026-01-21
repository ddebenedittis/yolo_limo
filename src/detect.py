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

traj_length = 200
plot_boxes = False

# ---- FLAG: fading or classic trajectories ----
fading_trajectory = True  # True = fading trails, False = non-fading full trajectories

# Fade/overlay parameters (used only if fading_trajectory=True)
decay = 0.995          # closer to 1.0 = longer trails, lower = faster fade
trail_alpha = 2.0     # how strong the canvas is when overlaid

model = YOLO("/home/ws/detect/yolov8n-custom/weights/best.pt")
video_path = "/home/ws/vid/C0940.mp4"
cap = cv2.VideoCapture(video_path)

track_history = defaultdict(lambda: [])

traj_canvas = None  # float32 canvas initialized on first frame (only for fading mode)

cv2.namedWindow("YOLO26 Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO26 Tracking", 960, 540)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Initialize canvas once we know frame size (only in fading mode)
    if fading_trajectory and traj_canvas is None:
        traj_canvas = np.zeros_like(frame, dtype=np.float32)

    results = model.track(frame, persist=True)

    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
    active_ids = set(track_ids)

    annotated_frame = results[0].plot(boxes=plot_boxes, conf=False, labels=False)

    if fading_trajectory:
        # 1) decay old trails
        traj_canvas *= decay

    # ---- Update track history; in fading mode, draw only current segments ----
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))
        if len(track) > traj_length:
            track.pop(0)

        if fading_trajectory and len(track) >= 2:
            p1 = tuple(np.int32(track[-2]))
            p2 = tuple(np.int32(track[-1]))

            c = colors[track_id % len(colors)]
            c = tuple(int(c.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))  # RGB
            c_bgr = (c[2], c[1], c[0])  # BGR for OpenCV

            cv2.line(traj_canvas, p1, p2, c_bgr, thickness=5, lineType=cv2.LINE_AA)

    if fading_trajectory:
        # 3) overlay the trail canvas onto the current frame
        over = cv2.addWeighted(
            annotated_frame.astype(np.float32), 1.0,
            traj_canvas, trail_alpha,
            0.0
        )
        annotated_frame = np.clip(over, 0, 255).astype(np.uint8)

    else:
        # -------- NON-FADING MODE (classic trajectories) --------
        # Draw trajectories ONLY for currently active IDs
        for track_id in active_ids:
            track = track_history[track_id]
            if len(track) >= 2:
                pts = np.int32(track).reshape(-1, 1, 2)

                c = colors[track_id % len(colors)]
                c = tuple(int(c.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))  # RGB
                c_bgr = (c[2], c[1], c[0])

                cv2.polylines(
                    annotated_frame,
                    [pts],
                    isClosed=False,
                    color=c_bgr,
                    thickness=5,
                    lineType=cv2.LINE_AA,
                )

    # -------- BIG CURRENT POSITION DOT (ephemeral) --------
    # Draw only for currently visible IDs
    radius = 10
    for track_id in active_ids:
        track = track_history[track_id]
        if not track:
            continue

        x, y = track[-1]
        center = tuple(np.int32((x, y)))

        c = colors[track_id % len(colors)]
        c = tuple(int(c.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))  # RGB
        c_bgr = (c[2], c[1], c[0])

        cv2.circle(
            annotated_frame,
            center,
            radius,
            c_bgr,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    cv2.imshow("YOLO26 Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
