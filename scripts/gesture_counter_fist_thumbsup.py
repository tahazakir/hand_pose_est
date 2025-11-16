import argparse
import math
from collections import deque, Counter
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------
# Hand landmark indices (Ultralytics / MediaPipe layout)
# ----------------------------
WRIST = 0
THUMB = (1, 4)
INDEX = (5, 8)
MIDDLE = (9, 12)
RING = (13, 16)
PINKY = (17, 20)

FINGERS = [THUMB, INDEX, MIDDLE, RING, PINKY]
FINGERTIPS = [4, 8, 12, 16, 20]

# ----------------------------
# Config
# ----------------------------
@dataclass
class GestureConfig:
    det_conf_thresh: float = 0.45
    thumb_extended_thresh: float = 0.65   # thumb extension ratio for thumbs up
    finger_curled_thresh: float = 0.50    # other fingers should be curled
    fist_thresh: float = 0.45             # all fingers curled threshold
    smooth_window: int = 5
    stable_min: int = 3

# ----------------------------
# Helpers
# ----------------------------
def pairwise_avg_dist(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    dsum, cnt = 0.0, 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dsum += float(np.linalg.norm(points[i] - points[j]))
            cnt += 1
    return dsum / max(cnt, 1)

def majority_vote(labels: List[str]) -> Optional[str]:
    if not labels:
        return None
    c = Counter(labels)
    return c.most_common(1)[0][0]

# ----------------------------
# Fist to Thumbs-Up gesture classifier
# ----------------------------
def classify_fist_thumbsup(kps_xy: np.ndarray, cfg: GestureConfig) -> Optional[str]:
    """
    Returns 'thumbs_up' or 'fist' (or None if ambiguous).

    Thumbs up: thumb extended upward, other 4 fingers curled
    Fist: all 5 fingers curled
    """
    if kps_xy.shape[0] <= 20:
        return None

    # --- compute intrinsic palm scale ---
    palm_width = np.linalg.norm(kps_xy[5] - kps_xy[17])    # index_base to pinky_base
    palm_len = np.linalg.norm(kps_xy[0] - kps_xy[9])       # wrist to middle_base
    scale = max((palm_width + palm_len) / 2.0, 1e-6)

    # --- per-finger extension ratio ---
    finger_dists = []
    for base, tip in FINGERS:
        dist = np.linalg.norm(kps_xy[tip] - kps_xy[base]) / scale
        finger_dists.append(dist)

    thumb_dist, index_dist, middle_dist, ring_dist, pinky_dist = finger_dists

    # --- Check for thumbs up ---
    # Thumb should be extended
    thumb_extended = thumb_dist > cfg.thumb_extended_thresh

    # Other 4 fingers should be curled
    other_fingers_curled = (
        index_dist < cfg.finger_curled_thresh and
        middle_dist < cfg.finger_curled_thresh and
        ring_dist < cfg.finger_curled_thresh and
        pinky_dist < cfg.finger_curled_thresh
    )

    if thumb_extended and other_fingers_curled:
        return "thumbs_up"

    # --- Check for fist (all fingers curled) ---
    all_curled = all(d < cfg.fist_thresh for d in finger_dists)

    if all_curled:
        return "fist"

    return None

# ----------------------------
# Rep counter for fist → thumbs_up → fist
# ----------------------------
class RepCounter:
    def __init__(self):
        self.state = "need_start_closed"
        self.count = 0

    def update(self, gesture: Optional[str]):
        if gesture is None:
            return
        if self.state == "need_start_closed" and gesture == "fist":
            self.state = "seeking_thumbsup"
        elif self.state == "seeking_thumbsup" and gesture == "thumbs_up":
            self.state = "seeking_return"
        elif self.state == "seeking_return" and gesture == "fist":
            self.count += 1
            self.state = "seeking_thumbsup"

# ----------------------------
# Visualization
# ----------------------------
def draw_overlay(frame, kps_xy, gesture, reps):
    # draw all 21 keypoints
    for idx, (x, y) in enumerate(kps_xy):
        xi, yi = int(x), int(y)
        # Highlight thumb (indices 1-4) in different color
        color = (0, 255, 255) if 1 <= idx <= 4 else (255, 0, 255)
        cv2.circle(frame, (xi, yi), 3, color, -1)
        cv2.putText(frame, str(idx), (xi + 4, yi - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    label = f"{gesture or '...'} | reps: {reps}"
    cv2.rectangle(frame, (8, 8), (280, 36), (0, 0, 0), -1)
    cv2.putText(frame, label, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

# ----------------------------
# Main loop
# ----------------------------
def run(weights: str, source: str, device: str = "", cfg: GestureConfig = GestureConfig()):
    model = YOLO(weights)

    recent_gestures: deque = deque(maxlen=cfg.smooth_window)
    stable_gesture: Optional[str] = None
    stable_run = 0
    counter = RepCounter()

    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            verbose=False,
            device=device if device else None,
            conf=cfg.det_conf_thresh
        )

        gesture_now = None
        kps = None

        for r in results:
            if len(r.boxes) == 0 or r.keypoints is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            box_confs = r.boxes.conf.cpu().numpy()
            kps_xy = r.keypoints.xy.cpu().numpy()

            i = int(np.argmax(box_confs))
            this_kps = kps_xy[i]
            gesture_now = classify_fist_thumbsup(this_kps, cfg)
            kps = this_kps

        # --- temporal smoothing ---
        recent_gestures.append(gesture_now or "None")
        voted = majority_vote([g for g in recent_gestures if g != "None"])
        if voted is not None:
            if stable_gesture == voted:
                stable_run += 1
            else:
                stable_gesture = voted
                stable_run = 1

        accepted_gesture = stable_gesture if stable_run >= cfg.stable_min else None
        counter.update(accepted_gesture)

        # --- draw ---
        if kps is not None:
            draw_overlay(frame, kps, accepted_gesture, counter.count)

        cv2.putText(frame, f"fist→thumbsup reps: {counter.count}", (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 30), 2, cv2.LINE_AA)

        cv2.imshow("Hand Rehab - Fist to Thumbs Up", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="models/best.pt")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--thumb_extended_thresh", type=float, default=None)
    parser.add_argument("--finger_curled_thresh", type=float, default=None)
    parser.add_argument("--fist_thresh", type=float, default=None)
    args = parser.parse_args()

    cfg = GestureConfig()
    if args.thumb_extended_thresh is not None:
        cfg.thumb_extended_thresh = args.thumb_extended_thresh
    if args.finger_curled_thresh is not None:
        cfg.finger_curled_thresh = args.finger_curled_thresh
    if args.fist_thresh is not None:
        cfg.fist_thresh = args.fist_thresh

    run(args.weights, args.source, args.device, cfg)
