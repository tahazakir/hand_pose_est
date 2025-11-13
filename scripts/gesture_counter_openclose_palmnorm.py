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
FINGERS = [
    (1, 4),    # thumb base → tip
    (5, 8),    # index
    (9, 12),   # middle
    (13, 16),  # ring
    (17, 20),  # pinky
]
FINGERTIPS = [4, 8, 12, 16, 20]

# ----------------------------
# Config
# ----------------------------
@dataclass
class GestureConfig:
    det_conf_thresh: float = 0.45
    open_thresh: float = 0.75     # base→tip / palm scale for “extended”
    fist_thresh: float = 0.45     # base→tip / palm scale for “curled”
    spread_thresh: float = 0.45   # fingertip spread / palm width for open palm
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
# Palm-based gesture classifier
# ----------------------------
def classify_open_close(kps_xy: np.ndarray, cfg: GestureConfig) -> Optional[str]:
    """
    Returns 'open_palm' or 'fist' (or None if ambiguous).
    Uses intrinsic palm geometry for normalization instead of YOLO bbox.
    """
    if kps_xy.shape[0] <= 20:
        return None

    # --- compute intrinsic palm scale ---
    palm_width = np.linalg.norm(kps_xy[5] - kps_xy[17])    # index_base to pinky_base
    palm_len = np.linalg.norm(kps_xy[0] - kps_xy[9])       # wrist to middle_base
    scale = max((palm_width + palm_len) / 2.0, 1e-6)

    # --- per-finger extension ratio ---
    finger_dists = [np.linalg.norm(kps_xy[tip] - kps_xy[base]) / scale for base, tip in FINGERS]

    # --- fingertip spread ---
    tips = np.stack([kps_xy[t] for t in FINGERTIPS], axis=0)
    spread = pairwise_avg_dist(tips) / palm_width if palm_width > 0 else 0

    extended = sum(d > cfg.open_thresh for d in finger_dists)
    curled = sum(d < cfg.fist_thresh for d in finger_dists)

    if extended == 5 and spread > cfg.spread_thresh:
        return "open_palm"
    elif curled >= 4 or spread < 0.15:
        return "fist"
    else:
        return None

# ----------------------------
# Rep counter (same logic)
# ----------------------------
class RepCounter:
    def __init__(self):
        self.state = "need_start_closed"
        self.count = 0

    def update(self, gesture: Optional[str]):
        if gesture is None:
            return
        if self.state == "need_start_closed" and gesture == "fist":
            self.state = "seeking_open"
        elif self.state == "seeking_open" and gesture == "open_palm":
            self.state = "seeking_return"
        elif self.state == "seeking_return" and gesture == "fist":
            self.count += 1
            self.state = "seeking_open"

# ----------------------------
# Visualization
# ----------------------------
def draw_overlay(frame, kps_xy, gesture, reps):
    # draw all 21 keypoints
    for idx, (x, y) in enumerate(kps_xy):
        xi, yi = int(x), int(y)
        cv2.circle(frame, (xi, yi), 3, (255, 0, 255), -1)
        cv2.putText(frame, str(idx), (xi + 4, yi - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    label = f"{gesture or '...'} | reps: {reps}"
    cv2.rectangle(frame, (8, 8), (260, 36), (0, 0, 0), -1)
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
            gesture_now = classify_open_close(this_kps, cfg)
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

        cv2.putText(frame, f"open_close reps: {counter.count}", (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 30), 2, cv2.LINE_AA)

        cv2.imshow("Hand Rehab - Open/Close (Palm Norm)", frame)
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
    parser.add_argument("--open_thresh", type=float, default=None)
    parser.add_argument("--fist_thresh", type=float, default=None)
    parser.add_argument("--spread_thresh", type=float, default=None)
    args = parser.parse_args()

    cfg = GestureConfig()
    if args.open_thresh is not None:
        cfg.open_thresh = args.open_thresh
    if args.fist_thresh is not None:
        cfg.fist_thresh = args.fist_thresh
    if args.spread_thresh is not None:
        cfg.spread_thresh = args.spread_thresh

    run(args.weights, args.source, args.device, cfg)
