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
    # Hardcoded calibrated values from thumbsup_calibration.json
    # Update these values by running gesture_counter_fist_thumbsup_autocalib.py --calibrate
    thumb_extended_thresh: float = 1.434020124812319   # calibrated thumb extension ratio
    finger_curled_thresh: float = 0.45592519897553657   # calibrated other fingers curled threshold
    fist_thresh: float = 0.7517664067612753             # calibrated fist threshold (for average)
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

    Thumbs up: thumb extended upward, other 4 fingers curled (using averages)
    Fist: average of all fingers below threshold (matching calibration logic)
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
    
    # --- Compute averages (matching calibration script logic) ---
    other_fingers_avg = np.mean([index_dist, middle_dist, ring_dist, pinky_dist])
    all_fingers_avg = np.mean(finger_dists)

    # --- Check for thumbs up ---
    # Thumb should be extended, other 4 fingers should be curled (using average)
    thumb_extended = thumb_dist > cfg.thumb_extended_thresh
    other_fingers_curled = other_fingers_avg < cfg.finger_curled_thresh

    if thumb_extended and other_fingers_curled:
        return "thumbs_up"

    # --- Check for fist (average of all fingers below threshold) ---
    # This matches the calibration script logic: all_avg < cfg.fist_thresh
    # IMPORTANT: Don't classify as fist if thumb is extended (prevents misclassifying thumbs up as fist)
    if all_fingers_avg < cfg.fist_thresh and not thumb_extended:
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

    # Using hardcoded calibrated values from GestureConfig defaults
    print(f"📏 Using calibrated thresholds:")
    print(f"   fist_thresh: {cfg.fist_thresh:.3f}")
    print(f"   thumb_extended_thresh: {cfg.thumb_extended_thresh:.3f}")
    print(f"   finger_curled_thresh: {cfg.finger_curled_thresh:.3f}")

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

        # --- temporal smoothing with hysteresis ---
        recent_gestures.append(gesture_now or "None")
        non_none_gestures = [g for g in recent_gestures if g != "None"]
        voted = majority_vote(non_none_gestures) if non_none_gestures else None
        
        if voted is not None:
            if stable_gesture == voted:
                # Same gesture: increment stability counter
                stable_run += 1
            else:
                # Different gesture detected: require stronger evidence to switch
                # Count how many times the new gesture appears in recent window
                new_gesture_count = sum(1 for g in recent_gestures if g == voted)
                total_non_none = len(non_none_gestures)
                # Require at least 70% of recent non-None frames to switch (more conservative)
                if total_non_none > 0 and new_gesture_count / total_non_none >= 0.7:
                    stable_gesture = voted
                    stable_run = 1
                # Otherwise, keep the current stable gesture (don't switch on weak evidence)
                # Don't decay stability - just ignore the inconsistent vote

        accepted_gesture = stable_gesture if stable_run >= cfg.stable_min else None
        counter.update(accepted_gesture)

        # --- draw ---
        if kps is not None:
            draw_overlay(frame, kps, accepted_gesture, counter.count)

        cv2.putText(frame, f"fist to thumbsup reps: {counter.count}", (12, 60),
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
    parser.add_argument("--thumb_extended_thresh", type=float, default=None,
                        help="Override thumb extended threshold (default: calibrated value)")
    parser.add_argument("--finger_curled_thresh", type=float, default=None,
                        help="Override finger curled threshold (default: calibrated value)")
    parser.add_argument("--fist_thresh", type=float, default=None,
                        help="Override fist threshold (default: calibrated value)")
    args = parser.parse_args()

    cfg = GestureConfig()
    
    # Command-line args override hardcoded calibrated values
    if args.thumb_extended_thresh is not None:
        cfg.thumb_extended_thresh = args.thumb_extended_thresh
    if args.finger_curled_thresh is not None:
        cfg.finger_curled_thresh = args.finger_curled_thresh
    if args.fist_thresh is not None:
        cfg.fist_thresh = args.fist_thresh

    run(args.weights, args.source, args.device, cfg)
