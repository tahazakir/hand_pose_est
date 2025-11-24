import argparse
import json
import math
import os
import time
from collections import deque, Counter
from dataclasses import dataclass
from typing import List, Optional, Dict

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

# Path to calibration file relative to this script
_CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "thumbsup_calibration.json")

# ----------------------------
# Config
# ----------------------------
@dataclass
class GestureConfig:
    det_conf_thresh: float = 0.45
    thumb_extended_thresh: float = 0.65
    finger_curled_thresh: float = 0.50
    fist_thresh: float = 0.45
    smooth_window: int = 5
    stable_min: int = 3
    calibration_file: str = _CALIBRATION_FILE

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
# Compute hand metrics for thumbs up
# ----------------------------
def compute_hand_metrics(kps_xy: np.ndarray) -> Dict[str, float]:
    """Compute metrics specific to fist/thumbs-up classification."""
    palm_width = np.linalg.norm(kps_xy[5] - kps_xy[17])
    palm_len = np.linalg.norm(kps_xy[0] - kps_xy[9])
    scale = max((palm_width + palm_len) / 2.0, 1e-6)

    # Individual finger distances
    finger_dists = []
    for base, tip in FINGERS:
        dist = np.linalg.norm(kps_xy[tip] - kps_xy[base]) / scale
        finger_dists.append(dist)

    thumb_dist, index_dist, middle_dist, ring_dist, pinky_dist = finger_dists

    # Average of non-thumb fingers
    other_fingers_avg = np.mean([index_dist, middle_dist, ring_dist, pinky_dist])

    return {
        "thumb": float(thumb_dist),
        "other_fingers_avg": float(other_fingers_avg),
        "all_fingers_avg": float(np.mean(finger_dists))
    }

# ----------------------------
# Classifier
# ----------------------------
def classify_fist_thumbsup(kps_xy: np.ndarray, cfg: GestureConfig) -> Optional[str]:
    """
    Returns 'thumbs_up' or 'fist' (or None if ambiguous).
    """
    if kps_xy.shape[0] <= 20:
        return None

    metrics = compute_hand_metrics(kps_xy)
    thumb = metrics["thumb"]
    other = metrics["other_fingers_avg"]
    all_avg = metrics["all_fingers_avg"]

    # Thumbs up: thumb extended, others curled
    if thumb > cfg.thumb_extended_thresh and other < cfg.finger_curled_thresh:
        return "thumbs_up"

    # Fist: all fingers curled
    if all_avg < cfg.fist_thresh:
        return "fist"

    return None

# ----------------------------
# Rep counter
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
    for idx, (x, y) in enumerate(kps_xy):
        xi, yi = int(x), int(y)
        # Highlight thumb in different color
        color = (0, 255, 255) if 1 <= idx <= 4 else (255, 0, 255)
        cv2.circle(frame, (xi, yi), 3, color, -1)

    cv2.putText(frame, f"{gesture or '...'} | reps: {reps}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)

# ----------------------------
# Draw instruction helper
# ----------------------------
def draw_instruction(frame, text, color=(0, 255, 0), sub=None, countdown=None):
    h, w = frame.shape[:2]
    y = h // 2
    cv2.putText(frame, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4, cv2.LINE_AA)
    if sub:
        cv2.putText(frame, sub, (50, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    if countdown is not None:
        cv2.putText(frame, f"{int(countdown)}", (w - 100, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5, cv2.LINE_AA)

# ----------------------------
# Calibration mode with on-screen guidance
# ----------------------------
def calibrate_thresholds(model: YOLO, cfg: GestureConfig, source: str, device: str = ""):
    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    samples = {"fist": [], "thumbs_up": []}

    def record_samples(label: str, seconds: int = 3):
        # Grace period
        grace = 3
        start = time.time()
        while time.time() - start < grace:
            ok, frame = cap.read()
            if not ok:
                break
            instruction = "Make a FIST" if label == "fist" else "Show THUMBS UP"
            draw_instruction(frame, f"Get ready: {instruction}", (0, 255, 255),
                             sub="Hold still when countdown ends",
                             countdown=grace - (time.time() - start))
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                return False

        # Recording
        print(f"\nRecording {label.upper()} for {seconds} seconds...")
        start = time.time()
        while time.time() - start < seconds:
            ok, frame = cap.read()
            if not ok:
                break
            results = model.predict(source=frame, verbose=False,
                                    device=device if device else None,
                                    conf=cfg.det_conf_thresh)
            for r in results:
                if len(r.boxes) == 0 or r.keypoints is None:
                    continue
                kps_xy = r.keypoints.xy.cpu().numpy()[0]
                metrics = compute_hand_metrics(kps_xy)
                samples[label].append(metrics)
                draw_overlay(frame, kps_xy, label, 0)

            progress = (time.time() - start) / seconds
            cv2.rectangle(frame, (40, 40), (int(40 + progress * 400), 60), (0, 255, 0), -1)
            instruction = "HOLD FIST" if label == "fist" else "HOLD THUMBS UP"
            draw_instruction(frame, instruction, (0, 0, 255))
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                return False
        return True

    # Record fist samples
    if not record_samples("fist"):
        cap.release()
        cv2.destroyAllWindows()
        return

    # Record thumbs up samples
    if not record_samples("thumbs_up"):
        cap.release()
        cv2.destroyAllWindows()
        return

    cap.release()
    cv2.destroyAllWindows()

    if not samples["fist"] or not samples["thumbs_up"]:
        print("⚠️ Calibration failed. Not enough samples.")
        return

    # Compute means
    mean_fist = {k: np.mean([m[k] for m in samples["fist"]]) for k in samples["fist"][0]}
    mean_thumbsup = {k: np.mean([m[k] for m in samples["thumbs_up"]]) for k in samples["thumbs_up"][0]}

    # Set thresholds based on calibration data
    # For fist: use average of all fingers
    cfg.fist_thresh = float(mean_fist["all_fingers_avg"] * 1.15)  # slightly above fist average

    # For thumbs up thumb: between fist thumb and thumbs-up thumb
    cfg.thumb_extended_thresh = float((mean_fist["thumb"] + mean_thumbsup["thumb"]) / 2)

    # For other fingers when curled: between fist and thumbs-up
    cfg.finger_curled_thresh = float((mean_fist["other_fingers_avg"] + mean_thumbsup["other_fingers_avg"]) / 2)

    calib = {
        "fist_thresh": cfg.fist_thresh,
        "thumb_extended_thresh": cfg.thumb_extended_thresh,
        "finger_curled_thresh": cfg.finger_curled_thresh,
        "_debug": {
            "mean_fist": mean_fist,
            "mean_thumbsup": mean_thumbsup
        }
    }

    with open(cfg.calibration_file, "w") as f:
        json.dump(calib, f, indent=2)

    print("\n✅ Calibration complete. Saved to", cfg.calibration_file)
    print(json.dumps({k: v for k, v in calib.items() if k != "_debug"}, indent=2))

# ----------------------------
# Main run loop
# ----------------------------
def run(weights: str, source: str, device: str, cfg: GestureConfig):
    model = YOLO(weights)
    recent_gestures: deque = deque(maxlen=cfg.smooth_window)
    stable_gesture: Optional[str] = None
    stable_run = 0
    counter = RepCounter()

    try:
        with open(cfg.calibration_file, "r") as f:
            data = json.load(f)
            cfg.fist_thresh = data["fist_thresh"]
            cfg.thumb_extended_thresh = data["thumb_extended_thresh"]
            cfg.finger_curled_thresh = data["finger_curled_thresh"]
            print(f"📏 Loaded calibration from {cfg.calibration_file}")
    except FileNotFoundError:
        print("ℹ️ No calibration file found. Using defaults.")

    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(source=frame, verbose=False,
                                device=device if device else None,
                                conf=cfg.det_conf_thresh)

        gesture_now = None
        kps = None

        for r in results:
            if len(r.boxes) == 0 or r.keypoints is None:
                continue
            kps_xy = r.keypoints.xy.cpu().numpy()[0]
            gesture_now = classify_fist_thumbsup(kps_xy, cfg)
            kps = kps_xy

        # Temporal smoothing
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

        # Draw
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
    parser.add_argument("--calibrate", action="store_true", help="Run calibration mode")
    args = parser.parse_args()

    cfg = GestureConfig()
    model = YOLO(args.weights)

    if args.calibrate:
        calibrate_thresholds(model, cfg, args.source, args.device)
    else:
        run(args.weights, args.source, args.device, cfg)
