import argparse
import json
import math
import time
from collections import deque, Counter
from dataclasses import dataclass
from typing import List, Optional, Dict

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------
# Hand keypoint indices
# ----------------------------
WRIST = 0
FINGERS = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 20)]
FINGERTIPS = [4, 8, 12, 16, 20]

# ----------------------------
# Config
# ----------------------------
@dataclass
class GestureConfig:
    det_conf_thresh: float = 0.45
    open_thresh: float = 0.75
    fist_thresh: float = 0.45
    spread_thresh: float = 0.45
    smooth_window: int = 5
    stable_min: int = 3
    calibration_file: str = "hand_calibration.json"

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
# Compute hand metrics
# ----------------------------
def compute_hand_metrics(kps_xy: np.ndarray) -> Dict[str, float]:
    palm_width = np.linalg.norm(kps_xy[5] - kps_xy[17])
    palm_len = np.linalg.norm(kps_xy[0] - kps_xy[9])
    scale = max((palm_width + palm_len) / 2.0, 1e-6)
    finger_dists = [np.linalg.norm(kps_xy[tip] - kps_xy[base]) / scale for base, tip in FINGERS]
    tips = np.stack([kps_xy[t] for t in FINGERTIPS], axis=0)
    spread = pairwise_avg_dist(tips) / palm_width if palm_width > 0 else 0
    return {"mean_finger": float(np.mean(finger_dists)), "spread": float(spread)}

# ----------------------------
# Classifier
# ----------------------------
def classify_open_close(kps_xy: np.ndarray, cfg: GestureConfig) -> Optional[str]:
    if kps_xy.shape[0] <= 20:
        return None
    metrics = compute_hand_metrics(kps_xy)
    f = metrics["mean_finger"]
    s = metrics["spread"]
    if f > cfg.open_thresh and s > cfg.spread_thresh:
        return "open_palm"
    elif f < cfg.fist_thresh or s < 0.15:
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
    for idx, (x, y) in enumerate(kps_xy):
        xi, yi = int(x), int(y)
        cv2.circle(frame, (xi, yi), 3, (255, 0, 255), -1)
    cv2.putText(frame, f"{gesture or '...'} | reps: {reps}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)

# ----------------------------
# Draw text helper
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

    samples = {"fist": [], "open": []}

    def record_samples(label: str, seconds: int = 3):
        # Grace period
        grace = 3
        start = time.time()
        while time.time() - start < grace:
            ok, frame = cap.read()
            if not ok:
                break
            draw_instruction(frame, f"Get ready for {label.upper()}...", (0, 255, 255),
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
            draw_instruction(frame, f"HOLD {label.upper()}", (0, 0, 255))
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                return False
        return True

    if not record_samples("fist"):
        cap.release(); cv2.destroyAllWindows(); return
    if not record_samples("open"):
        cap.release(); cv2.destroyAllWindows(); return

    cap.release()
    cv2.destroyAllWindows()

    if not samples["fist"] or not samples["open"]:
        print("⚠️ Calibration failed. Not enough samples.")
        return

    mean_fist = {k: np.mean([m[k] for m in samples["fist"]]) for k in samples["fist"][0]}
    mean_open = {k: np.mean([m[k] for m in samples["open"]]) for k in samples["open"][0]}

    cfg.fist_thresh = float((mean_fist["mean_finger"] + mean_open["mean_finger"]) / 2 * 0.9)
    cfg.open_thresh = float((mean_fist["mean_finger"] + mean_open["mean_finger"]) / 2 * 1.1)
    cfg.spread_thresh = float((mean_fist["spread"] + mean_open["spread"]) / 2)

    calib = {
        "fist_thresh": cfg.fist_thresh,
        "open_thresh": cfg.open_thresh,
        "spread_thresh": cfg.spread_thresh
    }
    with open(cfg.calibration_file, "w") as f:
        json.dump(calib, f, indent=2)
    print("\n✅ Calibration complete. Saved to", cfg.calibration_file)
    print(json.dumps(calib, indent=2))

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
            cfg.open_thresh = data["open_thresh"]
            cfg.spread_thresh = data["spread_thresh"]
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
            gesture_now = classify_open_close(kps_xy, cfg)
            kps = kps_xy

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

        if kps is not None:
            draw_overlay(frame, kps, accepted_gesture, counter.count)

        cv2.imshow("Hand Rehab - Open/Close", frame)
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
