#!/usr/bin/env python3
"""
Index Finger Up vs Fist - Improved for Straight-Up Detection
----------------------------------------------------------
This version improves detection when the index finger is straight up (perpendicular to camera),
not just when tilted 30-40 degrees. Uses a combination of vertical component, normalized distance,
and finger angle to detect extension.

Examples:
  python index_counter_openfist_straightup.py --source 1 --model models/best.pt --thresholds index_thresh.json --view
"""
import argparse, json, os, sys, time
from collections import deque
from datetime import datetime
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except Exception as e:
    print("ERROR: Failed to import ultralytics. Please `pip install ultralytics`.", e)
    sys.exit(1)

# MediaPipe hand indices
WRIST = 0
ID_FI = [5,6,7,8]  # Index finger: MCP, PIP, DIP, TIP
MI_FI = [9,10,11,12]
RI_FI = [13,14,15,16]
PI_FI = [17,18,19,20]
OTHER_FINGERS = [MI_FI, RI_FI, PI_FI]

def l2(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def safe_get_point(kpts, idx):
    if idx < 0 or idx >= len(kpts):
        return None
    x, y, v = kpts[idx]
    if v < 0.1:
        return None
    return (x, y)

def palm_scale_and_width(kpts):
    p5 = safe_get_point(kpts, 5)
    p17 = safe_get_point(kpts, 17)
    p0 = safe_get_point(kpts, 0)
    p9 = safe_get_point(kpts, 9)
    parts = []
    if p5 is not None and p17 is not None:
        parts.append(l2(p5, p17))
    if p0 is not None and p9 is not None:
        parts.append(l2(p0, p9))
    if not parts:
        return None, None
    width = parts[0] if p5 is not None and p17 is not None else None
    scale = float(np.median(parts))
    return scale, width

def finger_base_tip_dist(kpts, finger_indices):
    mcp = safe_get_point(kpts, finger_indices[0])
    tip = safe_get_point(kpts, finger_indices[-1])
    if mcp is None or tip is None:
        return None
    return l2(mcp, tip)

def compute_finger_angle(kpts, finger_indices):
    """Calculate the angle of the finger relative to vertical (0 = straight up, 90 = horizontal)."""
    base = safe_get_point(kpts, finger_indices[0])
    tip = safe_get_point(kpts, finger_indices[-1])
    if base is None or tip is None:
        return None
    
    # Vector from base to tip
    dx = tip[0] - base[0]
    dy = tip[1] - base[1]  # Note: Y increases downward in image coords
    
    # Angle from vertical (straight up = 0 degrees)
    # When finger is straight up: dx is small, dy is negative (tip higher than base)
    angle_rad = np.arctan2(abs(dx), -dy) if dy < 0 else np.arctan2(abs(dx), abs(dy))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def compute_metrics(kpts):
    """
    Enhanced metrics for better straight-up detection:
    - Normalized distance (works when finger is extended)
    - Vertical component (works when finger is tilted)
    - Finger angle (detects when pointing more directly at camera)
    """
    scale, width = palm_scale_and_width(kpts)
    if scale is None or scale <= 1e-6:
        return None, None, None, None, False
    
    idx_len = finger_base_tip_dist(kpts, ID_FI)
    other_lens = []
    for f in OTHER_FINGERS:
        d = finger_base_tip_dist(kpts, f)
        if d is not None:
            other_lens.append(d)
    
    if idx_len is None or not other_lens:
        return None, None, None, None, False
    
    index_len_norm = idx_len / scale
    others_len_norm = float(np.mean([d/scale for d in other_lens]))
    
    # Vertical component
    idx_base = safe_get_point(kpts, ID_FI[0])
    idx_tip = safe_get_point(kpts, ID_FI[-1])
    if idx_base is not None and idx_tip is not None:
        idx_vertical = (idx_base[1] - idx_tip[1]) / scale
    else:
        idx_vertical = None
    
    # Finger angle (0 = straight up, 90 = horizontal)
    idx_angle = compute_finger_angle(kpts, ID_FI)
    
    return index_len_norm, others_len_norm, idx_vertical, idx_angle, True

def draw_text(img, text, org, scale=0.7, thick=2, color=(255,255,255), bg=(0,0,0)):
    (w,h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x,y = org
    cv2.rectangle(img, (x, y-h-6), (x+w+6, y+6), bg, -1)
    cv2.putText(img, text, (x+3, y-3), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

class Smoother:
    def __init__(self, window=15):  # Balanced window for stability and responsiveness
        self.buf = deque(maxlen=window)
        self.last_stable = None
        self.stable_count = 0
        self.stable_threshold = 6  # Require 6 consistent frames before changing (more responsive)
    def push(self, val):
        self.buf.append(val)
        # Track stability - only change if we see consistent new value
        if val == self.last_stable:
            self.stable_count += 1
        else:
            # Check if new value is dominant in buffer
            filtered = [v for v in self.buf if v is not None]
            if filtered:
                vals, counts = np.unique(filtered, return_counts=True)
                most_common = vals[np.argmax(counts)]
                most_common_count = counts[np.argmax(counts)]
                # Only change if new value is clearly dominant (at least 65% of buffer)
                if most_common != self.last_stable and most_common_count >= len(filtered) * 0.65:
                    self.stable_count = 1
                    self.last_stable = most_common
                else:
                    self.stable_count = 0
    def mode(self):
        if not self.buf:
            return self.last_stable
        filtered = [v for v in self.buf if v is not None]
        if not filtered:
            return self.last_stable  # Return last stable value if buffer is empty
        vals, counts = np.unique(filtered, return_counts=True)
        most_common = vals[np.argmax(counts)]
        most_common_count = counts[np.argmax(counts)]
        # Return new value if it's the same as last stable OR if it's been stable enough AND dominant
        if most_common == self.last_stable:
            return most_common
        elif self.stable_count >= self.stable_threshold and most_common_count >= len(filtered) * 0.65:
            return most_common
        else:
            return self.last_stable  # Return last stable value to prevent drift

class RepCounter:
    """Count a full rep when user performs: INDEX_UP -> FIST -> INDEX_UP"""
    def __init__(self):
        self.state = "INIT"
        self.count = 0
    def update(self, gesture):
        if gesture is None:
            return self.count
        if self.state == "INIT":
            if gesture == "INDEX_UP":
                self.state = "UP_HELD"
            elif gesture == "FIST":
                # Allow starting in the down position so reps aren't stuck at launch
                self.state = "DOWN_HELD"
        elif self.state == "UP_HELD":
            if gesture == "FIST":
                self.state = "DOWN_HELD"
        elif self.state == "DOWN_HELD":
            if gesture == "INDEX_UP":
                self.count += 1
                self.state = "UP_HELD"
        return self.count

def decide(index_len_norm, others_len_norm, index_vertical, index_angle, thr):
    """
    Enhanced decision logic for straight-up detection:
    - PRIMARY: Use vertical component when available (works for both straight up and tilted)
    - SECONDARY: Use normalized distance for straight-up cases (angle < 45°)
    - FIST: When vertical is small/negative AND distance is small
    """
    if index_len_norm is None or others_len_norm is None:
        return None
    
    # Get thresholds
    vertical_up_min = float(thr.get("vertical_up_min", 0.15))
    index_up_min = float(thr.get("index_up_min", 0.55))  # Normalized distance threshold
    others_down_max = float(thr.get("others_down_max", 0.55))
    fist_all_max = float(thr.get("fist_all_max", 0.45))
    h = float(thr.get("hysteresis", 0.05))
    
    # Determine if finger is pointing more directly at camera (straight up)
    is_straight_up = index_angle is not None and index_angle < 45
    
    # For straight-up fingers (pointing at camera), vertical component is inverted
    # Use normalized distance as primary indicator
    if is_straight_up:
        # When straight up: normalized distance is the reliable indicator
        # Lower threshold slightly for straight-up case since palm scale might be different
        if index_len_norm >= (index_up_min - h) * 0.9:  # 10% lower threshold for straight-up
            return "FIST"  # FIXED: swapped
        elif index_len_norm <= fist_all_max + h:
            return "INDEX_UP"  # FIXED: swapped
    else:
        # For tilted fingers: use vertical component as primary
        if index_vertical is not None:
            if index_vertical >= vertical_up_min - h:
                return "FIST"  # FIXED: swapped
            # If vertical is negative and distance is small, it's a fist
            elif index_vertical < 0 and index_len_norm <= fist_all_max + h:
                return "INDEX_UP"  # FIXED: swapped
        
        # Fallback: check normalized distance for tilted cases
        if index_len_norm >= index_up_min - h:
            return "FIST"  # FIXED: swapped
        elif index_len_norm <= fist_all_max + h:
            return "INDEX_UP"  # FIXED: swapped
    
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=0, help="camera index or path to video/image")
    ap.add_argument("--model", default="yolov8n-pose.pt", help="YOLO pose model path (v8/11)")
    ap.add_argument("--device", default=None, help="device id, e.g., 0 for CUDA; leave blank for auto")
    ap.add_argument("--view", action="store_true", help="show visualization")
    ap.add_argument("--thresholds", default="index_thresh.json", help="thresholds json from autocalib")
    ap.add_argument("--save_counts", action="store_true", help="write counts to counts.txt on exit")
    args = ap.parse_args()

    # Load thresholds
    if os.path.exists(args.thresholds):
        with open(args.thresholds, "r") as f:
            thr = json.load(f)
    else:
        thr = {
            "vertical_up_min": 0.15,
            "index_up_min": 0.55,
            "others_down_max": 0.55,
            "fist_all_max": 0.45,
            "hysteresis": 0.05
        }
        print("WARNING: thresholds file not found, using defaults:", thr)

    model = YOLO(args.model)
    cap = cv2.VideoCapture(int(args.source)) if str(args.source).isdigit() else cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("ERROR: Cannot open source:", args.source)
        sys.exit(1)

    smoother = Smoother(window=15)  # Balanced window for stability and responsiveness
    counter = RepCounter()
    last_gesture = None
    
    # Setup logging
    log_file = f"live_counter_straightup_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_file, "w") as f:
        f.write("timestamp,idx_norm,oth_norm,idx_vertical,idx_angle,raw_gesture\n")
    print(f"Logging to: {log_file}")
    print("Press 'q' or ESC to exit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False, device=args.device, conf=0.15, imgsz=640)  # Optimized threshold
        kpts = None
        for r in results:
            if len(r.keypoints) > 0:
                pts = r.keypoints.xyn[0].cpu().numpy() if hasattr(r.keypoints, "xyn") else r.keypoints.xy[0].cpu().numpy()
                if pts.shape[0] >= 21:
                    confs = r.keypoints.conf[0].cpu().numpy() if hasattr(r.keypoints, "conf") else np.ones((pts.shape[0],))
                    kpts = [(float(pts[i,0]), float(pts[i,1]), float(confs[i])) for i in range(min(21, pts.shape[0]))]
                    break

        gesture = None
        idx_norm, oth_norm, idx_vertical, idx_angle = None, None, None, None
        if kpts is not None:
            idx_norm, oth_norm, idx_vertical, idx_angle, valid = compute_metrics(kpts)
            if valid:
                gesture = decide(idx_norm, oth_norm, idx_vertical, idx_angle, thr)
                # Log metrics
                with open(log_file, "a") as f:
                    v_str = f"{idx_vertical:.3f}" if idx_vertical is not None else "None"
                    a_str = f"{idx_angle:.1f}" if idx_angle is not None else "None"
                    f.write(f"{time.time():.3f}, idx_norm={idx_norm:.3f}, oth_norm={oth_norm:.3f}, "
                           f"idx_vertical={v_str}, idx_angle={a_str}, gesture={gesture}\n")

        smoother.push(gesture)
        smooth_g = smoother.mode()
        reps = counter.update(smooth_g)
        
        # Log smoothed gesture
        if smooth_g != last_gesture:
            with open(log_file, "a") as f:
                f.write(f"{time.time():.3f}, SMOOTHED: {smooth_g}, REPS: {reps}\n")
            last_gesture = smooth_g

        vis = frame.copy()
        h, w = vis.shape[:2]
        
        # Large, prominent display at top center
        gesture_text = f"Gesture: {smooth_g if smooth_g else 'None'}"
        reps_text = f"Reps: {reps}"
        
        # Calculate text sizes for centering
        (gw, gh), _ = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        (rw, rh), _ = cv2.getTextSize(reps_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
        
        # Draw gesture status (top center, large)
        gx = (w - gw) // 2
        gy = 60
        draw_text(vis, gesture_text, (gx, gy), scale=1.2, thick=3, 
                 color=(0, 255, 0) if smooth_g == "INDEX_UP" else (255, 255, 255), 
                 bg=(0, 0, 0))
        
        # Draw rep count (below gesture, even larger)
        rx = (w - rw) // 2
        ry = gy + gh + 50
        draw_text(vis, reps_text, (rx, ry), scale=1.5, thick=4, 
                 color=(0, 255, 255), bg=(0, 0, 0))
        
        # Debug info in corner (smaller)
        if idx_angle is not None:
            draw_text(vis, f"Angle: {idx_angle:.1f}°", (20, h - 40), scale=0.5, bg=(32,32,32))

        if args.view:
            cv2.imshow("Index Up vs Fist (Straight-Up)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q'):
                break

    cap.release()
    if args.view:
        cv2.destroyAllWindows()

    if args.save_counts:
        with open("counts.txt", "w") as f:
            f.write(str(counter.count))
        print("Saved count to counts.txt")

if __name__ == "__main__":
    main()

