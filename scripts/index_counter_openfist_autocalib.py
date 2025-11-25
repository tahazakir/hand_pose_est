#!/usr/bin/env python3
"""
Index Finger Up vs Fist - Auto Calibration + Counter
----------------------------------------------------
Run this first to personalize thresholds, then use the palm-norm runner.

Examples:
  python index_counter_openfist_autocalib.py --source 0 --model yolov8n-pose.pt --device 0 --view
  python index_counter_openfist_autocalib.py --source path/to/video.mp4 --model yolo11n-pose.pt

Dependencies:
  - ultralytics (YOLO pose)
  - opencv-python
  - numpy
  - (optional) torch with CUDA for speed

The script will:
  1) Detect hand landmarks via YOLO pose (21-keypoint MediaPipe layout).
  2) Compute palm-intrinsic normalized features.
  3) Guide you through two calibration holds: INDEX_UP and FIST.
  4) Save thresholds JSON (default: index_thresh.json) to reuse in the palm-norm runner.
"""
import argparse, time, json, os, sys, math, statistics
from collections import deque

import numpy as np
import cv2

try:
    from ultralytics import YOLO
except Exception as e:
    print("ERROR: Failed to import ultralytics. Please `pip install ultralytics`.", e)
    sys.exit(1)

# MediaPipe hand indices
WRIST = 0
TH_TH = [1,2,3,4]            # thumb
ID_FI = [5,6,7,8]            # index
MI_FI = [9,10,11,12]         # middle
RI_FI = [13,14,15,16]        # ring
PI_FI = [17,18,19,20]        # pinky

OTHER_FINGERS = [MI_FI, RI_FI, PI_FI]  # (optionally include thumb if desired)

def l2(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def safe_get_point(kpts, idx):
    if idx < 0 or idx >= len(kpts):
        return None
    x, y, v = kpts[idx]
    if v < 0.1:  # low visibility -> unreliable
        return None
    return (x, y)

def palm_scale_and_width(kpts):
    """
    Compute a robust palm scale using MCPs and wrist.
    We'll use MCP index (5) to MCP pinky (17) as 'palm width',
    and wrist (0) to middle MCP (9) as an auxiliary scale.
    """
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
    """Distance from MCP to TIP for a finger."""
    mcp = safe_get_point(kpts, finger_indices[0])
    tip = safe_get_point(kpts, finger_indices[-1])
    if mcp is None or tip is None:
        return None
    return l2(mcp, tip)

def compute_metrics(kpts):
    """
    Returns:
      index_len_norm: normalized MCP->TIP distance for index finger
      others_len_norm: average normalized MCP->TIP distance for middle/ring/pinky
      index_vertical: normalized vertical (Y-axis) component (base_y - tip_y) / scale
      valid: bool whether metrics are valid
    """
    scale, width = palm_scale_and_width(kpts)
    if scale is None or scale <= 1e-6:
        return None, None, None, False

    idx_len = finger_base_tip_dist(kpts, ID_FI)
    other_lens = []
    for f in OTHER_FINGERS:
        d = finger_base_tip_dist(kpts, f)
        if d is not None:
            other_lens.append(d)

    if idx_len is None or not other_lens:
        return None, None, None, False

    index_len_norm = idx_len / scale
    others_len_norm = float(np.mean([d/scale for d in other_lens]))
    
    # OPTION 1: Add vertical (Y-axis) component
    idx_base = safe_get_point(kpts, ID_FI[0])  # Index MCP (base)
    idx_tip = safe_get_point(kpts, ID_FI[-1])  # Index tip
    if idx_base is not None and idx_tip is not None:
        # Y difference: base_y - tip_y (positive when tip is higher/up)
        idx_vertical = (idx_base[1] - idx_tip[1]) / scale
    else:
        idx_vertical = None
    
    return index_len_norm, others_len_norm, idx_vertical, True

def draw_text(img, text, org, scale=0.7, thick=2, color=(255,255,255), bg=(0,0,0)):
    (w,h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x,y = org
    cv2.rectangle(img, (x, y-h-6), (x+w+6, y+6), bg, -1)
    cv2.putText(img, text, (x+3, y-3), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def run_calibration(model, cap, device, view, seconds_each=4, min_stable=30):
    """
    Guide user to hold INDEX_UP and then FIST.
    Collect metrics over time and compute robust thresholds.
    """
    print("Calibration started. Please follow on-screen instructions.")
    phases = [("Hold INDEX UP (other fingers down)", "INDEX_UP"),
              ("Make a FIST (all fingers curled)", "FIST")]
    collected = {"INDEX_UP": {"idx": [], "oth": [], "vertical": []},
                 "FIST": {"idx": [], "oth": [], "vertical": []}}

    def collect_phase(phase_name, label):
        stable = 0
        start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, verbose=False, device=device, conf=0.25, imgsz=640)
            kpts = None
            # take the first hand with kpts if present
            for r in results:
                if len(r.keypoints) > 0:
                    # r.keypoints.xy and r.keypoints.conf are typical; pack as (x,y,v)
                    pts = r.keypoints.xyn[0].cpu().numpy() if hasattr(r.keypoints, "xyn") else r.keypoints.xy[0].cpu().numpy()
                    # normalize to image size for stability; if already normalized, it's fine
                    if pts.shape[0] >= 21:
                        # Make (x,y,v)
                        confs = r.keypoints.conf[0].cpu().numpy() if hasattr(r.keypoints, "conf") else np.ones((pts.shape[0],))
                        kpts = [(float(pts[i,0]), float(pts[i,1]), float(confs[i])) for i in range(min(21, pts.shape[0]))]
                        break

            if kpts is not None:
                idx_norm, oth_norm, idx_vertical, valid = compute_metrics(kpts)
                if valid and idx_vertical is not None:
                    collected[label]["idx"].append(idx_norm)
                    collected[label]["oth"].append(oth_norm)
                    collected[label]["vertical"].append(idx_vertical)
                    stable += 1
                else:
                    stable = 0
            else:
                stable = 0

            # draw prompts
            vis = frame.copy()
            draw_text(vis, f"Calibration: {phase_name}", (20, 40), scale=0.8, bg=(32,32,32))
            draw_text(vis, f"Hold steady... frames ok: {stable}/{min_stable}", (20, 80), scale=0.7, bg=(32,32,32))
            if view:
                cv2.imshow("Calibration", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q') or key == ord('Q'):
                    return False  # ESC/q to abort

            # require both time and min stable frames
            if time.time() - start > seconds_each and stable >= min_stable:
                break
        return True

    for msg, lbl in phases:
        ok = collect_phase(msg, lbl)
        if not ok:
            return None

    # Compute robust thresholds
    def robust(vs):
        if not vs:
            return None
        vs = sorted(vs)
        # trim 10% tails
        n = len(vs)
        lo = int(0.1*n)
        hi = int(0.9*n)
        vs = vs[lo:max(hi, lo+1)]
        return (float(np.median(vs)), float(np.mean(vs)))

    idx_up_med, idx_up_mean = robust(collected["INDEX_UP"]["idx"])
    oth_up_med, oth_up_mean = robust(collected["INDEX_UP"]["oth"])
    idx_fist_med, idx_fist_mean = robust(collected["FIST"]["idx"])
    oth_fist_med, oth_fist_mean = robust(collected["FIST"]["oth"])
    
    # OPTION 1: Compute vertical component thresholds
    vert_up_med, vert_up_mean = robust(collected["INDEX_UP"]["vertical"])
    vert_fist_med, vert_fist_mean = robust(collected["FIST"]["vertical"])

    if None in [idx_up_med, oth_up_med, idx_fist_med, oth_fist_med, vert_up_med, vert_fist_med]:
        return None

    # Thresholds with margins (hysteresis-ready)
    # OPTION 1: Use vertical component as primary indicator
    thresholds = {
        "vertical_up_min": max(0.0, vert_up_med * 0.8),  # Minimum vertical for "up" (80% of median)
        "index_up_min": max(0.0, idx_up_med * 0.85),     # Keep for backward compatibility
        "others_down_max": oth_fist_med * 1.15,          # others should be <= this
        "fist_all_max": idx_fist_med * 1.15,             # index should be <= this for fist
        "hysteresis": 0.05
    }
    return thresholds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=0, help="camera index or path to video/image")
    ap.add_argument("--model", default="yolov8n-pose.pt", help="YOLO pose model path (v8/11)")
    ap.add_argument("--device", default=None, help="device id, e.g., 0 for CUDA; leave blank for auto")
    ap.add_argument("--view", action="store_true", help="show visualization windows")
    ap.add_argument("--save", default="index_thresh.json", help="where to save thresholds json")
    args = ap.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(int(args.source)) if str(args.source).isdigit() else cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("ERROR: Cannot open source:", args.source)
        sys.exit(1)

    thresholds = run_calibration(model, cap, args.device, args.view)
    cap.release()
    if args.view:
        cv2.destroyAllWindows()

    if thresholds is None:
        print("Calibration failed or aborted.")
        sys.exit(2)

    with open(args.save, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Saved thresholds to: {args.save}")
    print(json.dumps(thresholds, indent=2))

if __name__ == "__main__":
    main()
