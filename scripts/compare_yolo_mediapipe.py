"""
Compare YOLOv8, YOLO11 Pose, and MediaPipe Hands on the same validation dataset.

Evaluates all three models on common metrics:
- Detection rate
- Average IoU
- Mean OKS
- PCK@0.05
- Mean L2 Error
- Mean inference time / FPS

Outputs:
  results/comparison/summary.json
  results/comparison/comparison_chart.png
"""

import os
import time
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from ultralytics import YOLO
import mediapipe as mp

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
VAL_IMAGES = BASE_DIR / "data" / "images" / "val"
VAL_LABELS = BASE_DIR / "data" / "labels" / "val"
YOLO8_MODEL_PATH = BASE_DIR / "models" / "best_v8.pt"
YOLO11_MODEL_PATH = BASE_DIR / "models" / "best.pt"
RESULTS_DIR = BASE_DIR / "results" / "comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- UTILITIES ----------------
def parse_yolo_label(label_path):
    """Parse YOLO format label file."""
    anns = []
    with open(label_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            bbox = [
                float(p[1]) - float(p[3]) / 2,
                float(p[2]) - float(p[4]) / 2,
                float(p[1]) + float(p[3]) / 2,
                float(p[2]) + float(p[4]) / 2,
            ]
            kpts = [list(map(float, p[i:i + 3])) for i in range(5, 5 + 63, 3) if i + 2 < len(p)]
            anns.append({"bbox": bbox, "keypoints": np.array(kpts)})
    return anns

def iou(boxA, boxB):
    """Compute IoU between two boxes."""
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

def oks(pred, gt, area, kappa=0.1):
    valid, oks_sum = 0, 0
    for i in range(len(gt)):
        if gt[i, 2] > 0:
            dx, dy = pred[i, 0] - gt[i, 0], pred[i, 1] - gt[i, 1]
            s = np.sqrt(area)
            oks_sum += np.exp(-((dx ** 2 + dy ** 2) / (2 * (s ** 2) * kappa ** 2)))
            valid += 1
    return oks_sum / valid if valid else 0

def pck(pred, gt, area, thresh=0.05):
    """Percentage of correct keypoints within threshold."""
    correct, total = 0, 0
    s = np.sqrt(area)
    for i in range(len(gt)):
        if gt[i, 2] > 0:
            dist = np.linalg.norm(pred[i, :2] - gt[i, :2])
            if dist < thresh * s:
                correct += 1
            total += 1
    return correct / total if total else 0

def mean_l2(pred, gt):
    valid_dists = [np.linalg.norm(pred[i, :2] - gt[i, :2])
                   for i in range(len(gt)) if gt[i, 2] > 0]
    return np.mean(valid_dists) if valid_dists else 0

# ---------------- YOLO EVALUATION ----------------
def evaluate_yolo(model_path, max_images=None):
    model = YOLO(str(model_path))
    img_files = sorted(list(VAL_IMAGES.glob("*.jpg")) + list(VAL_IMAGES.glob("*.png")))
    if max_images:
        img_files = img_files[:max_images]

    det_count, iou_scores, oks_scores, pck_scores, l2_scores, times = 0, [], [], [], [], []

    for img_path in tqdm(img_files, desc="YOLO eval"):
        label_path = VAL_LABELS / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        anns = parse_yolo_label(label_path)
        if not anns:
            continue
        gt = anns[0]
        gt_kpts, gt_box = gt["keypoints"], gt["bbox"]
        area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

        start = time.time()
        res = model(img_path, verbose=False)[0]
        times.append((time.time() - start) * 1000)

        if len(res.keypoints.xy) == 0:
            continue
        det_count += 1

        pred_kpts = res.keypoints.xy[0].cpu().numpy() / np.array([res.orig_shape[1], res.orig_shape[0]])
        pred_box = res.boxes.xyxy[0].cpu().numpy() / np.array(
            [res.orig_shape[1], res.orig_shape[0], res.orig_shape[1], res.orig_shape[0]]
        )

        iou_scores.append(iou(gt_box, pred_box))
        oks_scores.append(oks(pred_kpts, gt_kpts, area))
        pck_scores.append(pck(pred_kpts, gt_kpts, area))
        l2_scores.append(mean_l2(pred_kpts, gt_kpts))

    total = len(img_files)
    return {
        "Detection Rate": det_count / total if total else 0,
        "Mean IoU": np.mean(iou_scores) if iou_scores else 0,
        "Mean OKS": np.mean(oks_scores) if oks_scores else 0,
        "PCK@0.05": np.mean(pck_scores) if pck_scores else 0,
        "Mean L2 Error": np.mean(l2_scores) if l2_scores else 0,
        "Mean Inference Time (ms)": np.mean(times) if times else 0,
        "FPS": 1000 / np.mean(times) if times else 0,
    }

# ---------------- MEDIAPIPE EVALUATION ----------------
def evaluate_mediapipe(max_images=None):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=1)
    img_files = sorted(list(VAL_IMAGES.glob("*.jpg")) + list(VAL_IMAGES.glob("*.png")))
    if max_images:
        img_files = img_files[:max_images]

    det_count, iou_scores, oks_scores, pck_scores, l2_scores, times = 0, [], [], [], [], []

    for img_path in tqdm(img_files, desc="MediaPipe eval"):
        label_path = VAL_LABELS / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        anns = parse_yolo_label(label_path)
        if not anns:
            continue
        gt = anns[0]
        gt_kpts, gt_box = gt["keypoints"], gt["bbox"]
        area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

        img = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        start = time.time()
        res = hands.process(rgb)
        times.append((time.time() - start) * 1000)

        if not res.multi_hand_landmarks:
            continue
        det_count += 1

        pred_kpts = np.array([[lm.x, lm.y, 1.0] for lm in res.multi_hand_landmarks[0].landmark])
        pred_box = [pred_kpts[:, 0].min(), pred_kpts[:, 1].min(),
                    pred_kpts[:, 0].max(), pred_kpts[:, 1].max()]

        iou_scores.append(iou(gt_box, pred_box))
        oks_scores.append(oks(pred_kpts, gt_kpts, area))
        pck_scores.append(pck(pred_kpts, gt_kpts, area))
        l2_scores.append(mean_l2(pred_kpts, gt_kpts))

    total = len(img_files)
    return {
        "Detection Rate": det_count / total if total else 0,
        "Mean IoU": np.mean(iou_scores) if iou_scores else 0,
        "Mean OKS": np.mean(oks_scores) if oks_scores else 0,
        "PCK@0.05": np.mean(pck_scores) if pck_scores else 0,
        "Mean L2 Error": np.mean(l2_scores) if l2_scores else 0,
        "Mean Inference Time (ms)": np.mean(times) if times else 0,
        "FPS": 1000 / np.mean(times) if times else 0,
    }

# ---------------- COMPARISON & PLOT ----------------
def compare_models(yolo8_metrics, yolo11_metrics, mp_metrics):
    summary = {"YOLOv8": yolo8_metrics, "YOLO11": yolo11_metrics, "MediaPipe": mp_metrics}

    metrics = list(yolo8_metrics.keys())
    yolo8_vals = [yolo8_metrics[m] for m in metrics]
    yolo11_vals = [yolo11_metrics[m] for m in metrics]
    mp_vals = [mp_metrics[m] for m in metrics]

    # Save JSON
    out_json = RESULTS_DIR / "summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Summary saved to {out_json}")

    # Radar / spider chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    yolo8_vals += yolo8_vals[:1]
    yolo11_vals += yolo11_vals[:1]
    mp_vals += mp_vals[:1]
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], metrics, color="grey", size=9)
    ax.plot(angles, yolo8_vals, linewidth=2, label="YOLOv8")
    ax.fill(angles, yolo8_vals, alpha=0.2)
    ax.plot(angles, yolo11_vals, linewidth=2, label="YOLO11")
    ax.fill(angles, yolo11_vals, alpha=0.2)
    ax.plot(angles, mp_vals, linewidth=2, label="MediaPipe")
    ax.fill(angles, mp_vals, alpha=0.2)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.title("YOLOv8 vs YOLO11 vs MediaPipe Comparison", size=13, weight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_chart.png", dpi=150)
    plt.close()
    print(f"📊 Chart saved to {RESULTS_DIR / 'comparison_chart.png'}")

# ---------------- MAIN ----------------
def main():
    print("\n=== Evaluating YOLOv8 Pose ===")
    yolo8_metrics = evaluate_yolo(YOLO8_MODEL_PATH)
    print(json.dumps(yolo8_metrics, indent=2))

    print("\n=== Evaluating YOLO11 Pose ===")
    yolo11_metrics = evaluate_yolo(YOLO11_MODEL_PATH)
    print(json.dumps(yolo11_metrics, indent=2))

    print("\n=== Evaluating MediaPipe Hands ===")
    mp_metrics = evaluate_mediapipe()
    print(json.dumps(mp_metrics, indent=2))

    compare_models(yolo8_metrics, yolo11_metrics, mp_metrics)

if __name__ == "__main__":
    main()
