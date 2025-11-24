"""
Evaluate YOLOv11 baseline (pretrained model) on the validation dataset.

Evaluates the pretrained YOLOv11 pose model (before finetuning) on:
- Detection rate
- Average IoU
- Mean OKS
- PCK@0.05
- Mean L2 Error
- Mean inference time / FPS

Outputs:
  results/comparison/yolo11_baseline_metrics.json
"""

import os
import time
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
VAL_IMAGES = BASE_DIR / "data" / "images" / "val"
VAL_LABELS = BASE_DIR / "data" / "labels" / "val"
YOLO11_BASELINE_MODEL = "train/yolo11n-pose.pt"  # Pretrained model name
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
def evaluate_yolo_baseline(model_name, max_images=None):
    """Evaluate pretrained YOLO model."""
    print(f"\nLoading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    img_files = sorted(list(VAL_IMAGES.glob("*.jpg")) + list(VAL_IMAGES.glob("*.png")))
    if max_images:
        img_files = img_files[:max_images]

    det_count, iou_scores, oks_scores, pck_scores, l2_scores, times = 0, [], [], [], [], []
    total_images = len(img_files)

    for img_path in tqdm(img_files, desc="YOLO11 baseline eval"):
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

        pred_kpts_raw = res.keypoints.xy[0].cpu().numpy()
        pred_kpts = pred_kpts_raw / np.array([res.orig_shape[1], res.orig_shape[0]])
        
        # Add visibility dimension if missing (assume all visible)
        if pred_kpts.shape[1] == 2:
            pred_kpts = np.hstack([pred_kpts, np.ones((pred_kpts.shape[0], 1))])
        
        pred_box = res.boxes.xyxy[0].cpu().numpy() / np.array(
            [res.orig_shape[1], res.orig_shape[0], res.orig_shape[1], res.orig_shape[0]]
        )

        # Handle keypoint count mismatch (pretrained model might have different number of keypoints)
        min_kpts = min(len(pred_kpts), len(gt_kpts))
        pred_kpts_aligned = pred_kpts[:min_kpts]
        gt_kpts_aligned = gt_kpts[:min_kpts]

        iou_scores.append(iou(gt_box, pred_box))
        oks_scores.append(oks(pred_kpts_aligned, gt_kpts_aligned, area))
        pck_scores.append(pck(pred_kpts_aligned, gt_kpts_aligned, area))
        l2_scores.append(mean_l2(pred_kpts_aligned, gt_kpts_aligned))

    return {
        "Detection Rate": det_count / total_images if total_images else 0,
        "Mean IoU": np.mean(iou_scores) if iou_scores else 0,
        "Mean OKS": np.mean(oks_scores) if oks_scores else 0,
        "PCK@0.05": np.mean(pck_scores) if pck_scores else 0,
        "Mean Distance Error": np.mean(l2_scores) if l2_scores else 0,
        "Mean Inference Time (ms)": np.mean(times) if times else 0,
        "FPS": 1000 / np.mean(times) if times else 0,
    }

# ---------------- MAIN ----------------
def main():
    print("="*70)
    print("Evaluating YOLOv11 Baseline (Pretrained Model)")
    print("="*70)
    
    metrics = evaluate_yolo_baseline(YOLO11_BASELINE_MODEL)
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    print(json.dumps(metrics, indent=2))
    
    # Save results
    out_json = RESULTS_DIR / "yolo11_baseline_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved to {out_json}")
    print("="*70)

if __name__ == "__main__":
    main()

