"""
Evaluate three rule-based gesture classifiers on validation dataset with MediaPipe benchmark.

Evaluates:
1. Open/Close classifier (open_palm vs fist)
2. Fist/Thumbs-up classifier (thumbs_up vs fist)
3. Index finger classifier (index_up vs fist)

Benchmark:
- MediaPipe Hands (for comparison)

Features:
- Optimized detection confidence threshold (0.15)
- Keypoint quality validation
- Comprehensive metrics (accuracy, precision, recall, F1-score)
- Confusion matrices
"""

import os
import json
import csv
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import sys

# Import classifier functions
sys.path.insert(0, str(Path(__file__).parent))
from gesture_counter_openclose_palmnorm import classify_open_close, GestureConfig as OpenCloseConfig
from gesture_counter_fist_thumbsup import classify_fist_thumbsup, GestureConfig as FistThumbsupConfig

from ultralytics import YOLO

# Try to import MediaPipe (optional)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
VAL_IMAGES = BASE_DIR / "validation" / "images"
VAL_LABELS_CSV = BASE_DIR / "validation" / "labels.csv"
MODEL_PATH = BASE_DIR / "models" / "best.pt"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Detection settings (optimized for best performance)
DETECTION_CONF_THRESHOLD = 0.15  # Optimal threshold from diagnosis (valid keypoint rate: 0.624)
MIN_KEYPOINT_CONFIDENCE = 0.15   # Lowered minimum keypoint confidence

# ---------------- KEYPOINT VALIDATION ----------------
def validate_keypoints(kps_xy, kp_conf=None):
    """
    Validate keypoint geometry for anatomical consistency.
    Returns True if keypoints are valid, False otherwise.
    """
    if kps_xy.shape[0] < 21:
        return False
    
    # Check palm width/length ratio (should be roughly square-ish)
    palm_width = np.linalg.norm(kps_xy[5] - kps_xy[17])  # index_base to pinky_base
    palm_len = np.linalg.norm(kps_xy[0] - kps_xy[9])     # wrist to middle_base
    
    if palm_width < 1e-6 or palm_len < 1e-6:
        return False
    
    ratio = palm_width / palm_len
    if ratio < 0.3 or ratio > 3.0:  # More lenient palm proportions
        return False
    
    # Check keypoint confidence if available
    if kp_conf is not None and len(kp_conf) >= 21:
        avg_conf = np.mean(kp_conf)
        if avg_conf < MIN_KEYPOINT_CONFIDENCE:
            return False
    
    # Basic geometric check: fingertips should generally be further from wrist than knuckles
    wrist = kps_xy[0]
    finger_bases = [1, 5, 9, 13, 17]  # Thumb, index, middle, ring, pinky bases
    finger_tips = [4, 8, 12, 16, 20]
    
    valid_fingers = 0
    for base_idx, tip_idx in zip(finger_bases, finger_tips):
        tip_dist = np.linalg.norm(kps_xy[tip_idx] - wrist)
        base_dist = np.linalg.norm(kps_xy[base_idx] - wrist)
        # Tip should be at least 50% as far as base (more lenient for curled fingers)
        if tip_dist >= base_dist * 0.5:
            valid_fingers += 1
    
    # Require at least 2 valid fingers (more lenient)
    return valid_fingers >= 2

# ---------------- INDEX FINGER CLASSIFIER ----------------
def classify_index_fist(kps_xy, thresholds=None):
    """
    Index finger classifier with optimized thresholds.
    """
    if kps_xy.shape[0] <= 20:
        return None
    
    # Much more lenient thresholds (based on analysis)
    if thresholds is None:
        thresholds = {
            "vertical_up_min": 0.05,   # Much reduced from 0.15
            "index_up_min": 0.50,       # Reduced from 0.65
            "others_down_max": 0.65,    # Increased from 0.50
            "fist_all_max": 0.55        # Increased from 0.45
        }
    
    # Compute palm scale
    palm_width = np.linalg.norm(kps_xy[5] - kps_xy[17])
    palm_len = np.linalg.norm(kps_xy[0] - kps_xy[9])
    scale = max((palm_width + palm_len) / 2.0, 1e-6)
    
    # Index finger distance
    index_dist = np.linalg.norm(kps_xy[8] - kps_xy[5]) / scale
    
    # Other fingers (middle, ring, pinky)
    middle_dist = np.linalg.norm(kps_xy[12] - kps_xy[9]) / scale
    ring_dist = np.linalg.norm(kps_xy[16] - kps_xy[13]) / scale
    pinky_dist = np.linalg.norm(kps_xy[20] - kps_xy[17]) / scale
    others_avg = np.mean([middle_dist, ring_dist, pinky_dist])
    
    # Vertical component (index tip higher than base)
    index_base_y = kps_xy[5, 1]
    index_tip_y = kps_xy[8, 1]
    vertical_component = (index_base_y - index_tip_y) / scale
    
    # Index up: index extended and vertical, others curled
    if (vertical_component > thresholds["vertical_up_min"] and 
        index_dist > thresholds["index_up_min"] and 
        others_avg < thresholds["others_down_max"]):
        return "index_up"
    
    # Fist: all fingers curled
    all_avg = np.mean([index_dist, middle_dist, ring_dist, pinky_dist])
    if all_avg < thresholds["fist_all_max"]:
        return "fist"
    
    return None

# ---------------- MEDIAPIPE HELPERS ----------------
def mediapipe_landmarks_to_keypoints(landmarks, img_width, img_height):
    """Convert MediaPipe landmarks to pixel coordinates."""
    kps = []
    for lm in landmarks.landmark:
        x = lm.x * img_width
        y = lm.y * img_height
        kps.append([x, y])
    return np.array(kps)

# ---------------- LOAD VALIDATION DATA ----------------
def load_validation_data():
    """Load image paths and ground truth labels."""
    data = []
    with open(VAL_LABELS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = VAL_IMAGES / row['image_filename']
            if img_path.exists():
                data.append({
                    'image_path': img_path,
                    'ground_truth': row['gesture_label']
                })
    return data

# ---------------- EVALUATION FUNCTION ----------------
def evaluate_classifier(classifier_name, classifier_func, config, validation_data, model, 
                        relevant_classes, use_mediapipe=False, improved_detection=True):
    """
    Evaluation function with optimized detection handling.
    """
    predictions = []
    ground_truths = []
    detection_count = 0
    valid_keypoint_count = 0
    total_count = len(validation_data)
    
    # Initialize MediaPipe if needed
    mp_hands = None
    hands = None
    if use_mediapipe and MEDIAPIPE_AVAILABLE:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=1)
    
    conf_threshold = DETECTION_CONF_THRESHOLD if improved_detection else 0.45
    
    for item in tqdm(validation_data, desc=f"Evaluating {classifier_name}"):
        gt = item['ground_truth']
        
        # Only evaluate on relevant classes
        if gt not in relevant_classes:
            continue
        
        pred = None
        
        if use_mediapipe and MEDIAPIPE_AVAILABLE:
            # Use MediaPipe for detection
            img = cv2.imread(str(item['image_path']))
            if img is None:
                predictions.append("no_detection")
                ground_truths.append(gt)
                continue
                
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            res = hands.process(rgb)
            
            if res.multi_hand_landmarks:
                landmarks = res.multi_hand_landmarks[0]
                kps_xy = mediapipe_landmarks_to_keypoints(landmarks, w, h)
                if kps_xy.shape[0] >= 21:
                    if validate_keypoints(kps_xy):
                        pred = classifier_func(kps_xy, config)
                        valid_keypoint_count += 1
                    detection_count += 1
        else:
            # Use YOLO for detection with improved settings
            results = model.predict(str(item['image_path']), verbose=False, conf=conf_threshold)
            
            for r in results:
                if len(r.boxes) > 0 and r.keypoints is not None:
                    kps_xy = r.keypoints.xy[0].cpu().numpy()
                    kp_conf = None
                    
                    # Get keypoint confidence if available
                    if hasattr(r.keypoints, 'conf') and len(r.keypoints.conf) > 0:
                        kp_conf = r.keypoints.conf[0].cpu().numpy()
                    
                    if kps_xy.shape[0] >= 21:
                        detection_count += 1
                        # Validate keypoints before classification
                        if validate_keypoints(kps_xy, kp_conf):
                            pred = classifier_func(kps_xy, config)
                            valid_keypoint_count += 1
                        break
        
        predictions.append(pred if pred else "no_detection")
        ground_truths.append(gt)
    
    # Clean up MediaPipe
    if hands is not None:
        hands.close()
    
    # Filter out no_detection cases for metrics calculation
    valid_indices = [i for i, p in enumerate(predictions) if p != "no_detection"]
    if not valid_indices:
        return {
            'classifier': classifier_name,
            'detection_rate': 0,
            'valid_keypoint_rate': 0,
            'accuracy': 0,
            'precision': {},
            'recall': {},
            'f1_score': {},
            'confusion_matrix': {},
            'total_samples': len(ground_truths),
            'detected_samples': 0,
            'valid_keypoint_samples': 0
        }
    
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_ground_truths = [ground_truths[i] for i in valid_indices]
    
    # Calculate metrics
    accuracy = accuracy_score(valid_ground_truths, valid_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        valid_ground_truths, valid_predictions, labels=relevant_classes, zero_division=0
    )
    
    # Build metrics dict
    metrics = {
        'classifier': classifier_name,
        'detection_rate': detection_count / total_count if total_count > 0 else 0,
        'valid_keypoint_rate': valid_keypoint_count / total_count if total_count > 0 else 0,
        'accuracy': float(accuracy),
        'precision': {cls: float(p) for cls, p in zip(relevant_classes, precision)},
        'recall': {cls: float(r) for cls, r in zip(relevant_classes, recall)},
        'f1_score': {cls: float(f) for cls, f in zip(relevant_classes, f1)},
        'support': {cls: int(s) for cls, s in zip(relevant_classes, support)},
        'total_samples': len(ground_truths),
        'detected_samples': len(valid_predictions),
        'valid_keypoint_samples': valid_keypoint_count,
        'no_detection_count': len(predictions) - len(valid_predictions)
    }
    
    # Confusion matrix
    cm = confusion_matrix(valid_ground_truths, valid_predictions, labels=relevant_classes)
    metrics['confusion_matrix'] = {
        'labels': relevant_classes,
        'matrix': cm.tolist()
    }
    
    return metrics

# ---------------- TABLE FORMATTING ----------------
def format_results_table(all_metrics):
    """Format evaluation results as a table."""
    lines = []
    lines.append("=" * 100)
    lines.append("RULE-BASED GESTURE CLASSIFIER EVALUATION RESULTS")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Detection Settings: conf_threshold={DETECTION_CONF_THRESHOLD}, min_kp_conf={MIN_KEYPOINT_CONFIDENCE}")
    lines.append("")
    
    # Group by gesture task
    tasks = {
        'Open/Close': ['Open/Close Classifier', 'MediaPipe (Open/Close)'],
        'Fist/Thumbs-up': ['Fist/Thumbs-up Classifier', 'MediaPipe (Fist/Thumbs-up)'],
        'Index Finger': ['Index Finger Classifier', 'MediaPipe (Index Finger)']
    }
    
    for task_name, classifier_names in tasks.items():
        lines.append(f"\n{task_name.upper()} TASK")
        lines.append("=" * 100)
        
        task_metrics = [m for m in all_metrics if m['classifier'] in classifier_names]
        
        for metrics in task_metrics:
            classifier = metrics['classifier']
            lines.append(f"\n{classifier}")
            lines.append("-" * 100)
            lines.append(f"Detection Rate: {metrics['detection_rate']:.3f} ({metrics['detected_samples']}/{metrics['total_samples']})")
            if 'valid_keypoint_rate' in metrics:
                lines.append(f"Valid Keypoint Rate: {metrics['valid_keypoint_rate']:.3f} ({metrics['valid_keypoint_samples']}/{metrics['total_samples']})")
            lines.append(f"Overall Accuracy: {metrics['accuracy']:.3f}")
            lines.append("")
            lines.append("Per-Class Metrics:")
            lines.append(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
            lines.append("-" * 100)
            
            for cls in metrics['precision'].keys():
                prec = metrics['precision'][cls]
                rec = metrics['recall'][cls]
                f1 = metrics['f1_score'][cls]
                sup = metrics['support'][cls]
                lines.append(f"{cls:<15} {prec:<12.3f} {rec:<12.3f} {f1:<12.3f} {sup:<10}")
            
            lines.append("")
            lines.append("Confusion Matrix:")
            cm_labels = metrics['confusion_matrix']['labels']
            cm_matrix = np.array(metrics['confusion_matrix']['matrix'])
            
            # Header
            header = " " * 15 + " ".join([f"{label:>10}" for label in cm_labels])
            lines.append(header)
            lines.append("-" * len(header))
            
            # Rows
            for i, label in enumerate(cm_labels):
                row = f"{label:<15}" + " ".join([f"{cm_matrix[i,j]:>10}" for j in range(len(cm_labels))])
                lines.append(row)
            lines.append("")
    
    lines.append("=" * 100)
    return "\n".join(lines)

# ---------------- MAIN ----------------
def main():
    print("=" * 100)
    print("Evaluating Rule-Based Gesture Classifiers")
    print("=" * 100)
    print(f"\nConfiguration:")
    print(f"  - Detection confidence threshold: {DETECTION_CONF_THRESHOLD}")
    print(f"  - Keypoint validation: enabled")
    print(f"  - Minimum keypoint confidence: {MIN_KEYPOINT_CONFIDENCE}")
    
    if not MEDIAPIPE_AVAILABLE:
        print("\n⚠️  WARNING: MediaPipe is not available. Benchmark will be skipped.")
        print("   Install with: pip install mediapipe")
        print("   Continuing with rule-based classifiers only...\n")
    
    # Load validation data
    print("\nLoading validation dataset...")
    validation_data = load_validation_data()
    print(f"Loaded {len(validation_data)} validation images")
    
    # Load YOLO model
    print(f"\nLoading YOLO model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    
    # Evaluate each classifier
    all_metrics = []
    
    # 1. Open/Close Classifier (YOLO-based)
    print("\n" + "=" * 100)
    print("Evaluating Open/Close Classifier (open_palm vs fist) - YOLO")
    print("=" * 100)
    open_close_config = OpenCloseConfig()
    metrics1 = evaluate_classifier(
        "Open/Close Classifier",
        classify_open_close,
        open_close_config,
        validation_data,
        model,
        relevant_classes=['open_palm', 'fist'],
        use_mediapipe=False,
        improved_detection=True
    )
    all_metrics.append(metrics1)
    
    # 1b. Open/Close Classifier (MediaPipe benchmark)
    if MEDIAPIPE_AVAILABLE:
        print("\n" + "=" * 100)
        print("Evaluating Open/Close Classifier (open_palm vs fist) - MediaPipe Benchmark")
        print("=" * 100)
        metrics1b = evaluate_classifier(
            "MediaPipe (Open/Close)",
            classify_open_close,
            open_close_config,
            validation_data,
            None,
            relevant_classes=['open_palm', 'fist'],
            use_mediapipe=True,
            improved_detection=True
        )
        all_metrics.append(metrics1b)
    
    # 2. Fist/Thumbs-up Classifier (YOLO-based)
    print("\n" + "=" * 100)
    print("Evaluating Fist/Thumbs-up Classifier (thumbs_up vs fist) - YOLO")
    print("=" * 100)
    fist_thumbsup_config = FistThumbsupConfig()
    # Improve thumbs-up recall with more lenient thresholds
    fist_thumbsup_config.thumb_extended_thresh = 1.25  # Reduced from 1.434
    fist_thumbsup_config.finger_curled_thresh = 0.60   # Increased from 0.456
    metrics2 = evaluate_classifier(
        "Fist/Thumbs-up Classifier",
        classify_fist_thumbsup,
        fist_thumbsup_config,
        validation_data,
        model,
        relevant_classes=['thumbs_up', 'fist'],
        use_mediapipe=False,
        improved_detection=True
    )
    all_metrics.append(metrics2)
    
    # 2b. Fist/Thumbs-up Classifier (MediaPipe benchmark)
    if MEDIAPIPE_AVAILABLE:
        print("\n" + "=" * 100)
        print("Evaluating Fist/Thumbs-up Classifier (thumbs_up vs fist) - MediaPipe Benchmark")
        print("=" * 100)
        metrics2b = evaluate_classifier(
            "MediaPipe (Fist/Thumbs-up)",
            classify_fist_thumbsup,
            fist_thumbsup_config,
            validation_data,
            None,
            relevant_classes=['thumbs_up', 'fist'],
            use_mediapipe=True,
            improved_detection=True
        )
        all_metrics.append(metrics2b)
    
    # 3. Index Finger Classifier (YOLO-based)
    print("\n" + "=" * 100)
    print("Evaluating Index Finger Classifier (index_up vs fist) - YOLO")
    print("=" * 100)
    improved_index_thresholds = {
        "vertical_up_min": 0.05,   # Much more lenient
        "index_up_min": 0.50,       # Reduced
        "others_down_max": 0.65,    # Increased
        "fist_all_max": 0.55        # Increased
    }
    metrics3 = evaluate_classifier(
        "Index Finger Classifier",
        lambda kps, cfg: classify_index_fist(kps, improved_index_thresholds),
        None,
        validation_data,
        model,
        relevant_classes=['index_up', 'fist'],
        use_mediapipe=False,
        improved_detection=True
    )
    all_metrics.append(metrics3)
    
    # 3b. Index Finger Classifier (MediaPipe benchmark)
    if MEDIAPIPE_AVAILABLE:
        print("\n" + "=" * 100)
        print("Evaluating Index Finger Classifier (index_up vs fist) - MediaPipe Benchmark")
        print("=" * 100)
        metrics3b = evaluate_classifier(
            "MediaPipe (Index Finger)",
            lambda kps, cfg: classify_index_fist(kps, improved_index_thresholds),
            None,
            validation_data,
            None,
            relevant_classes=['index_up', 'fist'],
            use_mediapipe=True,
            improved_detection=True
        )
        all_metrics.append(metrics3b)
    
    # Save results
    results_json = RESULTS_DIR / "gesture_classifier_evaluation.json"
    with open(results_json, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n✅ Results saved to {results_json}")
    
    # Generate and save table
    table = format_results_table(all_metrics)
    results_table = RESULTS_DIR / "gesture_classifier_evaluation_table.txt"
    with open(results_table, 'w') as f:
        f.write(table)
    print(f"✅ Table saved to {results_table}")
    
    # Print table to console
    print("\n" + table)
    
    print("\n" + "=" * 100)
    print("Evaluation Complete!")
    print("=" * 100)
    print("\n💡 Compare with baseline results to see improvements:")
    print("   - Detection rate should increase significantly")
    print("   - Index finger classifier should now detect index_up gestures")
    print("   - Overall accuracy should improve across all tasks")

if __name__ == "__main__":
    main()

