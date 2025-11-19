# Hand Pose Estimation & Gesture Recognition

A YOLO11-based hand pose estimation system for tracking rehabilitation exercises using 21-keypoint hand detection.

## Features

- Real-time hand keypoint detection (21 landmarks per hand)
- Gesture classification for open palm and fist
- Rep counter for palm open-close exercises
- Auto-calibration mode for personalized thresholds
- Comparison benchmarks vs MediaPipe Hands

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Download the hand keypoints dataset and extract it to the `data/` folder:

```bash
# Download from:
# https://github.com/ultralytics/assets/releases/download/v0.0.0/hand-keypoints.zip

# Extract to data/ directory
unzip hand-keypoints.zip -d data/
```

The dataset contains 26,768 images with 21 hand keypoint annotations per hand.

## Pre-trained Model

A pre-trained YOLO11 pose model is included at `models/best.pt`. You can use this directly for inference or train your own model using the provided notebook.

## Usage

### 1. Gesture Counter with Auto-Calibration

This script includes a calibration mode to personalize gesture detection thresholds.

Track open-close exercises with personalized calibration:


```bash
# Run calibration first (one-time setup)
python scripts/gesture_counter_openclose_autocalib.py --calibrate --weights models/best.pt

# Then run the counter
python scripts/gesture_counter_openclose_autocalib.py --weights models/best.pt
```

Track fist-to-thumbs-up exercises with personalized calibration:

```bash
# Run calibration first (one-time setup)
python scripts/gesture_counter_fist_thumbsup_autocalib.py --calibrate --weights models/best.pt

# Then run the counter
python scripts/gesture_counter_fist_thumbsup_autocalib.py --weights models/best.pt
```

**Controls:**
- `q` or `ESC`: Quit
- During calibration: Follow on-screen instructions to hold fist and thumbs-up poses

### 2. Gesture Counter (Palm-Normalized)

Open palm to fist counter with fixed thresholds:

```bash
python scripts/gesture_counter_openclose_palmnorm.py --weights models/best.pt
```

**Optional parameters:**
```bash
--open_thresh FLOAT   # Threshold for open palm detection (default: 0.75)
--fist_thresh FLOAT   # Threshold for fist detection (default: 0.45)
--spread_thresh FLOAT # Fingertip spread threshold (default: 0.45)
```

Open fist to thumbs-up counter with fixed thresholds:

```bash
python scripts/gesture_counter_fist_thumbsup.py --weights models/best.pt
```

**Optional parameters:**
```bash
--thumb_extended_thresh FLOAT  # Thumb extension threshold (default: 1.43)
--finger_curled_thresh FLOAT   # Other fingers curled threshold (default: 0.45)
--fist_thresh FLOAT            # Fist detection threshold (default: 0.75)
```

### 3. Model Comparison (YOLO11 vs MediaPipe)

Compare YOLO11 pose detection against MediaPipe Hands:

```bash
python scripts/compare_yolo_mediapipe.py
```

Results are saved to `results/comparison/`:
- `summary.json`: Metrics comparison
- `comparison_chart.png`: Visualization

## Gesture Definitions

The system recognizes the following gestures for rehabilitation exercises:

### Open Palm / Fist Exercises

1. **Open Palm**: All 5 fingers extended, fingertips spread apart
2. **Fist**: All fingers curled toward palm

**Rep Counting Logic:**
- Start with a fist
- Open palm fully (rep counter waits)
- Return to fist → counts as 1 rep
- Repeat

### Fist / Thumbs-Up Exercises

1. **Fist**: All 5 fingers curled toward palm
2. **Thumbs Up**: Thumb extended upward, other 4 fingers curled

**Rep Counting Logic:**
- Start with a fist
- Extend thumb to thumbs-up position (rep counter waits)
- Return to fist → counts as 1 rep
- Repeat

## Training Your Own Model

Use the included Jupyter notebook `yolo11_pose_est.ipynb` to:
- Train YOLO11-pose on the hand-keypoints dataset
- Customize training parameters (epochs, batch size, image size)
- Export and evaluate your model

