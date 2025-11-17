# Index Finger "Straight Up" Counter

## Overview
Improved gesture recognition script that detects index finger up/down gestures, including when the finger is held straight up (perpendicular to camera) - a limitation in previous versions.

## Files Included
- **`index_counter_openfist_straightup.py`** - Main live counter script (ready to use)
- **`index_thresh_backup_20251117_022411.json`** - Pre-calibrated thresholds (working, but personalized)
- **`straightup_summary.pdf`** - Executive summary document

## Quick Start

### 1. Install Dependencies
```bash
pip install opencv-python numpy ultralytics
```

### 2. Run the Counter
```bash
python index_counter_openfist_straightup.py --source 1 --model models/best.pt --thresholds index_thresh_backup_20251117_022411.json --view
```

**Arguments:**
- `--source 1` - Camera index (use `0` for laptop camera, `1` for external)
- `--model models/best.pt` - YOLO pose model path
- `--thresholds` - Path to thresholds JSON file
- `--view` - Show visualization window

### 3. Usage
- **Hold index finger straight up** → Shows "INDEX_UP"
- **Make a fist** → Shows "FIST"
- **Alternate between them** → Counter increments
- **Press 'q' or ESC** → Exit

## Important: Personalized Calibration

⚠️ **The included thresholds are calibrated for a specific user/hand size.**

**For other users:**
- Thresholds may be too sensitive or not sensitive enough
- May experience misclassification or stuck states
- **Solution:** Use the original `index_counter_openfist_autocalib.py` to generate personalized thresholds, then use those with this script

**Example:**
```bash
# Step 1: Generate your own thresholds
python index_counter_openfist_autocalib.py --source 1 --model models/best.pt --view --save my_thresh.json

# Step 2: Use with straight-up counter
python index_counter_openfist_straightup.py --source 1 --model models/best.pt --thresholds my_thresh.json --view
```

## Troubleshooting

**Counter stuck on one gesture:**
- Thresholds may not match your hand size - try recalibrating

**Labels inverted (shows FIST when finger is up):**
- Thresholds need adjustment - use calibration script

**Counter not starting:**
- Make sure hand is visible in camera
- Check camera index (`--source 0` vs `--source 1`)

**No detection:**
- Ensure good lighting
- Hand should be clearly visible
- Try adjusting camera distance

## Key Improvements Over Previous Versions

- ✅ Detects finger when straight up (not just 30-40° tilt)
- ✅ Stable classification (no drift over time)
- ✅ Responsive counter (doesn't get stuck)
- ✅ Large, clear UI for patient-facing interface
- ✅ Works with both straight-up and tilted finger poses

## About Calibration Scripts

**Do you need to fix the straight-up autocalib?**

**Short term: No.** The original `index_counter_openfist_autocalib.py` works as a workaround. The backup thresholds came from it and work with the straight-up script.

**Long term: Yes.** The straight-up script uses:
- **Angle-based detection** (straight-up vs tilted)
- **Different thresholds for straight-up cases** (10% lower)
- **Logic that prioritizes normalized distance** for straight-up poses

The original autocalib doesn't account for these, so thresholds may be suboptimal. A dedicated straight-up calibration script is in development to generate thresholds optimized for this specific use case.

## Notes

- For now, use the original `index_counter_openfist_autocalib.py` for calibration
- The script uses a 15-frame smoothing window for stability

