# Coverage Predictor

Predicts defensive coverage schemes (Cover 0/1/2/3/4, 2-Man) from American football game film using computer vision and a neural network.

Takes an All-22 video clip of a dropback passing play → detects and tracks players with YOLO + ByteTrack → transforms pixel coordinates to field positions via homography → builds a 339-dimensional feature vector → classifies the coverage with an MLP.

## Pipeline

```
Video clip
  │
  ▼
YOLO (helmet detection, 2 classes) + ByteTrack (persistent IDs)
  │
  ▼
Broken track stitching (spatial proximity across occlusions)
  │
  ▼
4-point homography (pixel coords → field yards)
  │
  ▼
Offense/defense split (largest-gap heuristic at scrimmage)
  │
  ▼
339-dim feature vector (8 defenders × 25 + 6 offense × 22 + 7 formation)
  │
  ▼
MLP classifier → coverage prediction
```

## Project Structure

```
Yolo/
  track.py          # YOLO tracking + broken ID stitching → CSV
  pipeline.py       # Homography, O/D split, feature extraction, full inference pipeline
  bytetrack.yaml    # ByteTrack tracker config (buffer=60)
model/
  model.py          # PyTorch MLP (339 → 256 → 128 → 64 → 6)
  train.py          # Training script with StandardScaler preprocessing
data/
  processing.ipynb  # NFL Big Data Bowl CSV → 339-dim feature vectors → processed_2023.npz
```

## How It Works

### 1. Player Detection & Tracking (`Yolo/track.py`)

Fine-tuned YOLOv11m detects two classes: `Helmet` (on-field) and `Helmet-Sideline` (filtered out). ByteTrack assigns persistent IDs across frames. A post-processing step (`stitch_broken_tracks`) merges IDs that break during occlusion by matching tracks that disappear and reappear nearby within 15 frames / 50 pixels.

```bash
python Yolo/track.py path/to/clip.mp4 --model path/to/best.pt --out tracking.csv
```

### 2. Homography & Feature Extraction (`Yolo/pipeline.py`)

A 4-point calibration maps pixel coordinates to field yards (0–120 × 0–53.3) using `cv2.findHomography`. The camera is assumed static per clip. Players are split into offense/defense by estimating the line of scrimmage (largest gap in x-positions), then a 339-dim feature vector is built from snap + 10 post-snap frames:

| Component | Features per player | Count | Total |
|---|---|---|---|
| Defenders | x, y, depth, nearest receiver dist, dist from center, 20 displacement values | 8 | 200 |
| Offense | x, y, 20 displacement values | 6 | 132 |
| Formation | n_deep, n_box, safety_high, avg_depth, def_spread, avg_corner_press, max_depth | — | 7 |
| | | **Total** | **339** |

### 3. Training Data (`data/processing.ipynb`)

Built from 2023 NFL Big Data Bowl Kaggle tracking CSVs + play-by-play coverage labels via `nfl_data_py`. ~11,500 plays after filtering. Labels:

| Coverage | Count | % |
|---|---|---|
| Cover 1 | 3,593 | 31.3% |
| Cover 4 | 2,848 | 24.8% |
| Cover 3 | 2,221 | 19.3% |
| Cover 2 | 1,893 | 16.5% |
| 2-Man | 689 | 6.0% |
| Cover 0 | 248 | 2.2% |

Cover 6, Cover 9, and Combo are merged into Cover 4. Play direction is normalized (all plays go right).

### 4. Neural Network (`model/`)

PyTorch MLP: `339 → 256 → 128 → 64 → 6` with BatchNorm, ReLU, and 0.3 dropout between layers. Trained with Adam (lr=1e-4), CrossEntropyLoss, batch size 64, 200 epochs. Features are StandardScaler-normalized with zero-padding preserved.

## Known Limitations

- Offense/defense split uses a naive largest-gap heuristic — breaks on goal-line and exotic formations
- Homography requires manual 4-point calibration per video clip
- Snap frame must be manually specified
- Assumes all plays go left-to-right (no auto play-direction detection)
- Track stitching parameters (50px distance, 15-frame gap) tuned for ~1080p broadcast footage

## Requirements

- Python 3.10+
- ultralytics
- opencv-python
- torch
- numpy
- scikit-learn
- nfl_data_py (for data processing only)
