# Coverage Predictor

Predicts defensive coverage schemes (Cover 0/1/2/3/4, 2-Man) from American football game film using computer vision and a neural network.

Takes a video clip of a dropback passing play → detects and tracks players with YOLO + ByteTrack → auto-detects yard lines and hash marks for field calibration → splits teams by jersey color → builds a 339-dimensional feature vector → classifies the coverage with an MLP.

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
Auto yard-line & hash mark detection (Hough lines on white/green)
  → homography: pixel coords → field yards (0–120 × 0–53.3)
  │
  ▼
K-means jersey color clustering → offense/defense split
  │
  ▼
339-dim feature vector (8 defenders × 25 + 6 offense × 22 + 7 formation)
  │
  ▼
MLP classifier → coverage prediction
```

## Usage

```bash
# 1. Train the model (saves model/coverage_model.pt)
python model/train.py

# 2. Run the full pipeline on a video clip
python run.py "test videos/clip1.mp4" --snap-frame 100 --yard-line 40
```

Arguments:
- `video` — path to game video clip
- `--snap-frame` — frame index of the snap (required)
- `--yard-line` — leftmost visible yard line number, e.g. 10, 30, 40 (required). Must be an actual numbered yard line (10–50), not the goal line — endzone paint breaks auto-detection
- `--yolo-model` — custom YOLO weights (default: nfl-player-tracker-2class-2)
- `--nn-model` — custom NN checkpoint (default: model/coverage_model.pt)
- `--conf` — YOLO confidence threshold (default: 0.5)

The pipeline auto-detects yard lines and hash marks from the frame — no manual point clicking needed. Just specify which yard line is the leftmost visible one.

## Project Structure

```
run.py              # End-to-end pipeline: video → tracking → features → prediction
Yolo/
  track.py          # YOLO tracking + broken ID stitching
  pipeline.py       # Auto-calibration, homography, jersey color clustering, feature extraction
  bytetrack.yaml    # ByteTrack tracker config (buffer=60)
  test-vision.py    # Visual test: run YOLO on a video and display detections
model/
  model.py          # PyTorch MLP (339 → 256 → 128 → 64 → 6)
  train.py          # Training script, saves model + scaler to coverage_model.pt
data/
  processing.ipynb  # NFL Big Data Bowl CSV → 339-dim feature vectors → processed_2023.npz
test videos/        # Test video clips
```

## How It Works

### 1. Player Detection & Tracking (`Yolo/track.py`)

Fine-tuned YOLOv11m detects two classes: `Helmet` (on-field) and `Helmet-Sideline` (filtered out). ByteTrack assigns persistent IDs across frames. A post-processing step (`stitch_broken_tracks`) merges IDs that break during occlusion by matching tracks that disappear and reappear nearby within 15 frames / 50 pixels.

### 2. Auto-Calibration & Feature Extraction (`Yolo/pipeline.py`)

**Auto-calibration:** Yard lines are detected via Hough line transform on white pixels within the green field mask. Hash marks are found as short horizontal white segments, clustered into two y-bands (top hash at y=29.72, bottom hash at y=23.58). Combined with the user-specified leftmost yard line number, this produces 10+ calibration points for a full homography — no manual clicking needed.

**Team splitting:** K-means (k=2) on jersey colors extracted from each player's bounding box, with green field pixels masked out. The cluster with lower average x-position = offense (behind LOS).

**Feature vector:** Position-based nearest-neighbor matching tracks players across post-snap frames (bypasses unstable YOLO track IDs). 339-dim vector built from snap + 10 post-snap frames:

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

- Snap frame must be manually specified
- Assumes all plays go left-to-right (no auto play-direction detection)
- Auto-calibration requires at least 2 visible yard lines and both hash mark rows
- Track stitching parameters (50px distance, 15-frame gap) tuned for ~1080p broadcast footage

## Requirements

- Python 3.10+
- ultralytics
- opencv-python
- torch
- numpy
- scikit-learn
- nfl_data_py (for data processing only)
