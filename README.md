# Coverage Predictor

Predicts defensive coverage schemes (Cover 0/1/2/3/4, 2-Man) from American football game film using computer vision and a neural network.

Takes a video clip of a dropback passing play → detects and tracks players with YOLO + ByteTrack → auto-detects the snap frame → auto-detects yard lines and hash marks for field calibration → splits teams by jersey color → builds a 339-dimensional feature vector → classifies the coverage with an MLP.

## Quick Start

```bash
pip install -r requirements.txt

# Web UI (Gradio)
python app.py
# Opens at http://localhost:7860

# CLI
python run.py "test videos/clip1.mp4" --yard-line 40 --side left
```

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
Auto snap frame detection (per-player displacement spike, min 14 tracked)
  │
  ▼
Auto yard-line & hash mark detection (Hough lines on white/green)
  → homography: pixel coords → field yards (0–120 × 0–53.3)
  │
  ▼
K-means jersey color clustering → offense/defense split
  │
  ▼
Play direction normalization (flip coords when offense goes left)
  │
  ▼
339-dim feature vector (8 defenders × 25 + 6 offense × 22 + 7 formation)
  │
  ▼
MLP classifier → coverage prediction
```

## Usage

### Web UI

```bash
python app.py
```

Upload a video clip, set parameters, click Predict. The app auto-detects the snap frame and calibrates the field.

### CLI

```bash
python run.py <video> --yard-line N --side left|right [options]
```

Arguments:
- `video` — path to game video clip
- `--yard-line` — leftmost visible yard line number (0–50, required)
- `--side` — which side of midfield the play is on (required)
- `--play-direction` — direction offense is moving: `left` or `right` (default: right)
- `--yard-interval` — yards between visible lines: `5` for All-22, `10` for broadcast (default: 5)
- `--snap-frame` — frame index of the snap (auto-detected if omitted)
- `--conf` — YOLO confidence threshold (default: 0.5)

## Project Structure

```
app.py              # Gradio web UI — video upload → prediction
run.py              # CLI pipeline: video → tracking → features → prediction
requirements.txt    # Python dependencies
Yolo/
  track.py          # YOLO tracking + broken ID stitching
  pipeline.py       # Snap detection, auto-calibration, homography, jersey clustering, features
  bytetrack.yaml    # ByteTrack tracker config (buffer=60)
model/
  model.py          # PyTorch MLP (339 → 256 → 128 → 64 → 6)
  train.py          # Training script, saves model + scaler to coverage_model.pt
  coverage_model.pt # Trained model checkpoint
data/
  processing.ipynb  # NFL Big Data Bowl CSV → 339-dim feature vectors → processed_2023.npz
test videos/        # Test video clips
```

## How It Works

### 1. Player Detection & Tracking (`Yolo/track.py`)

Fine-tuned YOLOv11m detects two classes: `Helmet` (on-field) and `Helmet-Sideline` (filtered out). ByteTrack assigns persistent IDs across frames. A post-processing step (`stitch_broken_tracks`) merges IDs that break during occlusion by matching tracks that disappear and reappear nearby within 15 frames / 50 pixels.

### 2. Snap Frame Detection (`Yolo/pipeline.py`)

Automatic snap detection uses per-player displacement averaged across all tracked players. Frames with fewer than 14 tracked players are filtered out (skips huddle periods). The first sustained motion spike above baseline is identified as the snap, and the function returns a few frames before to capture the last pre-snap frame.

### 3. Auto-Calibration & Feature Extraction (`Yolo/pipeline.py`)

**Auto-calibration:** Yard lines are detected via Hough line transform on white pixels within the green field mask. Hash marks are found as short horizontal white segments, clustered into two y-bands (top hash at y=29.72, bottom hash at y=23.58). Combined with the user-specified leftmost yard line number, this produces 10+ calibration points for a full homography. If the snap frame is too zoomed in, calibration falls back to earlier frames automatically.

**Team splitting:** K-means (k=2) on jersey colors extracted from each player's bounding box, with green field pixels masked out. The cluster with lower average x-position = offense (behind LOS).

**Feature vector:** Position-based nearest-neighbor matching tracks players across post-snap frames. 339-dim vector built from snap + 10 post-snap frames:

| Component | Features per player | Count | Total |
|---|---|---|---|
| Defenders | x, y, depth, nearest receiver dist, dist from center, 20 displacement values | 8 | 200 |
| Offense | x, y, 20 displacement values | 6 | 132 |
| Formation | n_deep, n_box, safety_high, avg_depth, def_spread, avg_corner_press, max_depth | — | 7 |
| | | **Total** | **339** |

### 4. Training Data (`data/processing.ipynb`)

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

### 5. Neural Network (`model/`)

PyTorch MLP: `339 → 256 → 128 → 64 → 6` with BatchNorm, ReLU, and 0.3 dropout between layers. Trained with Adam (lr=1e-4), CrossEntropyLoss, batch size 64, 200 epochs. Features are StandardScaler-normalized with zero-padding preserved.

## Known Limitations

- Auto-calibration requires at least 2 visible yard lines and both hash mark rows
- Jersey color clustering can give uneven O/D splits on some broadcast angles
- Track stitching parameters (50px distance, 15-frame gap) tuned for ~1080p broadcast footage

## Requirements

- Python 3.10+
- ultralytics
- opencv-python
- torch
- numpy
- gradio
- lap
