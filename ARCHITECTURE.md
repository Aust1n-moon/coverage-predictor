# Coverage Predictor — System Architecture

## 1. Stack

| Layer | Tech | Hosting | Role |
|:---|:---|:---|:---|
| **Frontend** | React / TypeScript | Vercel (Free) | Video upload, visualization, browsable play history |
| **Detection Model** | Custom YOLOv11m (fine-tuned) | Trained via Roboflow + Ultralytics | Player/ref detection tuned for overlapping and sideline players |
| **Backend** | FastAPI + YOLO + ByteTrack + MLP | Hugging Face Spaces (Free) or Docker on private server | Video inference, tracking, feature extraction, coverage prediction |
| **Database** | Supabase | Supabase (Free Tier) | Historical predictions, clip references, confidence logs |

---

## 2. Constraints & Scope

*   **Camera Angle:** All-22 coaches film only (ensures all 22 players and deep safeties remain in-frame).
*   **Play Type:** Dropback passing plays only (eliminates run-fit and blocking scheme noise).
*   **Training Data:** Kaggle NFL Big Data Bowl tracking CSVs (ground-truth coordinates and verified coverage labels).
*   **YOLO Training Data:** Custom Roboflow dataset annotated with two classes: `player` (class 0) and `ref` (class 1). Fine-tuned from `yolo11m.pt` to improve detection of overlapping players and sideline players.

---

## 3. Data Flow

```text
React (Vercel) ──► POST /analyze (video) ──► FastAPI (HF Spaces / Docker)
                                                ├── Custom YOLO .track() + ByteTrack
                                                ├── Drop ref detections by class
                                                ├── K-Means jersey color clustering
                                                ├── Filter OL + QB → 16 players
                                                ├── Extract 64 features
                                                ├── TF/Keras MLP predict
                                                └── Save to Supabase
               ◄── JSON (prediction, confidence) ◄──
```

---

## 4. Pipeline Breakdown

1.  **Video Ingestion:** React frontend sends the uploaded All-22 video clip via `POST /analyze` to the FastAPI backend.
2.  **Custom YOLO + ByteTrack:** Fine-tuned YOLOv11m detects `player` and `ref` bounding boxes (trained on Roboflow-annotated All-22 frames to handle overlapping players and sideline noise). ByteTrack assigns persistent IDs across frames.
3.  **Ref Filtering:** Detections with class `ref` are dropped by label before clustering.
4.  **Team Separation:** K-Means color clustering on jersey pixels within each remaining bounding box groups tracked players into two teams.
5.  **Receiver Filter:** Drops the 5 interior offensive linemen (clustered near the line of scrimmage) and QB from the offensive side.
6.  **Feature Extraction (64-Value Tensor):**
    *   Anchor frames captured at snap (t = 0) and post-snap (t = 1.5s - 2.0s).
    *   Coordinates normalized to a [0.0, 1.0] field grid.
    *   Players sorted horizontally (left-to-right) by position.
    *   **11 Defensive Players:** 11 × 4 (x, x', y, y') = 44 values.
    *   **5 Eligible Receivers:** 5 × 4 (x, x', y, y') = 20 values (OL + QB filtered out).
    *   **Total Input:** Flat 1D vector of **64 features** `(batch_size, 64)`.
7.  **MLP Prediction:** Multi-layer perceptron (TensorFlow/Keras Dense classifier) outputs a Softmax probability distribution over defensive coverage schemes (Cover 0, Cover 1, Cover 2, Cover 2 Man, Cover 3, Cover 4/Quarters, Blitz).
8.  **Persistence & Response:** Prediction and metadata are saved to Supabase; JSON response is returned to the frontend.

---

## 5. Supabase Schema

### Table: `predictions`
*   `id` (UUID, Primary Key)
*   `user_id` (UUID, Foreign Key to Supabase Auth, nullable)
*   `video_url` (TEXT)
*   `predicted_coverage` (TEXT)
*   `confidence` (FLOAT)
*   `feature_vector` (FLOAT[])
*   `created_at` (TIMESTAMPTZ, default: `NOW()`)

---

## 6. API Specification

### `POST /analyze`
*   **Request:** `multipart/form-data` with `file` (`.mp4` / `.mov`)
*   **Response (`200 OK`):**
```json
{
  "predicted_coverage": "Cover 3",
  "confidence": 0.884,
  "distribution": {
    "Cover 3": 0.884,
    "Cover 1": 0.082,
    "Cover 4": 0.021,
    "Cover 2": 0.013
  }
}
```
