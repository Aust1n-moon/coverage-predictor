"""
End-to-end pipeline: video → YOLO tracking → homography → 339-dim features → coverage prediction.

Usage:
    python run.py <video> --snap-frame N [--yolo-model path] [--nn-model path]

The snap frame is the frame index where the ball is snapped.
On first run, an OpenCV window opens for 4-point field calibration
(click 4 known yard-line points and enter their field coordinates).
"""

import argparse
import sys
import os

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Yolo'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

from track import track_video, stitch_broken_tracks
from pipeline import compute_homography, process_video_clip
from model import NeuralNetwork

COVERAGE_LABELS = ['2_MAN', 'COVER_0', 'COVER_1', 'COVER_2', 'COVER_3', 'COVER_4']


def calibrate(video_path, frame_idx):
    """Show a frame, let user click 4 points and enter field coordinates."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit(f"Could not read frame {frame_idx}")

    display = frame.copy()
    clicks = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(display, str(len(clicks)), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Calibration", display)

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration", on_click)

    pixel_pts, field_pts = [], []

    print("\nClick 4 points on the field (yard-line intersections work best).")
    print("After each click, enter field coordinates in the terminal.")
    print("  x: 0 = own endzone, 120 = opponent endzone")
    print("  y: 0 = bottom sideline, 53.3 = top sideline\n")

    for i in range(4):
        while len(clicks) <= i:
            cv2.imshow("Calibration", display)
            if cv2.waitKey(50) == 27:
                cv2.destroyAllWindows()
                sys.exit("Aborted")

        px = clicks[i]
        pixel_pts.append(px)
        print(f"Point {i + 1} at pixel ({px[0]}, {px[1]})")
        field_x = float(input("  Field x (yards): "))
        field_y = float(input("  Field y (yards): "))
        field_pts.append((field_x, field_y))

    cv2.destroyAllWindows()
    return pixel_pts, field_pts


def main():
    parser = argparse.ArgumentParser(description="Video → coverage prediction")
    parser.add_argument("video", help="Path to game video clip")
    parser.add_argument("--snap-frame", type=int, required=True,
                        help="Frame index of the snap")
    parser.add_argument("--yolo-model",
                        default="Yolo/yolo-training/runs/detect/runs/nfl-player-tracker-2class-2/weights/best.pt")
    parser.add_argument("--nn-model", default="model/coverage_model.pt")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="YOLO confidence threshold")
    args = parser.parse_args()

    # 1. YOLO tracking
    print("Running YOLO tracking...")
    tracker_cfg = os.path.join(os.path.dirname(__file__), "Yolo", "bytetrack.yaml")
    detections = track_video(args.video, args.yolo_model, args.conf, tracker=tracker_cfg)
    detections = stitch_broken_tracks(detections)
    n_frames = len({d["frame"] for d in detections})
    print(f"  {len(detections)} detections, {n_frames} frames")

    # 2. Homography calibration
    pixel_pts, field_pts = calibrate(args.video, args.snap_frame)
    H = compute_homography(pixel_pts, field_pts)

    # 3. Feature extraction (339-dim, matching training data format)
    print("Extracting features...")
    features = process_video_clip(detections, H, args.snap_frame)
    if features is None:
        sys.exit("Could not build feature vector — not enough players detected")
    print(f"  Feature vector: {features.shape}")

    # 4. Load NN and predict
    checkpoint = torch.load(args.nn_model, map_location="cpu", weights_only=False)
    model = NeuralNetwork()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Apply same scaling as training (StandardScaler, preserving zero-padding)
    scaler_mean = checkpoint["scaler_mean"]
    scaler_scale = checkpoint["scaler_scale"]
    mask = features != 0
    scaled = (features - scaler_mean) / scaler_scale
    scaled[~mask] = 0

    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze()

    pred_idx = probs.argmax().item()

    # 5. Results
    print(f"\n{'=' * 40}")
    print(f"  Prediction: {COVERAGE_LABELS[pred_idx]}")
    print(f"  Confidence: {probs[pred_idx].item():.1%}")
    print(f"{'=' * 40}")
    print("\nAll probabilities:")
    for i, label in enumerate(COVERAGE_LABELS):
        bar = "█" * int(probs[i].item() * 30)
        print(f"  {label:>8}: {probs[i].item():6.1%} {bar}")


if __name__ == "__main__":
    main()
