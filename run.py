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
from pipeline import compute_homography, process_video_clip, auto_calibrate, detect_snap_frame
from model import NeuralNetwork

COVERAGE_LABELS = ['2_MAN', 'COVER_0', 'COVER_1', 'COVER_2', 'COVER_3', 'COVER_4']


def main():
    parser = argparse.ArgumentParser(description="Video → coverage prediction")
    parser.add_argument("video", help="Path to game video clip")
    parser.add_argument("--snap-frame", type=int, default=None,
                        help="Frame index of the snap (auto-detected if omitted)")
    parser.add_argument("--yolo-model",
                        default="Yolo/yolo-training/runs/detect/runs/nfl-player-tracker-2class-2/weights/best.pt")
    parser.add_argument("--nn-model", default="model/coverage_model.pt")
    parser.add_argument("--yard-line", type=int, required=True,
                        help="Leftmost visible yard line number (e.g. 30, 40)")
    parser.add_argument("--side", choices=["left", "right"], required=True,
                        help="Which side of the field the play is on (left or right of midfield)")
    parser.add_argument("--play-direction", choices=["left", "right"], default="right",
                        help="Direction offense is going (left or right on screen)")
    parser.add_argument("--yard-interval", type=int, default=5, choices=[5, 10],
                        help="Yards between detected lines (5 for All22, 10 for broadcast)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="YOLO confidence threshold")
    parser.add_argument("--debug", action="store_true", help="Print O/D clustering details")
    args = parser.parse_args()

    # 1. YOLO tracking
    print("Running YOLO tracking...")
    tracker_cfg = os.path.join(os.path.dirname(__file__), "Yolo", "bytetrack.yaml")
    detections = track_video(args.video, args.yolo_model, args.conf, tracker=tracker_cfg)
    print(f"  Tracking done. {len(detections)} raw detections")
    n_tracks = len({d["track_id"] for d in detections})
    print(f"  Stitching {n_tracks} tracks...")
    detections = stitch_broken_tracks(detections)
    n_frames = len({d["frame"] for d in detections})
    print(f"  {len(detections)} detections, {n_frames} frames")

    # 1b. Snap detection
    snap_frame = args.snap_frame
    if snap_frame is None:
        print("Auto-detecting snap frame...")
        snap_frame = detect_snap_frame(detections)
        print(f"  Snap detected at frame {snap_frame}")
    else:
        print(f"  Using manual snap frame: {snap_frame}")

    # 2. Homography calibration — try snap frame first, fall back to earlier frames
    print("Detecting yard lines and hash marks...")
    result = None
    cal_frame_idx = snap_frame
    for offset in [0, -10, -20, -50, -100, -snap_frame]:
        cal_frame_idx = max(0, snap_frame + offset)
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cal_frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue
        result = auto_calibrate(frame, args.yard_line, args.side, args.yard_interval)
        if result is not None:
            if offset != 0:
                print(f"  Calibrated from frame {cal_frame_idx} (snap frame too zoomed)")
            break
    if result is None:
        sys.exit("Auto-calibration failed — could not detect enough yard lines or hash marks")

    pixel_pts, field_pts = result
    print(f"  Auto-detected {len(pixel_pts)} calibration points")
    print(f"  Pixel points: {pixel_pts}")
    print(f"  Field points: {field_pts}")
    H = compute_homography(pixel_pts, field_pts)
    print(f"  H: {H}")

    # 3. Feature extraction (339-dim, matching training data format)
    print("Extracting features...")
    features = process_video_clip(detections, H, snap_frame, args.video,
                                   play_direction=args.play_direction, debug=args.debug)
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
