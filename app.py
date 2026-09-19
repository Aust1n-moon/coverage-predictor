"""
Gradio app — upload a game clip, get coverage prediction.
Run: python app.py
"""

import os
import sys

import cv2
import torch
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Yolo'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

from track import track_video, stitch_broken_tracks
from pipeline import (compute_homography, process_video_clip,
                      auto_calibrate, detect_snap_frame)
from model import NeuralNetwork

COVERAGE_LABELS = ['2_MAN', 'COVER_0', 'COVER_1', 'COVER_2', 'COVER_3', 'COVER_4']

YOLO_MODEL = os.path.join(os.path.dirname(__file__),
    "Yolo/yolo-training/runs/detect/runs/nfl-player-tracker-2class-2/weights/best.pt")
NN_MODEL = os.path.join(os.path.dirname(__file__), "model/coverage_model.pt")
TRACKER_CFG = os.path.join(os.path.dirname(__file__), "Yolo/bytetrack.yaml")

checkpoint = torch.load(NN_MODEL, map_location="cpu", weights_only=False)
nn_model = NeuralNetwork()
nn_model.load_state_dict(checkpoint["model_state_dict"])
nn_model.eval()
scaler_mean = checkpoint["scaler_mean"]
scaler_scale = checkpoint["scaler_scale"]


def predict_coverage(video_path, yard_line, side, play_direction, yard_interval,
                     snap_frame_input, conf_threshold, progress=gr.Progress()):
    if video_path is None:
        raise gr.Error("Upload a video clip first.")

    yard_line = int(yard_line)
    yard_interval = int(yard_interval)

    progress(0.1, desc="Running YOLO tracking...")
    detections = track_video(video_path, YOLO_MODEL, conf_threshold, tracker=TRACKER_CFG)
    detections = stitch_broken_tracks(detections)

    progress(0.4, desc="Detecting snap frame...")
    if snap_frame_input is None or snap_frame_input < 0:
        snap_frame = detect_snap_frame(detections)
    else:
        snap_frame = int(snap_frame_input)

    progress(0.5, desc="Calibrating field...")
    result = None
    for offset in [0, -10, -20, -50, -100]:
        cal_frame = max(0, snap_frame + offset)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cal_frame)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue
        result = auto_calibrate(frame, yard_line, side, yard_interval)
        if result is not None:
            break

    if result is None:
        raise gr.Error("Auto-calibration failed — not enough yard lines or hash marks visible.")

    pixel_pts, field_pts = result
    H = compute_homography(pixel_pts, field_pts)

    progress(0.7, desc="Extracting features...")
    features = process_video_clip(detections, H, snap_frame, video_path,
                                  play_direction=play_direction)
    if features is None:
        raise gr.Error("Not enough players detected to build feature vector.")

    progress(0.9, desc="Predicting...")
    mask = features != 0
    scaled = (features - scaler_mean) / scaler_scale
    scaled[~mask] = 0
    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(nn_model(x), dim=1).squeeze().numpy()

    pred_idx = int(probs.argmax())
    confidences = {label: float(p) for label, p in zip(COVERAGE_LABELS, probs)}

    summary = f"**{COVERAGE_LABELS[pred_idx]}** — {probs[pred_idx]:.1%} confidence\n\nSnap frame: {snap_frame}"
    return confidences, summary


with gr.Blocks(title="Coverage Predictor") as demo:
    gr.Markdown("# NFL Coverage Predictor\nUpload a game video clip to predict the defensive coverage type.")

    with gr.Row():
        with gr.Column(scale=2):
            video_input = gr.Video(label="Game Clip")
            with gr.Row():
                yard_line = gr.Number(label="Yard Line", value=40,
                                     info="Leftmost visible yard line (0–50)")
                side = gr.Dropdown(["left", "right"], label="Side", value="left",
                                  info="Which side of midfield")
            with gr.Row():
                play_dir = gr.Dropdown(["left", "right"], label="Play Direction", value="right",
                                      info="Direction offense is moving")
                yard_int = gr.Dropdown([5, 10], label="Yard Interval", value=5, type="value",
                                      info="5 for All-22, 10 for broadcast")
            with gr.Row():
                snap_frame = gr.Number(label="Snap Frame", value=-1,
                                      info="-1 for auto-detection")
                conf = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="YOLO Confidence")
            run_btn = gr.Button("Predict Coverage", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(label="Coverage Probabilities", num_top_classes=6)
            status_output = gr.Markdown()

    run_btn.click(
        fn=predict_coverage,
        inputs=[video_input, yard_line, side, play_dir, yard_int, snap_frame, conf],
        outputs=[label_output, status_output],
    )

if __name__ == "__main__":
    demo.launch()
