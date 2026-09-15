"""
Visual test: load the helmet model on a video and display detections.

Usage:
    python test-vision.py <video_path>
"""

import glob
import random
import sys
from ultralytics import YOLO

MODEL_PATH = "yolo-training/runs/detect/runs/nfl-player-tracker-2class-2/weights/best.pt"
VIDEO_DIR = "training-videos"

if __name__ == "__main__":
    videos = glob.glob(f"{VIDEO_DIR}/*.mp4") + glob.glob(f"{VIDEO_DIR}/*.avi") + glob.glob(f"{VIDEO_DIR}/*.mov")
    if not videos:
        print(f"No videos found in {VIDEO_DIR}/")
        sys.exit(1)

    video = random.choice(videos)
    print(f"Testing on: {video}")

    model = YOLO(MODEL_PATH)
    model.predict(source=video, show=True, conf=0.5, save=False)
