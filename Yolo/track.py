"""
Run YOLO tracking on game video, fix broken track IDs, export to CSV.

Usage:
    python track.py <video_path> [--model path/to/best.pt] [--out output.csv]
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
from ultralytics import YOLO

CLASS_NAMES = {0: "Helmet", 1: "Helmet-Sideline"}


def stitch_broken_tracks(detections, max_gap=15, max_dist=50):
    """
    Fix IDs that break during occlusion. If a track disappears and a new one
    appears nearby within `max_gap` frames, merge them under the original ID.

    max_gap: max frames between last seen and first reappear
    max_dist: max pixel distance to consider same player
    """
    # Build per-track first/last appearance
    tracks = defaultdict(lambda: {"first_frame": float("inf"), "last_frame": -1,
                                   "first_pos": None, "last_pos": None})
    for d in detections:
        t = tracks[d["track_id"]]
        if d["frame"] < t["first_frame"]:
            t["first_frame"] = d["frame"]
            t["first_pos"] = (d["cx"], d["cy"])
        if d["frame"] > t["last_frame"]:
            t["last_frame"] = d["frame"]
            t["last_pos"] = (d["cx"], d["cy"])

    # Match dead tracks to new tracks
    id_map = {}
    track_ids = sorted(tracks.keys(), key=lambda tid: tracks[tid]["first_frame"])

    for i, new_id in enumerate(track_ids):
        new = tracks[new_id]
        best_match, best_dist = None, max_dist
        for old_id in track_ids[:i]:
            old = tracks[id_map.get(old_id, old_id)]
            gap = new["first_frame"] - old["last_frame"]
            if 1 <= gap <= max_gap:
                dist = np.hypot(new["first_pos"][0] - old["last_pos"][0],
                                new["first_pos"][1] - old["last_pos"][1])
                if dist < best_dist:
                    best_dist = dist
                    best_match = id_map.get(old_id, old_id)
        if best_match is not None:
            id_map[new_id] = best_match

    # Apply remapping
    for d in detections:
        d["track_id"] = id_map.get(d["track_id"], d["track_id"])

    return detections


def track_video(video_path, model_path, conf=0.5, tracker="bytetrack.yaml"):
    """Run YOLO tracking, return list of per-frame detections."""
    model = YOLO(model_path)
    results = model.track(
        source=video_path,
        conf=conf,
        tracker=tracker,
        stream=True,
        show=False,
        verbose=True,
    )

    detections = []
    for frame_idx, r in enumerate(results):
        if r.boxes.id is None:
            continue
        ids = r.boxes.id.cpu().numpy().astype(int)
        classes = r.boxes.cls.cpu().numpy().astype(int)
        # xyxy → center x, center y, width, height
        xyxy = r.boxes.xyxy.cpu().numpy()
        for track_id, cls, box in zip(ids, classes, xyxy):
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            w = box[2] - box[0]
            h = box[3] - box[1]
            detections.append({
                "frame": frame_idx,
                "track_id": int(track_id),
                "class": CLASS_NAMES.get(cls, str(cls)),
                "cx": float(cx),
                "cy": float(cy),
                "w": float(w),
                "h": float(h),
            })

    return detections


def save_csv(detections, out_path):
    fields = ["frame", "track_id", "class", "cx", "cy", "w", "h"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(detections)
    print(f"Saved {len(detections)} detections to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO helmet tracking → CSV")
    parser.add_argument("video", help="Path to game video")
    parser.add_argument("--model", default="yolo-training/runs/detect/runs/nfl-player-tracker-2class-2/weights/best.pt")
    parser.add_argument("--out", default="tracking_output.csv")
    parser.add_argument("--conf", type=float, default=0.5)
    # ponytail: max_dist=50px assumes ~1080p broadcast. Tune if resolution differs.
    parser.add_argument("--max-gap", type=int, default=15, help="Max frames to stitch broken tracks")
    parser.add_argument("--max-dist", type=float, default=50, help="Max pixel distance for stitching")
    args = parser.parse_args()

    dets = track_video(args.video, args.model, args.conf)
    dets = stitch_broken_tracks(dets, args.max_gap, args.max_dist)
    save_csv(dets, args.out)
