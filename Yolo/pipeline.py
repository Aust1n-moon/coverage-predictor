"""
Video → field coordinates → feature vector pipeline.
All in-memory — no intermediate files. Built for HF Spaces backend.

CONTEXT (Sept 2026):
  - YOLO model has 2 classes: Helmet (0), Helmet-Sideline (1)
  - Helmet-Sideline detections are filtered out (not on field)
  - Videos are a mix of All22 top-down and lower-angle CFB broadcast films
  - Camera is mostly static per clip, so one homography per video works
  - Feature vector shape (339) matches processing.ipynb / model trained on NFL tracking data

PIPELINE:
  1. Homography: user marks 4+ yard-line points on video frame → cv2.findHomography()
     maps pixel coords to field yards (0-120 x 0-53.3)
  2. YOLO tracking (track.py) → pixel positions → transformed to field positions
  3. Offense/defense split: largest-gap heuristic at scrimmage line
     ponytail: won't work for goal-line or exotic formations — upgrade to jersey color
     clustering if that matters
  4. Feature vector: 8 defenders × 25 + 6 offense × 22 + 7 formation = 339 dims
     Must match processing.ipynb exactly or model outputs garbage
  5. Feed into TF/Keras MLP (or ONNX export) → coverage type prediction

TODO:
  - Auto yard-line detection (Hough lines on green/white) to replace manual 4-click
  - Snap frame detection (currently must be specified manually)
  - play_direction normalization (currently assumes all plays go right)
"""

import numpy as np
import cv2

# ── Homography ──────────────────────────────────────────────────────────────

def compute_homography(pixel_points, field_points):
    """
    Compute perspective transform from pixel coords to field coords.

    Args:
        pixel_points: list of 4+ (px_x, px_y) from the video frame
        field_points: list of 4+ (field_x, field_y) in yards (0-120, 0-53.3)

    Returns:
        3x3 homography matrix
    """
    src = np.array(pixel_points, dtype=np.float32)
    dst = np.array(field_points, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def transform_points(H, pixel_points):
    """Apply homography to convert pixel coords → field coords."""
    pts = np.array(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2)


# ── Offense / Defense split ─────────────────────────────────────────────────

def split_offense_defense(players, los_x):
    """
    Split players into offense/defense by scrimmage line.
    Players behind LOS (lower x if play goes right) = offense.
    Players beyond LOS = defense.

    Args:
        players: list of dicts with 'field_x', 'field_y', 'track_id'
        los_x: line of scrimmage x-coordinate in field yards

    Returns:
        (offense_list, defense_list)
    """
    offense, defense = [], []
    for p in players:
        if p["field_x"] < los_x:
            offense.append(p)
        else:
            defense.append(p)

    # Sanity: offense should have ~11, defense ~11
    # If split is too uneven, LOS estimate is probably off.
    # ponytail: naive x-threshold split. Improve with jersey color clustering if needed.
    return offense, defense


def estimate_los(players):
    """
    Estimate line of scrimmage as the x-position where players cluster most
    densely (the line where both teams meet).
    """
    xs = sorted(p["field_x"] for p in players)
    if len(xs) < 10:
        return np.median(xs)
    # Find the biggest gap between consecutive sorted x-values in the middle
    # — that's likely the scrimmage line
    mid_xs = xs[3:-3]  # trim outliers (deep safeties, QBs)
    gaps = [(mid_xs[i+1] - mid_xs[i], (mid_xs[i] + mid_xs[i+1]) / 2)
            for i in range(len(mid_xs) - 1)]
    # ponytail: largest-gap heuristic. Works for standard formations, not goal-line.
    _, los = max(gaps, key=lambda g: g[0])
    return los


# ── Feature vector ──────────────────────────────────────────────────────────

N_DEF = 8
N_OFF = 6
FIELD_CENTER_Y = 26.65


def build_feature_vector_from_frames(frames_data):
    """
    Build the 339-dim feature vector from multi-frame tracking data.

    Args:
        frames_data: dict mapping frame_index → list of player dicts
            each with 'track_id', 'field_x', 'field_y', 'side' ('O'/'D')
            Frame 0 = snap, frames 1-10 = post-snap (every other frame for 10fps)

    Returns:
        np.array of shape (339,) or None if not enough players
    """
    snap_frame = frames_data.get(0)
    if snap_frame is None:
        return None

    defense_snap = sorted([p for p in snap_frame if p["side"] == "D"],
                          key=lambda p: p["field_y"])
    offense_snap = sorted([p for p in snap_frame if p["side"] == "O"],
                          key=lambda p: p["field_y"])

    if len(defense_snap) < 4 or len(offense_snap) < 3:
        return None

    los_x = np.mean([p["field_x"] for p in offense_snap])
    post_frame_indices = list(range(1, 11))  # frames 1-10

    # Build displacement lookup: track_id → frame_idx → (x, y)
    positions = {}
    for fidx, players in frames_data.items():
        for p in players:
            positions.setdefault(p["track_id"], {})[fidx] = (p["field_x"], p["field_y"])

    # ── Per-defender features ──
    def_xy = np.array([[p["field_x"], p["field_y"]] for p in defense_snap])
    off_xy = np.array([[p["field_x"], p["field_y"]] for p in offense_snap])
    def_depths = def_xy[:, 0] - los_x

    def_feats = []
    nearest_dists = []
    for idx, p in enumerate(defense_snap):
        dists = np.sqrt((off_xy[:, 0] - p["field_x"])**2 +
                        (off_xy[:, 1] - p["field_y"])**2)
        nearest_dist = dists.min()
        nearest_dists.append(nearest_dist)
        depth = def_depths[idx]
        dist_center = abs(p["field_y"] - FIELD_CENTER_Y)

        feat = [p["field_x"], p["field_y"], depth, nearest_dist, dist_center]
        # Displacements from snap for each post-snap frame
        snap_pos = positions.get(p["track_id"], {}).get(0, (p["field_x"], p["field_y"]))
        for fidx in post_frame_indices:
            pos = positions.get(p["track_id"], {}).get(fidx, snap_pos)
            feat.append(pos[0] - snap_pos[0])  # dx
            feat.append(pos[1] - snap_pos[1])  # dy
        def_feats.append(feat)

    # ── Per-offense features ──
    off_feats = []
    for p in offense_snap:
        feat = [p["field_x"], p["field_y"]]
        snap_pos = positions.get(p["track_id"], {}).get(0, (p["field_x"], p["field_y"]))
        for fidx in post_frame_indices:
            pos = positions.get(p["track_id"], {}).get(fidx, snap_pos)
            feat.append(pos[0] - snap_pos[0])
            feat.append(pos[1] - snap_pos[1])
        off_feats.append(feat)

    # ── Formation-level features ──
    n_deep = int((def_depths > 10).sum())
    n_box = int((def_depths < 5).sum())
    safety_high = int((def_depths > 12).sum())
    avg_depth = float(def_depths.mean())
    def_spread = float(def_xy[:, 1].std())
    max_depth = float(def_depths.max())
    sorted_by_width = np.argsort(np.abs(def_xy[:, 1] - FIELD_CENTER_Y))[::-1]
    corner_dists = [nearest_dists[i] for i in sorted_by_width[:2]]
    avg_corner_press = float(np.mean(corner_dists))

    formation = [n_deep, n_box, safety_high, avg_depth, def_spread,
                 avg_corner_press, max_depth]

    # ── Pad and combine ──
    n_disp = len(post_frame_indices) * 2  # 20
    def_fpp = 5 + n_disp  # 25
    off_fpp = 2 + n_disp  # 22

    def pad(feats, n, fpp):
        arr = np.array(feats[:n])
        if len(arr) < n:
            arr = np.vstack([arr, np.zeros((n - len(arr), fpp))])
        return arr.flatten()

    return np.concatenate([
        pad(def_feats, N_DEF, def_fpp),
        pad(off_feats, N_OFF, off_fpp),
        np.array(formation, dtype=np.float32)
    ])  # shape: (339,)


# ── Full pipeline ───────────────────────────────────────────────────────────

def process_video_clip(detections, H, snap_frame, n_post_frames=10, frame_step=2):
    """
    Full pipeline: raw YOLO detections → 339-dim feature vector.

    Args:
        detections: list of dicts from track.py (frame, track_id, class, cx, cy)
        H: homography matrix from compute_homography()
        snap_frame: frame index of the snap
        n_post_frames: how many post-snap samples (default 10 = 2 seconds at 10fps equiv)
        frame_step: take every Nth frame for post-snap (default 2 for ~10fps from 20fps video)

    Returns:
        np.array of shape (339,) or None
    """
    # Filter out sideline detections
    field_dets = [d for d in detections if d["class"] == "Helmet"]

    # Frames we need: snap + post-snap
    needed_frames = [snap_frame]
    for i in range(1, n_post_frames + 1):
        needed_frames.append(snap_frame + i * frame_step)

    # Group detections by frame, transform to field coords
    frames_data = {}
    for local_idx, abs_frame in enumerate(needed_frames):
        frame_dets = [d for d in field_dets if d["frame"] == abs_frame]
        if not frame_dets:
            continue

        pixel_pts = [(d["cx"], d["cy"]) for d in frame_dets]
        field_pts = transform_points(H, pixel_pts)

        players = []
        for d, (fx, fy) in zip(frame_dets, field_pts):
            players.append({
                "track_id": d["track_id"],
                "field_x": float(fx),
                "field_y": float(fy),
            })

        # Split offense/defense on snap frame
        if local_idx == 0:
            los = estimate_los(players)
            offense, _ = split_offense_defense(players, los)
            off_ids = {p["track_id"] for p in offense}
            for p in players:
                p["side"] = "O" if p["track_id"] in off_ids else "D"
        else:
            for p in players:
                p["side"] = "O" if p["track_id"] in off_ids else "D"

        frames_data[local_idx] = players

    return build_feature_vector_from_frames(frames_data)
