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
  - Snap frame detection (currently must be specified manually)
  - play_direction normalization (currently assumes all plays go right)
"""

import numpy as np
import cv2

# ── Auto calibration ───────────────────────────────────────────────────────

HASH_TOP_Y = 29.72
HASH_BOT_Y = 23.58


def detect_yard_lines(frame):
    """
    Detect yard lines and hash marks from a video frame.
    Returns (yard_line_pixel_xs, hash_top_pixel_y, hash_bot_pixel_y) or None.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 40)
    white = (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 180)

    # ponytail: 60px kernel so dilation reaches the goal line at endzone boundary
    field_mask = cv2.dilate(green.astype(np.uint8), np.ones((60, 60)), iterations=1).astype(bool)
    white_on_field = (white & field_mask).astype(np.uint8) * 255

    # Detect long near-vertical lines (yard lines)
    lines = cv2.HoughLinesP(white_on_field, 1, np.pi / 180, threshold=80,
                            minLineLength=200, maxLineGap=30)
    if lines is None:
        return None

    lines = lines.reshape(-1, 4)
    angles = np.degrees(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0]))
    vert = (np.abs(angles) > 70) & (np.abs(angles) < 110)
    vert_lines = lines[vert]

    if len(vert_lines) < 3:
        return None

    # Group by x-midpoint
    mid_x = (vert_lines[:, 0] + vert_lines[:, 2]) / 2
    sorted_idx = np.argsort(mid_x)

    groups = [[sorted_idx[0]]]
    for i in sorted_idx[1:]:
        if mid_x[i] - mid_x[groups[-1][-1]] < 60:
            groups[-1].append(i)
        else:
            groups.append([i])

    yard_line_xs = []
    for g in groups:
        lengths = np.hypot(vert_lines[g, 2] - vert_lines[g, 0],
                           vert_lines[g, 3] - vert_lines[g, 1])
        if lengths.max() > 150:
            yard_line_xs.append(float(np.mean(mid_x[g])))

    if len(yard_line_xs) < 2:
        return None

    # Filter out yard lines with inconsistent spacing
    spacings = np.diff(yard_line_xs)
    median_spacing = np.median(spacings)
    consistent = [yard_line_xs[0]]
    for i, s in enumerate(spacings):
        if 0.5 * median_spacing < s < 1.5 * median_spacing:
            consistent.append(yard_line_xs[i + 1])
    yard_line_xs = consistent

    # Detect hash marks (short horizontal segments)
    horiz_lines = cv2.HoughLinesP(white_on_field, 1, np.pi / 180, threshold=20,
                                  minLineLength=15, maxLineGap=5)
    if horiz_lines is None:
        return None

    horiz_lines = horiz_lines.reshape(-1, 4)
    h_angles = np.degrees(np.arctan2(horiz_lines[:, 3] - horiz_lines[:, 1],
                                     horiz_lines[:, 2] - horiz_lines[:, 0]))
    h_lengths = np.hypot(horiz_lines[:, 2] - horiz_lines[:, 0],
                         horiz_lines[:, 3] - horiz_lines[:, 1])
    horiz = (np.abs(h_angles) < 15) & (h_lengths < 80) & (h_lengths > 10)
    hash_lines = horiz_lines[horiz]

    if len(hash_lines) < 6:
        return None

    hash_y = (hash_lines[:, 1] + hash_lines[:, 3]) / 2

    # Find two densest y-bands (the two hash mark rows)
    from collections import Counter
    y_rounded = np.round(hash_y / 10) * 10
    counts = Counter(y_rounded.astype(int))
    top_2 = sorted(counts.items(), key=lambda x: -x[1])[:2]
    if len(top_2) < 2:
        return None

    band_ys = sorted([y for y, _ in top_2])
    # Refine: average of all hash marks near each band
    hash_top_y = float(np.mean(hash_y[(hash_y > band_ys[0] - 30) & (hash_y < band_ys[0] + 30)]))
    hash_bot_y = float(np.mean(hash_y[(hash_y > band_ys[1] - 30) & (hash_y < band_ys[1] + 30)]))

    return yard_line_xs, hash_top_y, hash_bot_y


def auto_calibrate(frame, leftmost_yard_line, side="left"):
    """
    Auto-detect yard lines and hash marks, build calibration points.

    Args:
        frame: video frame (BGR)
        leftmost_yard_line: yard line number of the leftmost visible line
        side: "left" = left side of field (yard numbers increase L→R toward midfield)
              "right" = right side of field (yard numbers decrease L→R away from midfield)

    Returns:
        (pixel_points, field_points) or None if detection fails
    """
    result = detect_yard_lines(frame)
    if result is None:
        return None

    yard_line_xs, hash_top_y, hash_bot_y = result
    pixel_pts = []
    field_pts = []

    for i, px_x in enumerate(yard_line_xs):
        if side == "left":
            yard_num = leftmost_yard_line + i * 5
            if yard_num > 50:
                continue
            field_x = 10.0 + yard_num
        else:
            yard_num = leftmost_yard_line - i * 5
            if yard_num < 0:
                continue
            field_x = 110.0 - yard_num

        pixel_pts.append((px_x, hash_top_y))
        field_pts.append((field_x, HASH_TOP_Y))
        pixel_pts.append((px_x, hash_bot_y))
        field_pts.append((field_x, HASH_BOT_Y))

    if len(pixel_pts) < 4:
        return None

    return pixel_pts, field_pts


# ── Homography ──────────────────────────────────────────────────────────────

def compute_homography(pixel_points, field_points):
    """
    Compute transform from pixel coords to field coords.
    3 points → affine transform, 4+ → full homography.
    """
    src = np.array(pixel_points, dtype=np.float32)
    dst = np.array(field_points, dtype=np.float32)
    if len(src) == 3:
        return cv2.getAffineTransform(src, dst)
    H, _ = cv2.findHomography(src, dst)
    return H


def transform_points(H, pixel_points):
    """Apply homography or affine transform to convert pixel coords → field coords."""
    pts = np.array(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
    if H.shape == (2, 3):
        transformed = cv2.transform(pts, H)
    else:
        transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2)


# ── Offense / Defense split ─────────────────────────────────────────────────

def _dominant_jersey_color(frame, cx, cy, w, h):
    """Extract dominant non-green color from a player's bounding box."""
    fh, fw = frame.shape[:2]
    x1 = max(0, int(cx - w / 2))
    y1 = max(0, int(cy - h / 2))
    x2 = min(fw, int(cx + w / 2))
    y2 = min(fh, int(cy + h / 2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.array([128, 128, 128], dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Mask out green field pixels (hue ~35-85, high saturation)
    green_mask = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) & (hsv[:, :, 1] > 40)
    non_green = crop[~green_mask]
    if len(non_green) < 5:
        return np.array([128, 128, 128], dtype=np.float32)
    return np.median(non_green, axis=0).astype(np.float32)


def cluster_teams_by_color(frame, detections):
    """
    K-means on jersey colors to split detections into two teams.
    Returns list of cluster labels (0 or 1) matching detection order.
    """
    colors = []
    for d in detections:
        c = _dominant_jersey_color(frame, d["cx"], d["cy"], d["w"], d["h"])
        colors.append(c)
    colors = np.array(colors, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(colors, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    return labels.flatten().tolist()


def split_offense_defense_by_color(players, labels, los_x):
    """
    Split into offense/defense using jersey color clusters.
    Whichever cluster has lower average x (behind LOS) = offense.
    """
    cluster_0 = [p for p, l in zip(players, labels) if l == 0]
    cluster_1 = [p for p, l in zip(players, labels) if l == 1]

    avg_x_0 = np.mean([p["field_x"] for p in cluster_0]) if cluster_0 else 999
    avg_x_1 = np.mean([p["field_x"] for p in cluster_1]) if cluster_1 else 999

    if avg_x_0 < avg_x_1:
        return cluster_0, cluster_1
    return cluster_1, cluster_0


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

    total = len(defense_snap) + len(offense_snap)
    if total < 14:
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

def process_video_clip(detections, H, snap_frame, video_path, n_post_frames=10, frame_step=2):
    """
    Full pipeline: raw YOLO detections → 339-dim feature vector.

    Args:
        detections: list of dicts from track.py (frame, track_id, class, cx, cy, w, h)
        H: homography matrix from compute_homography()
        snap_frame: frame index of the snap
        video_path: path to video file (for jersey color extraction)
        n_post_frames: how many post-snap samples (default 10 = 2 seconds at 10fps equiv)
        frame_step: take every Nth frame for post-snap (default 2 for ~10fps from 20fps video)

    Returns:
        np.array of shape (339,) or None
    """
    # Read snap frame for jersey color clustering
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, snap_frame)
    ret, snap_image = cap.read()
    cap.release()

    # Filter out sideline detections
    field_dets = [d for d in detections if d["class"] == "Helmet"]

    # Frames we need: snap + post-snap
    needed_frames = [snap_frame]
    for i in range(1, n_post_frames + 1):
        needed_frames.append(snap_frame + i * frame_step)

    # Group detections by frame, transform to field coords
    frames_data = {}
    snap_players = None

    for local_idx, abs_frame in enumerate(needed_frames):
        frame_dets = [d for d in field_dets if d["frame"] == abs_frame]
        if not frame_dets:
            continue

        pixel_pts = [(d["cx"], d["cy"]) for d in frame_dets]
        field_pts = transform_points(H, pixel_pts)

        players = []
        for d, (fx, fy) in zip(frame_dets, field_pts):
            players.append({
                "field_x": float(fx),
                "field_y": float(fy),
            })

        if local_idx == 0:
            los = estimate_los(players)
            snap_dets = [d for d in field_dets if d["frame"] == abs_frame]
            if ret and snap_image is not None:
                labels = cluster_teams_by_color(snap_image, snap_dets)
                offense, defense = split_offense_defense_by_color(players, labels, los)
            else:
                by_x = sorted(players, key=lambda p: p["field_x"])
                half = len(players) // 2
                offense, defense = by_x[:half], by_x[half:]
            for p in offense:
                p["side"] = "O"
            for p in defense:
                p["side"] = "D"
            # Assign stable IDs based on snap-frame order
            for i, p in enumerate(players):
                p["track_id"] = i
            snap_players = players
        else:
            # Match to snap-frame players by nearest position instead of track ID
            # ponytail: O(n²) greedy matching, Hungarian if accuracy matters
            snap_pts = np.array([[p["field_x"], p["field_y"]] for p in snap_players])
            used = set()
            for p in players:
                dists = np.sqrt((snap_pts[:, 0] - p["field_x"])**2 +
                                (snap_pts[:, 1] - p["field_y"])**2)
                for idx in np.argsort(dists):
                    if idx not in used:
                        used.add(idx)
                        p["track_id"] = snap_players[idx]["track_id"]
                        p["side"] = snap_players[idx]["side"]
                        break

        frames_data[local_idx] = [p for p in players if "track_id" in p]

    return build_feature_vector_from_frames(frames_data)
