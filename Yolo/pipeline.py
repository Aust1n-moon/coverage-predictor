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


def _detect_yard_line_groups(frame):
    """
    Core yard line detection: returns (groups_info, white_on_field, dom_abs_angle)
    where groups_info is list of (mean_x, min_y, max_y) per yard line group,
    or None if detection fails.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 40)
    white = (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 180)

    field_mask = cv2.dilate(green.astype(np.uint8), np.ones((60, 60)), iterations=1).astype(bool)
    white_on_field = (white & field_mask).astype(np.uint8) * 255

    lines = cv2.HoughLinesP(white_on_field, 1, np.pi / 180, threshold=80,
                            minLineLength=200, maxLineGap=30)
    if lines is None:
        return None

    lines = lines.reshape(-1, 4)
    angles = np.degrees(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0]))
    # Try near-vertical first, but fall back to dominant diagonal angle
    # if near-vertical doesn't yield enough lines (broadcast views have
    # a few incidental near-vertical segments that aren't yard lines)
    vert_mask = (np.abs(angles) > 70) & (np.abs(angles) < 110)
    steep = (np.abs(angles) > 45) & (np.abs(angles) < 135)

    dom_abs_angle = 90
    vert_lines = lines[vert_mask] if vert_mask.sum() >= 10 else np.empty((0, 4))

    if len(vert_lines) < 3 and steep.sum() >= 3:
        from collections import Counter
        abs_a = np.abs(angles[steep])
        bins = np.round(abs_a / 5) * 5
        dom_abs_angle = Counter(bins.astype(int)).most_common(1)[0][0]
        diag = steep & (np.abs(np.abs(angles) - dom_abs_angle) < 8)
        vert_lines = lines[diag]

    if len(vert_lines) < 3:
        return None

    mid_x = (vert_lines[:, 0] + vert_lines[:, 2]) / 2
    if dom_abs_angle < 75:
        mid_y = (vert_lines[:, 1] + vert_lines[:, 3]) / 2
        dom_rad = np.radians(dom_abs_angle)
        proj = mid_x * np.sin(dom_rad) - mid_y * np.cos(dom_rad)
    else:
        proj = mid_x
    sorted_idx = np.argsort(proj)

    groups = [[sorted_idx[0]]]
    for i in sorted_idx[1:]:
        if proj[i] - proj[groups[-1][-1]] < 60:
            groups[-1].append(i)
        else:
            groups.append([i])

    # For each group: mean x-midpoint, topmost y, bottommost y
    groups_info = []
    for g in groups:
        lengths = np.hypot(vert_lines[g, 2] - vert_lines[g, 0],
                           vert_lines[g, 3] - vert_lines[g, 1])
        if lengths.max() > 150:
            all_ys = np.concatenate([vert_lines[g, 1], vert_lines[g, 3]])
            groups_info.append((
                float(np.mean(mid_x[g])),
                float(all_ys.min()),
                float(all_ys.max()),
            ))
    groups_info.sort(key=lambda g: g[0])

    if len(groups_info) < 2:
        return None

    # Filter by consistent spacing
    xs = [g[0] for g in groups_info]
    spacings = np.diff(xs)
    median_spacing = np.median(spacings)
    keep = [0]
    for i in range(1, len(xs)):
        gap = xs[i] - xs[keep[-1]]
        if 0.5 * median_spacing < gap < 1.5 * median_spacing:
            keep.append(i)
    groups_info = [groups_info[i] for i in keep]

    if len(groups_info) < 2:
        return None

    return groups_info, white_on_field, dom_abs_angle


def detect_yard_lines(frame):
    """
    Detect yard lines and hash marks from a video frame.
    Returns (yard_line_pixel_xs, hash_top_pixel_y, hash_bot_pixel_y) or None.

    Tries hash mark detection first. If that fails, uses yard line endpoints
    as proxy for hash positions (works for broadcast angles).
    """
    result = _detect_yard_line_groups(frame)
    if result is None:
        return None
    groups_info, white_on_field, dom_abs_angle = result
    yard_line_xs = [g[0] for g in groups_info]

    # Try hash mark detection
    hash_target_angle = dom_abs_angle - 90
    horiz_lines = cv2.HoughLinesP(white_on_field, 1, np.pi / 180, threshold=20,
                                  minLineLength=15, maxLineGap=5)
    hash_detected = False
    if horiz_lines is not None:
        horiz_lines = horiz_lines.reshape(-1, 4)
        h_angles = np.degrees(np.arctan2(horiz_lines[:, 3] - horiz_lines[:, 1],
                                         horiz_lines[:, 2] - horiz_lines[:, 0]))
        h_lengths = np.hypot(horiz_lines[:, 2] - horiz_lines[:, 0],
                             horiz_lines[:, 3] - horiz_lines[:, 1])
        angle_diff = np.minimum(np.abs(h_angles - hash_target_angle),
                                np.abs(h_angles + hash_target_angle))
        horiz = (angle_diff < 15) & (h_lengths < 80) & (h_lengths > 10)
        hash_lines = horiz_lines[horiz]

        if len(hash_lines) >= 6:
            hash_mid_x = (hash_lines[:, 0] + hash_lines[:, 2]) / 2
            hash_y = (hash_lines[:, 1] + hash_lines[:, 3]) / 2
            h = frame.shape[0]
            margin = h * 0.08
            in_field = (hash_y > margin) & (hash_y < h - margin)
            yl_arr = np.array(yard_line_xs)
            near_tol = max(40, frame.shape[1] * 0.02)
            near_yl = np.min(np.abs(hash_mid_x[:, None] - yl_arr[None, :]), axis=1) < near_tol
            keep = in_field & near_yl
            hash_y = hash_y[keep]

            if len(hash_y) >= 4:
                from collections import Counter
                h = frame.shape[0]
                bin_size = max(5, int(h / 200))
                y_rounded = np.round(hash_y / bin_size) * bin_size
                counts = Counter(y_rounded.astype(int))
                min_sep = h * 0.04
                sorted_bands = sorted(counts.items(), key=lambda x: -x[1])
                top_2 = None
                for i in range(len(sorted_bands)):
                    for j in range(i + 1, len(sorted_bands)):
                        if abs(sorted_bands[i][0] - sorted_bands[j][0]) > min_sep:
                            top_2 = [sorted_bands[i], sorted_bands[j]]
                            break
                    if top_2:
                        break
                if top_2 is not None:
                    band_ys = sorted([y for y, _ in top_2])
                    refine = bin_size * 3
                    hash_top_y = float(np.mean(hash_y[(hash_y > band_ys[0] - refine) & (hash_y < band_ys[0] + refine)]))
                    hash_bot_y = float(np.mean(hash_y[(hash_y > band_ys[1] - refine) & (hash_y < band_ys[1] + refine)]))
                    hash_detected = True

    if hash_detected:
        return yard_line_xs, hash_top_y, hash_bot_y

    # Fallback: use yard line top/bottom endpoints as hash proxies
    top_ys = [g[1] for g in groups_info]
    bot_ys = [g[2] for g in groups_info]
    hash_top_y = float(np.median(top_ys))
    hash_bot_y = float(np.median(bot_ys))
    if abs(hash_top_y - hash_bot_y) < frame.shape[0] * 0.03:
        return None
    return yard_line_xs, hash_top_y, hash_bot_y


def auto_calibrate(frame, leftmost_yard_line, side="left", yard_interval=5):
    """
    Auto-detect yard lines and hash marks, build calibration points.

    Args:
        frame: video frame (BGR)
        leftmost_yard_line: yard line number of the leftmost visible line
        side: "left" = left side of field (yard numbers increase L→R toward midfield)
              "right" = right side of field (yard numbers decrease L→R away from midfield)
        yard_interval: yards between detected lines (5 for All22, 10 for broadcast)

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
            yard_num = leftmost_yard_line + i * yard_interval
            if yard_num > 50:
                continue
            field_x = 10.0 + yard_num
        else:
            yard_num = leftmost_yard_line - i * yard_interval
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
    """Extract dominant non-green jersey color from below the helmet bbox."""
    fh, fw = frame.shape[:2]
    # Helmet bbox is the detection — jersey is below it
    helmet_bot = int(cy + h / 2)
    jersey_top = helmet_bot
    jersey_bot = min(fh, helmet_bot + int(h * 2.5))
    x1 = max(0, int(cx - w))
    x2 = min(fw, int(cx + w))
    crop = frame[jersey_top:jersey_bot, x1:x2]
    if crop.size == 0:
        return np.array([128, 128, 128], dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green_mask = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) & (hsv[:, :, 1] > 40)
    non_green = crop[~green_mask]
    if len(non_green) < 5:
        return np.array([128, 128, 128], dtype=np.float32)
    # LAB for perceptual color distance
    non_green_lab = cv2.cvtColor(non_green.reshape(1, -1, 3), cv2.COLOR_BGR2LAB)
    return np.median(non_green_lab.reshape(-1, 3), axis=0).astype(np.float32)


def cluster_teams_by_color(frame, detections, debug=False):
    """
    K-means (K=2) on jersey colors to split detections into two teams.
    Returns list of cluster labels (0 or 1) matching detection order.
    """
    colors = []
    for d in detections:
        c = _dominant_jersey_color(frame, d["cx"], d["cy"], d["w"], d["h"])
        colors.append(c)
    colors = np.array(colors, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(colors, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten().tolist()
    if debug:
        c0 = sum(1 for l in labels if l == 0)
        c1 = sum(1 for l in labels if l == 1)
        print(f"  Color clusters: {c0} vs {c1}")
        print(f"  Centers (LAB): {centers}")
    return labels


def split_offense_defense_by_color(players, labels, los_x):
    """
    Split into offense/defense using jersey color clusters.
    Label -1 = dropped (refs/noise). Whichever surviving cluster has
    lower average x (behind LOS) = offense.
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

def process_video_clip(detections, H, snap_frame, video_path, n_post_frames=10, frame_step=2, play_direction="right", debug=False):
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
            if play_direction == "left":
                fx, fy = 120.0 - fx, 53.3 - fy
            players.append({
                "field_x": float(fx),
                "field_y": float(fy),
            })

        if local_idx == 0:
            los = estimate_los(players)
            snap_dets = [d for d in field_dets if d["frame"] == abs_frame]
            if ret and snap_image is not None:
                labels = cluster_teams_by_color(snap_image, snap_dets, debug=debug)
                offense, defense = split_offense_defense_by_color(players, labels, los)
            else:
                by_x = sorted(players, key=lambda p: p["field_x"])
                half = len(players) // 2
                offense, defense = by_x[:half], by_x[half:]
            if debug:
                print(f"  O/D split: {len(offense)} offense, {len(defense)} defense")
                print(f"  LOS: {los:.1f}")
                if offense:
                    print(f"  Offense avg_x: {np.mean([p['field_x'] for p in offense]):.1f}")
                if defense:
                    print(f"  Defense avg_x: {np.mean([p['field_x'] for p in defense]):.1f}")
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


# ── Snap detection ────────────────────────────────────────────────────────

def detect_snap_frame(detections, fps=60.0, window=5, min_players=14):
    """
    Find the snap frame from YOLO tracking detections.

    At the snap, many tracked players suddenly accelerate from a set position.
    We use the derivative of motion (acceleration) rather than raw motion
    to avoid false positives from huddle/walking periods.

    Args:
        detections: list of dicts from track.py
        fps: video frame rate
        window: frames to average for smoothing
        min_players: minimum tracked players per frame to consider

    Returns:
        int: estimated snap frame index
    """
    from collections import defaultdict

    field_dets = [d for d in detections if d["class"] == "Helmet"]

    # Build per-track per-frame positions
    tracks = defaultdict(dict)
    max_frame = 0
    for d in field_dets:
        tracks[d["track_id"]][d["frame"]] = (d["cx"], d["cy"])
        max_frame = max(max_frame, d["frame"])

    # Compute average displacement and player count per frame
    displacements = []
    player_counts = []
    for f in range(1, max_frame + 1):
        total_disp = 0.0
        count = 0
        for tid, positions in tracks.items():
            if f in positions and (f - 1) in positions:
                dx = positions[f][0] - positions[f - 1][0]
                dy = positions[f][1] - positions[f - 1][1]
                total_disp += (dx**2 + dy**2) ** 0.5
                count += 1
        displacements.append(total_disp / count if count > 0 else 0.0)
        player_counts.append(count)

    if len(displacements) < window * 3:
        return 0

    disps = np.array(displacements)
    counts = np.array(player_counts)

    # Smooth with running average
    kernel = np.ones(window) / window
    smooth = np.convolve(disps, kernel, mode='valid')
    smooth_counts = np.convolve(counts.astype(float), kernel, mode='valid')

    # Only consider frames where enough players are tracked (in formation, not huddle)
    valid = smooth_counts >= min_players

    # Compute baseline from valid pre-snap frames (first half of valid region)
    valid_indices = np.where(valid)[0]
    if len(valid_indices) < window * 2:
        return 0
    baseline_region = valid_indices[:len(valid_indices) // 3]
    baseline = np.median(smooth[baseline_region])
    baseline_std = np.std(smooth[baseline_region])
    threshold = baseline + max(3 * baseline_std, 1.5)

    # Find first valid frame where motion exceeds threshold sustainably
    # Subtract window to return the last pre-snap frame (before motion starts)
    for i in range(len(smooth) - window):
        if not valid[i]:
            continue
        if all(smooth[i:i + window] > threshold) and all(valid[i:i + window]):
            return max(0, i - window)

    # Fallback: frame of max motion among valid frames
    masked = np.where(valid, smooth, 0.0)
    snap = int(np.argmax(masked))
    return max(0, snap)
