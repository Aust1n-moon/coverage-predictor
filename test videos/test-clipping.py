"""
Auto-clip pass plays from a full game YouTube video.
Detects snaps via motion analysis, filters for pass plays using motion spread.

Usage:
    python test-clipping.py

Change VIDEO_URL below to process a different video.
"""

import subprocess
import cv2
import numpy as np
import os

VIDEO_URL = "https://www.youtube.com/watch?v=7ERkAhvmHCo&list=PLbGLzzomCfA6R66tNqbLmv32dZoDj8-9l"

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_VIDEO = os.path.join(OUTPUT_DIR, "_full_game.mp4")

PRE_SNAP_SEC = 2
POST_SNAP_SEC = 6
MIN_CALM_SEC = 1.5


def download_video(url, output_path):
    subprocess.run([
        "yt-dlp", "--no-playlist", "--merge-output-format", "mp4",
        "-o", output_path, url
    ], check=True)


FRAME_SKIP = 3  # only process every Nth frame


def compute_motion(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_gray = None
    scores = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if prev_gray is not None:
            scores.append(float(cv2.absdiff(gray, prev_gray).mean()))
        else:
            scores.append(0.0)
        prev_gray = gray
        frame_idx += 1

        if frame_idx % (int(fps) * 60) < FRAME_SKIP:
            pct = frame_idx / total * 100
            print(f"  {pct:.0f}% ({frame_idx}/{total})")

    cap.release()
    effective_fps = fps / FRAME_SKIP
    return np.array(scores), effective_fps


def detect_snaps(motion, fps):
    kernel = max(1, int(fps * 0.5))
    smoothed = np.convolve(motion, np.ones(kernel) / kernel, mode='same')

    calm_thresh = np.percentile(smoothed, 30)
    active_thresh = np.percentile(smoothed, 70)
    min_calm = int(MIN_CALM_SEC * fps)

    snaps = []
    i = 0
    while i < len(smoothed) - min_calm:
        calm_start = None
        calm_count = 0
        for j in range(i, len(smoothed)):
            if smoothed[j] < calm_thresh:
                if calm_start is None:
                    calm_start = j
                calm_count += 1
            else:
                if calm_count >= min_calm:
                    break
                calm_start = None
                calm_count = 0

        if calm_start is None or calm_count < min_calm:
            break

        snap = None
        search_end = min(calm_start + calm_count + int(fps * 3), len(smoothed))
        for j in range(calm_start + calm_count, search_end):
            if smoothed[j] > active_thresh:
                snap = j
                break

        if snap is not None:
            snaps.append(snap)
            i = snap + int(POST_SNAP_SEC * fps)
        else:
            i = calm_start + calm_count + 1

    return snaps


def is_pass_play(video_path, snap_frame, fps):
    """
    Pass plays have motion spread across the frame (routes) and last longer.
    Run plays have concentrated, shorter bursts.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check 1-3 seconds after snap for motion spread
    start = snap_frame + int(1.0 * fps)
    end = min(snap_frame + int(3.5 * fps), total_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    prev_gray = None
    spreads = []
    motion_vals = []
    grid = 4

    for _ in range(end - start):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_vals.append(diff.mean())

            h, w = diff.shape
            ch, cw = h // grid, w // grid
            cells = []
            for r in range(grid):
                for c in range(grid):
                    cells.append(diff[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw].mean())
            cells = np.array(cells)
            active = (cells > np.median(cells) * 1.5).sum()
            spreads.append(active)

        prev_gray = gray

    cap.release()

    if not spreads or not motion_vals:
        return False

    avg_spread = np.mean(spreads)
    # Play duration: how many frames have significant motion after snap
    motion_frames = sum(1 for m in motion_vals if m > np.percentile(motion_vals, 40))
    duration_sec = motion_frames / fps

    # ponytail: rough heuristic — pass plays are spread + longer. Tune thresholds if needed.
    return avg_spread > grid * grid * 0.3 and duration_sec > 1.5


def clip_play(video_path, snap_frame, fps, output_path):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start = max(0, snap_frame - int(PRE_SNAP_SEC * fps))
    end = min(snap_frame + int(POST_SNAP_SEC * fps), total)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(end - start):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()


def main():
    if VIDEO_URL == "https://www.youtube.com/watch?v=CHANGE_ME":
        print("Set VIDEO_URL in the script first.")
        return

    if not os.path.exists(TEMP_VIDEO):
        print("Downloading video...")
        download_video(VIDEO_URL, TEMP_VIDEO)
    else:
        print("Using cached download")

    print("Computing motion scores...")
    motion, fps = compute_motion(TEMP_VIDEO)
    print(f"  {len(motion)} frames at {fps:.1f} fps ({len(motion)/fps:.0f}s)")

    print("Detecting snaps...")
    snaps = detect_snaps(motion, fps)
    print(f"  Found {len(snaps)} potential plays")

    print("Filtering for pass plays and clipping...")
    clip_num = 0
    for i, snap in enumerate(snaps):
        snap_time = snap / fps
        mins, secs = int(snap_time // 60), snap_time % 60
        print(f"  Play {i + 1}/{len(snaps)} at {mins}:{secs:04.1f}...", end=" ")

        if is_pass_play(TEMP_VIDEO, snap, fps):
            clip_num += 1
            out_path = os.path.join(OUTPUT_DIR, f"auto_clip{clip_num:03d}.mp4")
            clip_play(TEMP_VIDEO, snap, fps, out_path)
            snap_offset = int(PRE_SNAP_SEC * fps)
            print(f"pass → {os.path.basename(out_path)} (--snap-frame {snap_offset})")
        else:
            print("skip (run/short)")

    print(f"\nDone! {clip_num} pass play clips saved")
    print(f"Each clip's snap frame: --snap-frame {int(PRE_SNAP_SEC * fps)}")


if __name__ == "__main__":
    main()
