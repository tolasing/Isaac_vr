"""
GR00T Demo Segmenter
Plays back a recorded demo video. Press SPACE to mark start/end of each episode.
Saves each episode as demo_01.mp4, demo_02.mp4, ... + matching landmarks JSON.

Usage: python segment_demos.py demos/demo_6.mp4
Controls:
  SPACE     - mark episode start / end
  R         - redo last mark
  BACKSPACE - cancel current in-progress segment
  Q / ESC   - finish and export all marked segments
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2
import json
import sys
import numpy as np
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('demos/demo_6.mp4')
lm_path    = video_path.with_name(video_path.stem + '_landmarks.json')
out_dir    = video_path.parent

# ── Load landmarks ─────────────────────────────────────────────────────────────
landmarks = []
if lm_path.exists():
    with open(lm_path) as f:
        landmarks = json.load(f)
    print(f"Loaded {len(landmarks)} landmark frames")

# ── Open video ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(str(video_path))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS)
width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {total_frames} frames @ {fps:.1f}fps  ({total_frames/fps:.1f}s)")

# ── Find next available demo number ──────────────────────────────────────────
existing = sorted(out_dir.glob('demo_[0-9][0-9].mp4'))
next_num = int(existing[-1].stem.split('_')[1]) + 1 if existing else 1

# ── State ─────────────────────────────────────────────────────────────────────
segments   = []   # list of (start_frame, end_frame)
mark_start = None
frame_idx  = 0

WIN = "Segmenter — SPACE=mark in/out  R=redo  BKSP=cancel  Q=export"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

print("\nWatch the video — press SPACE at the start of each episode,")
print("then SPACE again at the end. Repeat for each episode. Q to export.\n")

# ── Playback loop ─────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        # End of video — pause and wait
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()

    display = frame.copy()
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    ts = frame_idx / fps

    # Progress bar
    bar_w = int((frame_idx / total_frames) * width)
    cv2.rectangle(display, (0, height - 8), (bar_w, height), (0, 200, 0), -1)

    # Draw marked segments on progress bar
    for (s, e) in segments:
        sx = int(s / total_frames * width)
        ex = int(e / total_frames * width)
        cv2.rectangle(display, (sx, height - 8), (ex, height), (0, 100, 255), -1)

    # Status
    if mark_start is not None:
        elapsed = (frame_idx - mark_start) / fps
        cv2.putText(display,
                    f"RECORDING EPISODE {len(segments)+1}  {elapsed:.1f}s — press SPACE to end",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.rectangle(display, (0, 0), (width, height), (0, 0, 200), 4)
    else:
        cv2.putText(display,
                    f"t={ts:.1f}s  frame={frame_idx}/{total_frames}  "
                    f"episodes marked: {len(segments)}  — SPACE to mark start",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow(WIN, display)
    key = cv2.waitKey(30) & 0xFF  # ~30fps playback

    if key == ord(' '):
        if mark_start is None:
            mark_start = frame_idx
            print(f"  Episode {len(segments)+1} start: frame {frame_idx} ({frame_idx/fps:.1f}s)")
        else:
            if frame_idx > mark_start + int(fps * 3):  # min 3s per episode
                segments.append((mark_start, frame_idx))
                print(f"  Episode {len(segments)} end:   frame {frame_idx} ({frame_idx/fps:.1f}s)  "
                      f"duration: {(frame_idx-mark_start)/fps:.1f}s")
                mark_start = None
            else:
                print("  Too short (< 3s) — ignored, keep going")

    elif key == ord('r'):
        if segments:
            removed = segments.pop()
            print(f"  Removed last segment ({removed[0]/fps:.1f}s - {removed[1]/fps:.1f}s)")
        mark_start = None

    elif key == 8:  # BACKSPACE
        mark_start = None
        print("  Cancelled current mark")

    elif key in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()

if not segments:
    print("No segments marked — exiting.")
    sys.exit(0)

# ── Export segments ────────────────────────────────────────────────────────────
print(f"\nExporting {len(segments)} episodes...")
cap = cv2.VideoCapture(str(video_path))

for i, (start, end) in enumerate(segments):
    demo_num  = next_num + i
    out_video = out_dir / f'demo_{demo_num:02d}.mp4'
    out_lm    = out_dir / f'demo_{demo_num:02d}_landmarks.json'

    # Write video
    writer = cv2.VideoWriter(str(out_video),
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(end - start):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    writer.release()

    # Slice landmarks
    episode_lm = [lm for lm in landmarks
                  if start <= lm.get('frame', -1) < end]
    # Reindex timestamps from 0
    t0 = episode_lm[0]['timestamp'] if episode_lm else 0
    for lm in episode_lm:
        lm['timestamp'] = round(lm['timestamp'] - t0, 4)
        lm['frame']     = lm['frame'] - start

    with open(out_lm, 'w') as f:
        json.dump(episode_lm, f, indent=2)

    duration = (end - start) / fps
    valid    = sum(1 for x in episode_lm if 'lost' not in x)
    print(f"  demo_{demo_num:02d}: {duration:.1f}s  {valid}/{len(episode_lm)} valid frames  "
          f"→ {out_video.name}")

cap.release()
print(f"\nDone. {len(segments)} demos saved to {out_dir}/")
