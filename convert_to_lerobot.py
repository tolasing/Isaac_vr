"""
Convert groot demo recordings to LeRobot v2 dataset format for GR00T finetuning.

Usage (inside groot-lerobot container):
  python convert_to_lerobot.py \
    --demos-dir /workspace/demos \
    --output-dir /workspace/lerobot_dataset \
    --task "pick and place box"

Dataset structure produced:
  lerobot_dataset/
    meta/info.json
    meta/episodes.jsonl
    meta/stats.json
    meta/tasks.jsonl
    data/chunk-000/episode_XXXXXX.parquet
    videos/chunk-000/observation.images.top/episode_XXXXXX.mp4
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--demos-dir',  default='/workspace/demos')
parser.add_argument('--output-dir', default='/workspace/lerobot_dataset')
parser.add_argument('--task',       default='pick and place box')
parser.add_argument('--fps',        type=int, default=25)
parser.add_argument('--camera-key', default='observation.images.top')
args = parser.parse_args()

DEMOS_DIR  = Path(args.demos_dir)
OUTPUT_DIR = Path(args.output_dir)
FPS        = args.fps
CAMERA_KEY = args.camera_key
TASK       = args.task

# State: [wrist_x, wrist_y, wrist_z, gripper]  (4D)
# Action: delta of state from current to next frame  (4D)
STATE_DIM  = 4
ACTION_DIM = 4

# ── Collect episodes ──────────────────────────────────────────────────────────
video_files = sorted(DEMOS_DIR.glob('demo_[0-9]*.mp4'),
                     key=lambda p: int(''.join(filter(str.isdigit, p.stem))))
# exclude the original long recording (demo_6.mp4 without zero padding)
video_files = [v for v in video_files if len(v.stem.split('_')[1]) >= 2]

print(f"Found {len(video_files)} episodes in {DEMOS_DIR}")
assert video_files, "No demo_XX.mp4 files found"

# ── Setup output dirs ─────────────────────────────────────────────────────────
(OUTPUT_DIR / 'meta').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'data' / 'chunk-000').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'videos' / 'chunk-000' / CAMERA_KEY).mkdir(parents=True, exist_ok=True)

# ── Process each episode ──────────────────────────────────────────────────────
episodes_meta = []
total_frames  = 0

for ep_idx, video_path in enumerate(tqdm(video_files, desc='Converting')):
    lm_path = DEMOS_DIR / (video_path.stem + '_landmarks.json')

    # Load landmarks
    landmarks = []
    if lm_path.exists():
        with open(lm_path) as f:
            landmarks = json.load(f)

    # Extract state per frame — prefer right hand, fall back to left
    states = []
    for lm in landmarks:
        if 'lost' in lm or 'hands' not in lm:
            states.append(None)
            continue
        hand = lm['hands'].get('right') or lm['hands'].get('left')
        if hand:
            w = hand['wrist_world']
            g = hand['gripper_open']
            states.append([w[0], w[1], w[2], g])
        else:
            states.append(None)

    # Fill missing states via forward fill
    last_valid = [0.0, 0.0, 0.0, 0.5]
    filled = []
    for s in states:
        if s is not None:
            last_valid = s
        filled.append(last_valid[:])

    n = len(filled)
    if n == 0:
        print(f"  Warning: {video_path.name} has no landmark data — skipping")
        continue

    # Compute actions as delta state (relative EEF)
    actions = []
    for i in range(n):
        if i < n - 1:
            delta = [filled[i+1][j] - filled[i][j] for j in range(STATE_DIM)]
        else:
            delta = [0.0] * ACTION_DIM  # last frame: zero action
        actions.append(delta)

    # Build parquet rows
    rows = []
    for i in range(n):
        rows.append({
            'observation.state'    : filled[i],
            'action'               : actions[i],
            'timestamp'            : round(i / FPS, 4),
            'frame_index'          : i,
            'episode_index'        : ep_idx,
            'index'                : total_frames + i,
            'task_index'           : 0,
            'next.done'            : (i == n - 1),
        })

    df = pd.DataFrame(rows)
    pq.write_table(
        pa.Table.from_pandas(df),
        OUTPUT_DIR / 'data' / 'chunk-000' / f'episode_{ep_idx:06d}.parquet'
    )

    # Copy video
    dst_video = OUTPUT_DIR / 'videos' / 'chunk-000' / CAMERA_KEY / f'episode_{ep_idx:06d}.mp4'
    shutil.copy2(video_path, dst_video)

    episodes_meta.append({
        'episode_index' : ep_idx,
        'tasks'         : [TASK],
        'length'        : n,
    })
    total_frames += n

# ── meta/episodes.jsonl ───────────────────────────────────────────────────────
with open(OUTPUT_DIR / 'meta' / 'episodes.jsonl', 'w') as f:
    for ep in episodes_meta:
        f.write(json.dumps(ep) + '\n')

# ── meta/tasks.jsonl ─────────────────────────────────────────────────────────
with open(OUTPUT_DIR / 'meta' / 'tasks.jsonl', 'w') as f:
    f.write(json.dumps({'task_index': 0, 'task': TASK}) + '\n')

# ── meta/stats.json ───────────────────────────────────────────────────────────
# Compute normalization stats from all episodes
all_states  = []
all_actions = []
for ep_idx in range(len(episodes_meta)):
    table = pq.read_table(
        OUTPUT_DIR / 'data' / 'chunk-000' / f'episode_{ep_idx:06d}.parquet'
    )
    all_states.extend(table['observation.state'].to_pylist())
    all_actions.extend(table['action'].to_pylist())

s = np.array(all_states)
a = np.array(all_actions)

stats = {
    'observation.state': {
        'mean': s.mean(0).tolist(), 'std': s.std(0).tolist(),
        'min':  s.min(0).tolist(),  'max': s.max(0).tolist(),
    },
    'action': {
        'mean': a.mean(0).tolist(), 'std': a.std(0).tolist(),
        'min':  a.min(0).tolist(),  'max': a.max(0).tolist(),
    },
}
with open(OUTPUT_DIR / 'meta' / 'stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

# ── meta/info.json ────────────────────────────────────────────────────────────
info = {
    'codebase_version'  : 'v2.0',
    'robot_type'        : 'human_hand',
    'total_episodes'    : len(episodes_meta),
    'total_frames'      : total_frames,
    'total_tasks'       : 1,
    'total_chunks'      : 1,
    'chunks_size'       : 1000,
    'fps'               : FPS,
    'splits'            : {'train': f'0:{len(episodes_meta)}'},
    'data_path'         : 'data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet',
    'video_path'        : f'videos/chunk-{{chunk_index:03d}}/{CAMERA_KEY}/episode_{{episode_index:06d}}.mp4',
    'features'          : {
        'observation.state': {
            'dtype': 'float32', 'shape': [STATE_DIM],
            'names': ['wrist_x', 'wrist_y', 'wrist_z', 'gripper'],
        },
        'action': {
            'dtype': 'float32', 'shape': [ACTION_DIM],
            'names': ['d_wrist_x', 'd_wrist_y', 'd_wrist_z', 'd_gripper'],
        },
        CAMERA_KEY: {
            'dtype': 'video', 'shape': [720, 1280, 3],
            'names': ['height', 'width', 'channel'],
            'video_info': {'video.fps': FPS, 'video.codec': 'mp4v',
                           'video.pix_fmt': 'yuv420p', 'video.is_depth_map': False},
        },
    },
}
with open(OUTPUT_DIR / 'meta' / 'info.json', 'w') as f:
    json.dump(info, f, indent=2)

print(f"\nDone!")
print(f"  Episodes : {len(episodes_meta)}")
print(f"  Frames   : {total_frames}")
print(f"  Output   : {OUTPUT_DIR}")
