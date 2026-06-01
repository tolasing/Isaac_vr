"""
GR00T Demo Recorder — MediaPipe Tasks API (HandLandmarker)
Records video + hand landmark JSON for each demo.

Usage: python record_demos.py
  SPACE  - start / stop recording
  Q/ESC  - quit

Output: demos/demo_N.mp4 + demos/demo_N_landmarks.json
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2
import numpy as np
import json
import time
import threading
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python import BaseOptions

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX  = '/dev/video2'
WIDTH, HEIGHT = 1280, 720
FPS           = 25  # 25fps aligns with 50Hz power line (India)
MODEL_PATH    = Path(__file__).parent / 'hand_landmarker.task'
DEMOS_DIR     = Path(__file__).parent / 'demos'
DEMOS_DIR.mkdir(exist_ok=True)

# ── Shared result (callback runs in separate thread) ──────────────────────────
latest_result = None
result_lock   = threading.Lock()

def on_result(result, output_image, timestamp_ms):
    global latest_result
    with result_lock:
        latest_result = result

# ── HandLandmarker (Tasks API, LIVE_STREAM mode) ──────────────────────────────
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionTaskRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
    result_callback=on_result,
)
detector = HandLandmarker.create_from_options(options)

mp_draw       = mp.solutions.drawing_utils
mp_hand_conn  = mp.solutions.hands.HAND_CONNECTIONS

# ── Fix banding: set C920 power line frequency to 50Hz (India) ───────────────
import subprocess
subprocess.run(['v4l2-ctl', '-d', CAMERA_INDEX,
                '--set-ctrl=power_line_frequency=1'], capture_output=True)

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

WIN = "GR00T Demo Recorder (SPACE=record, Q=quit)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
ret, frame = cap.read()
if ret:
    cv2.imshow(WIN, frame)
    cv2.waitKey(1)

# ── State ─────────────────────────────────────────────────────────────────────
recording  = False
writer     = None
landmarks  = []
demo_count = len(list(DEMOS_DIR.glob('demo_*.mp4')))
frame_idx  = 0
start_time = 0.0
ts_ms      = 0

print(f"Model  : {MODEL_PATH}")
print(f"Output : {DEMOS_DIR}/")
print(f"Demos already saved: {demo_count}")
print("SPACE = start/stop  |  Q/ESC = quit")

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    # Send to Tasks API (async — result arrives via callback)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detector.detect_async(mp_image, ts_ms)
    ts_ms += 1

    # Read latest result thread-safely
    with result_lock:
        result = latest_result

    clean = frame.copy()   # clean frame — written to video, no overlays
    display = frame.copy() # display frame — overlays for live preview only
    hand_detected = False

    if result and result.hand_landmarks:
        hand_detected = True
        h, w, _ = display.shape
        hands_data = {}

        for i, (lm_norm, lm_world) in enumerate(
                zip(result.hand_landmarks, result.hand_world_landmarks)):

            side = result.handedness[i][0].display_name if result.handedness else f"hand{i}"
            color = (0, 255, 0) if side == "Right" else (255, 100, 0)

            # Draw skeleton
            pts = [(int(l.x * w), int(l.y * h)) for l in lm_norm]
            for conn in mp_hand_conn:
                cv2.line(display, pts[conn[0]], pts[conn[1]], color, 2)
            for pt in pts:
                cv2.circle(display, pt, 4, color, -1)

            wrist     = np.array([lm_world[0].x, lm_world[0].y, lm_world[0].z])
            thumb_tip = np.array([lm_world[4].x, lm_world[4].y, lm_world[4].z])
            index_tip = np.array([lm_world[8].x, lm_world[8].y, lm_world[8].z])
            gripper   = float(np.linalg.norm(thumb_tip - index_tip))

            y_offset = 35 + i * 30
            cv2.putText(display, f"{side}  gripper: {gripper:.3f}m  wrist: ({wrist[0]:.2f}, {wrist[1]:.2f}, {wrist[2]:.2f})",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            hands_data[side.lower()] = {
                "handedness"     : side,
                "wrist_world"    : wrist.tolist(),
                "thumb_world"    : thumb_tip.tolist(),
                "index_world"    : index_tip.tolist(),
                "gripper_open"   : gripper,
                "landmarks_norm" : [[l.x, l.y, l.z] for l in lm_norm],
                "landmarks_world": [[l.x, l.y, l.z] for l in lm_world],
            }

        if recording:
            landmarks.append({
                "frame"    : frame_idx,
                "timestamp": time.time() - start_time,
                "hands"    : hands_data,
            })
    else:
        cv2.putText(display, "Show hands to camera", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        if recording:
            landmarks.append({"frame": frame_idx,
                               "timestamp": time.time() - start_time,
                               "lost": True})

    # Recording overlay
    if recording:
        elapsed = time.time() - start_time
        cv2.rectangle(display, (0, 0), (WIDTH, HEIGHT), (0, 0, 220), 6)
        cv2.putText(display, f"REC  {elapsed:.1f}s  demo_{demo_count}  frame {frame_idx}",
                    (10, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        writer.write(clean)  # write clean frame, no overlays
        frame_idx += 1
    else:
        cv2.putText(display, f"SPACE to record  |  saved: {demo_count} demos",
                    (10, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    cv2.imshow(WIN, display)
    key = cv2.waitKey(1) & 0xFF

    if key != 255:  # any key pressed
        print(f"Key received: {key} (SPACE=32, Q=113, ESC=27)")

    if key == ord(' '):
        if not recording:
            demo_path  = DEMOS_DIR / f'demo_{demo_count}.mp4'
            writer     = cv2.VideoWriter(str(demo_path),
                                         cv2.VideoWriter_fourcc(*'mp4v'),
                                         FPS, (WIDTH, HEIGHT))
            landmarks  = []
            frame_idx  = 0
            start_time = time.time()
            recording  = True
            print(f"● Recording demo_{demo_count}...")
        else:
            recording = False
            writer.release()
            writer = None

            lm_path = DEMOS_DIR / f'demo_{demo_count}_landmarks.json'
            with open(lm_path, 'w') as f:
                json.dump(landmarks, f, indent=2)

            valid = sum(1 for x in landmarks if 'lost' not in x)
            lost  = len(landmarks) - valid
            pct   = lost / max(len(landmarks), 1) * 100
            print(f"■ Saved demo_{demo_count}: {valid} valid, {lost} lost ({pct:.1f}%)")
            print(f"  {DEMOS_DIR}/demo_{demo_count}.mp4")
            print(f"  {lm_path}")
            demo_count += 1

    elif key == 27:  # ESC only to quit
        if recording and writer:
            writer.release()
        break

cap.release()
detector.close()
cv2.destroyAllWindows()
print(f"\nDone. {demo_count} demos in {DEMOS_DIR}/")
