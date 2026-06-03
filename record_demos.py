"""
GR00T Demo Recorder — with camera calibration and gripper smoothing.

CALIBRATION (run once, saves calib.json):
  Press C to enter calibration mode.
  Follow on-screen prompts to touch 3 reference positions.
  Reference positions in robot world frame (metres, Isaac Sim convention):
    Ref 1: hold wrist directly above the CUBE start position
    Ref 2: hold wrist directly above the BOX centre
    Ref 3: raise wrist straight up ~20cm from Ref 1

RECORDING:
  SPACE  - start / stop recording
  Q/ESC  - quit

Output:
  demos/demo_N.mp4            — clean video (no overlays)
  demos/demo_N_landmarks.json — per-frame data in ROBOT world frame
  demos/calib.json            — calibration transform (saved once)
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2
import numpy as np
import json
import time
import threading
from pathlib import Path
from collections import deque

import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python import BaseOptions

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX  = '/dev/video2'
WIDTH, HEIGHT = 1280, 720
FPS           = 25
MODEL_PATH    = Path(__file__).parent / 'hand_landmarker.task'
DEMOS_DIR     = Path(__file__).parent / 'demos'
CALIB_PATH    = DEMOS_DIR / 'calib.json'
DEMOS_DIR.mkdir(exist_ok=True)

# Robot world frame reference positions (Isaac Sim, metres)
# x=forward, y=left, z=up  |  robot base at x=0,y=0,z=0.75
REF_WORLD = np.array([
    [0.45, -0.15, 0.90],   # Ref 1: above cube start position
    [0.60,  0.10, 0.90],   # Ref 2: above box centre
    [0.45, -0.15, 1.10],   # Ref 3: 20cm directly above Ref 1
])
REF_LABELS = [
    "Ref 1: hold wrist ABOVE THE CUBE (table level + ~10cm)",
    "Ref 2: hold wrist ABOVE THE BOX  (table level + ~10cm)",
    "Ref 3: raise wrist 20cm STRAIGHT UP from Ref 1",
]

# ── Gripper config ─────────────────────────────────────────────────────────────
GRIPPER_EMA_ALPHA   = 0.10   # smoothing (lower = smoother, more lag)
GRIPPER_OPEN_THRESH = 0.042  # smoothed thumb-index dist > this → open
GRIPPER_CLOSE_THRESH= 0.022  # smoothed thumb-index dist < this → closed
GRIPPER_OPEN_VAL    = 0.04   # Franka fully open  (metres)
GRIPPER_CLOSE_VAL   = 0.00   # Franka fully closed


class GripperSmoother:
    def __init__(self):
        self._ema   = GRIPPER_OPEN_THRESH   # start open
        self._state = 'open'

    def update(self, raw_dist: float) -> float:
        """Return smoothed, hysteresis-gated Franka gripper value [0, 0.04]."""
        self._ema = GRIPPER_EMA_ALPHA * raw_dist + (1 - GRIPPER_EMA_ALPHA) * self._ema
        if self._ema > GRIPPER_OPEN_THRESH:
            self._state = 'open'
        elif self._ema < GRIPPER_CLOSE_THRESH:
            self._state = 'closed'
        # else: hold — hysteresis dead zone
        return GRIPPER_OPEN_VAL if self._state == 'open' else GRIPPER_CLOSE_VAL

    @property
    def state_str(self) -> str:
        return self._state

    @property
    def ema(self) -> float:
        return self._ema


# ── Calibration ───────────────────────────────────────────────────────────────

def compute_transform(mp_pts: np.ndarray, world_pts: np.ndarray):
    """
    Compute affine transform M, t such that:
        world_pos ≈ M @ mp_pos + t

    Uses 3 point correspondences. Returns (M, t) as np arrays.
    M: (3,3), t: (3,)
    """
    # Translate so Ref 1 is origin in both spaces
    mp_o    = mp_pts[0]
    world_o = world_pts[0]
    A = (mp_pts[1:] - mp_o).T     # (3, 2) — two difference vectors in MP frame
    B = (world_pts[1:] - world_o).T  # (3, 2) — same in world frame

    # Least-squares: find M (3x3) such that M @ A ≈ B
    # With only 2 vectors we get under-determined — compute via SVD and add orthogonal 3rd axis
    M, _, _, _ = np.linalg.lstsq(A.T, B.T, rcond=None)  # M shape (3,3)
    t = world_o - M @ mp_o
    return M, t


def transform_mp_to_world(pos_mp: np.ndarray, M: np.ndarray, t: np.ndarray) -> np.ndarray:
    return M @ pos_mp + t


def load_calibration():
    if CALIB_PATH.exists():
        data = json.load(open(CALIB_PATH))
        M = np.array(data['M'])
        t = np.array(data['t'])
        print(f"[calib] Loaded calibration from {CALIB_PATH}")
        return M, t
    return None, None


def save_calibration(M: np.ndarray, t: np.ndarray):
    json.dump({'M': M.tolist(), 't': t.tolist()}, open(CALIB_PATH, 'w'), indent=2)
    print(f"[calib] Saved to {CALIB_PATH}")


# ── MediaPipe setup ───────────────────────────────────────────────────────────
latest_result = None
result_lock   = threading.Lock()

def on_result(result, output_image, timestamp_ms):
    global latest_result
    with result_lock:
        latest_result = result

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

mp_draw      = mp.solutions.drawing_utils
mp_hand_conn = mp.solutions.hands.HAND_CONNECTIONS

# ── Camera ─────────────────────────────────────────────────────────────────────
import subprocess
subprocess.run(['v4l2-ctl', '-d', CAMERA_INDEX,
                '--set-ctrl=power_line_frequency=1'], capture_output=True)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

WIN = "GR00T Demo Recorder"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

# ── State ──────────────────────────────────────────────────────────────────────
recording   = False
calibrating = False
calib_step  = 0
calib_mp    = []   # collected MediaPipe ref points

writer      = None
landmarks   = []
demo_count  = len(list(DEMOS_DIR.glob('demo_*.mp4')))
frame_idx   = 0
start_time  = 0.0
ts_ms       = 0

M_cal, t_cal = load_calibration()
gripper_sm   = GripperSmoother()

print(f"Output : {DEMOS_DIR}/")
print(f"Demos  : {demo_count} already saved")
print("SPACE=record  C=calibrate  Q=quit")
if M_cal is None:
    print("⚠  No calibration found — press C to calibrate before recording!")

# ── Main loop ──────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detector.detect_async(mp_image, ts_ms)
    ts_ms += 1

    with result_lock:
        result = latest_result

    clean   = frame.copy()
    display = frame.copy()
    hand_detected = False
    wrist_mp      = None
    gripper_val   = GRIPPER_OPEN_VAL

    if result and result.hand_landmarks:
        hand_detected = True
        h, w, _ = display.shape

        for i, (lm_norm, lm_world) in enumerate(
                zip(result.hand_landmarks, result.hand_world_landmarks)):

            side  = result.handedness[i][0].display_name if result.handedness else f"hand{i}"
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
            raw_grip  = float(np.linalg.norm(thumb_tip - index_tip))

            # Prefer right hand
            if side == "Right" or wrist_mp is None:
                wrist_mp    = wrist
                gripper_val = gripper_sm.update(raw_grip)

            # Display info
            if M_cal is not None:
                wrist_world = transform_mp_to_world(wrist, M_cal, t_cal)
                cv2.putText(display,
                    f"{side}  grip:{gripper_sm.state_str}({gripper_sm.ema:.3f})  "
                    f"world:({wrist_world[0]:.2f},{wrist_world[1]:.2f},{wrist_world[2]:.2f})",
                    (10, 35 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)
            else:
                cv2.putText(display,
                    f"{side}  grip:{gripper_sm.state_str}({gripper_sm.ema:.3f})  "
                    f"mp:({wrist[0]:.3f},{wrist[1]:.3f},{wrist[2]:.3f})",
                    (10, 35 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)
    else:
        cv2.putText(display, "No hand detected", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # ── Calibration UI ────────────────────────────────────────────────────────
    if calibrating:
        msg = REF_LABELS[calib_step] if calib_step < len(REF_LABELS) else "Done!"
        cv2.rectangle(display, (0, HEIGHT - 90), (WIDTH, HEIGHT), (30, 30, 30), -1)
        cv2.putText(display, f"CALIBRATE [{calib_step+1}/{len(REF_LABELS)}]  SPACE=capture  ESC=cancel",
                    (10, HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        cv2.putText(display, msg, (10, HEIGHT - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 100), 2)

    # ── Recording overlay ─────────────────────────────────────────────────────
    if recording:
        elapsed = time.time() - start_time

        # Record clean frame
        writer.write(clean)

        # Record data
        if wrist_mp is not None and M_cal is not None:
            wrist_world = transform_mp_to_world(wrist_mp, M_cal, t_cal)
            landmarks.append({
                "frame"       : frame_idx,
                "timestamp"   : round(elapsed, 4),
                "wrist_robot" : wrist_world.tolist(),   # robot world frame (metres)
                "gripper"     : round(gripper_val, 5),  # Franka range [0, 0.04]
                "gripper_ema" : round(gripper_sm.ema, 5),
                "wrist_mp"    : wrist_mp.tolist(),       # raw MP (for debugging)
            })
        else:
            landmarks.append({
                "frame"    : frame_idx,
                "timestamp": round(elapsed, 4),
                "lost"     : True,
            })

        frame_idx += 1
        cv2.rectangle(display, (0, 0), (WIDTH, HEIGHT), (0, 0, 200), 6)
        cv2.putText(display,
            f"REC  {elapsed:.1f}s  demo_{demo_count}  f={frame_idx}  "
            f"grip={gripper_sm.state_str}",
            (10, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2)
    else:
        calib_ok = "✓ calibrated" if M_cal is not None else "⚠ NOT calibrated (press C)"
        cv2.putText(display,
            f"SPACE=record  C=calibrate  |  {calib_ok}  |  saved:{demo_count}",
            (10, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

    cv2.imshow(WIN, display)
    key = cv2.waitKey(1) & 0xFF

    # ── Key handling ──────────────────────────────────────────────────────────
    if calibrating:
        if key == ord(' '):  # capture current wrist position
            if wrist_mp is not None:
                calib_mp.append(wrist_mp.copy())
                print(f"  Captured Ref {calib_step+1}: MP={wrist_mp}")
                calib_step += 1
                if calib_step >= len(REF_LABELS):
                    # All 3 points captured — compute transform
                    mp_pts    = np.array(calib_mp)
                    world_pts = REF_WORLD
                    M_cal, t_cal = compute_transform(mp_pts, world_pts)
                    save_calibration(M_cal, t_cal)
                    # Verify
                    for pi, (pm, pw) in enumerate(zip(mp_pts, world_pts)):
                        pred = transform_mp_to_world(pm, M_cal, t_cal)
                        err  = np.linalg.norm(pred - pw) * 100
                        print(f"  Ref {pi+1} error: {err:.1f} cm  pred={np.round(pred,3)}")
                    calibrating = False
                    calib_step  = 0
                    calib_mp    = []
                    print("[calib] ✓ Complete!")
            else:
                print("  No hand detected — show your hand first")
        elif key == 27:  # ESC cancel calibration
            calibrating = False
            calib_step  = 0
            calib_mp    = []
            print("[calib] Cancelled")

    elif key == ord(' '):
        if not recording:
            if M_cal is None:
                print("⚠  Calibrate first (press C)!")
            else:
                demo_path = DEMOS_DIR / f'demo_{demo_count}.mp4'
                writer    = cv2.VideoWriter(str(demo_path),
                                            cv2.VideoWriter_fourcc(*'mp4v'),
                                            FPS, (WIDTH, HEIGHT))
                landmarks  = []
                frame_idx  = 0
                start_time = time.time()
                gripper_sm = GripperSmoother()  # fresh smoother per demo
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
            print(f"■ Saved demo_{demo_count}: {valid} valid frames, {lost} lost")
            print(f"  {DEMOS_DIR}/demo_{demo_count}.mp4")
            print(f"  {lm_path}")
            demo_count += 1

    elif key == ord('c') or key == ord('C'):
        if not recording:
            calibrating = True
            calib_step  = 0
            calib_mp    = []
            print("\n[calib] Starting calibration — follow on-screen prompts, press SPACE at each position")

    elif key == 27 or key == ord('q') or key == ord('Q'):
        if recording and writer:
            writer.release()
        break

cap.release()
detector.close()
cv2.destroyAllWindows()
print(f"\nDone. {demo_count} demos in {DEMOS_DIR}/")
