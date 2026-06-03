# GR00T Training Context — L40 Machine Summary
> v4 dataset FIXED (calibration M.T bug + synthetic recalibration). Fixed dataset uploaded to HF as `tolasing/groot-lerobot-v5`. Next step: retrain v5 on that dataset.

---

## What Was Done (v1 — original training)

### 1. Environment Setup
- OS: Ubuntu 22.04, user: `satish`
- Added `satish` to sudo group
- Installed Docker (v29.5.2) and added satish to docker group
- Built the GR00T Docker image: `docker build -f docker/Dockerfile.groot -t gr00t .`
- Repo: `https://github.com/tolasing/Isaac-GR00T` (branch: `groot_training`)

---

## Docker & Model Setup (current machine — root@e2e-60-207)

### Build Docker image
```bash
cd /root/Isaac-GR00T
bash docker/build.sh
# Builds gr00t:latest from nvidia/cuda:12.8.0-devel-ubuntu22.04
# Takes ~10–20 min on first build
```

### Download base model (nvidia/GR00T-N1.7-3B)
**Important:** The host machine has no pip/uv. Use the gr00t container — it has `huggingface_hub` pre-installed. Do NOT set `HF_HUB_ENABLE_HF_TRANSFER=1` — `hf_transfer` is not installed in the container venv and will error.

```bash
docker run --rm \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  gr00t \
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/GR00T-N1.7-3B', token='<your_hf_token>')
"
```

Model downloads to `/root/.cache/huggingface/hub/` (~7GB). Mount this cache in all future training runs with `-v /root/.cache/huggingface:/root/.cache/huggingface` to avoid re-downloading.

**Prerequisites (gated models — accept license on HuggingFace first):**
- `nvidia/GR00T-N1.7-3B` — huggingface.co/nvidia/GR00T-N1.7-3B
- `nvidia/Cosmos-Reason2-2B` — huggingface.co/nvidia/Cosmos-Reason2-2B

### Upload checkpoint to HuggingFace
```bash
docker run --rm \
  -v /root/Isaac-GR00T/checkpoints:/checkpoints \
  gr00t \
  python -c "
from huggingface_hub import HfApi
api = HfApi(token='<your_hf_token>')
api.create_repo('tolasing/groot-pick-place-vN', exist_ok=True)
api.upload_folder(
    folder_path='/checkpoints/groot_pick_place_vN/checkpoint-XXXX',
    repo_id='tolasing/groot-pick-place-vN',
    token='<your_hf_token>'
)
"
```

---

### 2. Dataset
- Received 28 human hand pick-and-place demo episodes via `scp` from local machine
- Moved dataset to: `/home/satish/Isaac-GR00T/lerobot_dataset/`
- Fixed `meta/info.json`: changed `chunk_index` → `episode_chunk` in path patterns (required by GR00T loader)
- Created `meta/modality.json` (required by GR00T dataset loader)

**Dataset stats:**
- 28 episodes, 8423 frames, 25fps
- State: `[wrist_x, wrist_y, wrist_z, gripper]` — 4D, MediaPipe world coords (metres)
- Action: `[d_wrist_x, d_wrist_y, d_wrist_z, d_gripper]` — 4D relative delta
- Video: `observation.images.top` (720×1280, overhead camera)
- Scene: white table, 30×20×20cm cardboard box (target), 8cm cube

### 3. v1 Modality Config
- Embodiment tag: `new_embodiment`
- State keys: `["single_arm", "gripper"]` — 3D wrist position + gripper
- Action: `NON_EEF`, `RELATIVE`, 16-step horizon

### 4. v1 Finetuning Command
```bash
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/satish/Isaac-GR00T/lerobot_dataset:/data/lerobot_dataset \
  -v /home/satish/Isaac-GR00T/checkpoints:/data/checkpoints \
  -v /home/satish/Isaac-GR00T/modality_human_hand.py:/data/modality_human_hand.py \
  -e USE_WANDB=0 \
  -e HF_TOKEN=<your_hf_token> \
  gr00t \
  bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /data/lerobot_dataset \
    --embodiment-tag new_embodiment \
    --modality-config-path /data/modality_human_hand.py \
    --output-dir /data/checkpoints/groot_pick_place
```

**Training details:**
- Base model: `nvidia/GR00T-N1.7-3B` (~3B params)
- Trainable parameters: 1.62B (51.54%) — projector + diffusion head only
- Stopped at step ~3000
- Final loss: ~0.058–0.069
- Best checkpoint: `checkpoint-2000` → uploaded to `https://huggingface.co/tolasing/groot-pick-place`

**HuggingFace access required (gated models):**
- `nvidia/GR00T-N1.7-3B` — accept at huggingface.co/nvidia/GR00T-N1.7-3B
- `nvidia/Cosmos-Reason2-2B` — accept at huggingface.co/nvidia/Cosmos-Reason2-2B

---

## What Was Found During Isaac Sim Eval (L4 Machine)

Eval was run using:
- Two Docker containers: `gr00t:latest` (policy server) and `isaac-lab-base-gui` (Isaac Sim)
- Custom scene: white table with legs, Franka Panda on table surface, 30×20×20cm cardboard box, 8cm cube, 45° overhead camera
- 15 episodes run → ~7% success rate (1–2 successes)

**Root cause of low success: state coordinate mismatch**

The v1 modality config included `single_arm` (3D wrist position) in the state. These values in training were MediaPipe world coordinates — tiny values centred near zero (mean `[0.031, 0.075, 0.002]`, std `[0.016, 0.017, 0.027]`).

At inference the Franka EEF position in Isaac Lab world frame was sent instead (values like `[0.5, 0.0, 1.25]`), which is **30–46 standard deviations** outside the training distribution after internal normalisation. The model received nonsensical state input.

**Fix: remove `single_arm` from state, keep only `gripper`.**
The gripper signal (open/close) is meaningful and mappable between human hand and robot.

---

## Re-Training (v2) — COMPLETE ✓

### What Changed
Two changes to `modality_human_hand.py`:

```python
# v1 (broken)
"state": ModalityConfig(delta_indices=[0], modality_keys=["single_arm", "gripper"]),
# action reps: RELATIVE, RELATIVE

# v2 (fixed)
"state": ModalityConfig(delta_indices=[0], modality_keys=["gripper"]),
# action reps: ABSOLUTE, ABSOLUTE  ← changed from RELATIVE to avoid state_key assertion error
```

**Why ABSOLUTE for actions:** `generate_rel_stats` requires a matching state key for each RELATIVE action key. Since `single_arm` was removed from state, the RELATIVE action stats computation crashed. Switching to ABSOLUTE normalises using the action's own mean/std — correct since the dataset already stores pre-computed deltas.

Also required: `meta/modality.json` must exist in the dataset (was missing, needed manual creation):
```json
{
    "state":  { "gripper":     { "start": 3, "end": 4 } },
    "action": { "single_arm":  { "start": 0, "end": 3 },
                "gripper":     { "start": 3, "end": 4 } },
    "video":  { "top":         { "original_key": "observation.images.top" } },
    "annotation": { "human.task_description": { "original_key": "task_index" } }
}
```

Also required: fix `meta/info.json` path patterns (`chunk_index` → `episode_chunk`):
```bash
sed -i 's/{chunk_index:03d}/{episode_chunk:03d}/g' lerobot_dataset/meta/info.json
```

### v2 Training Results
- Machine: `root@164.52.192.207` (E2E Networks cloud GPU)
- Dataset path: `/root/Isaac-GR00T/lerobot_dataset/`
- Checkpoint path: `/root/Isaac-GR00T/checkpoints/groot_pick_place_v2/`
- Speed: ~2 it/s on L40
- Loss curve (smoothed avg per 100 steps):
  - step 100: 1.105 → step 500: 0.456 → step 1000: 0.427 → step 2000: 0.417 → step 3000: 0.404
- Loss plateaued from ~step 500 onwards (expected for 28-episode dataset)
- **Best checkpoint: `checkpoint-2000`** → uploaded to `https://huggingface.co/tolasing/groot-pick-place-v2`

### v2 Docker Training Command (for reference)
```bash
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /root/Isaac-GR00T/lerobot_dataset:/data/lerobot_dataset \
  -v /root/Isaac-GR00T/checkpoints:/data/checkpoints \
  -v /root/Isaac-GR00T/modality_human_hand.py:/data/modality_human_hand.py \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -e USE_WANDB=0 \
  -e HF_TOKEN=<your_hf_token> \
  gr00t \
  bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /data/lerobot_dataset \
    --embodiment-tag new_embodiment \
    --modality-config-path /data/modality_human_hand.py \
    --output-dir /data/checkpoints/groot_pick_place_v2
```

**Notes:**
- Mount `/root/.cache/huggingface` to avoid re-downloading the 7GB base model
- Use `hf upload` (not `huggingface-cli`, deprecated) with `HF_XET_HIGH_PERFORMANCE=1`
- Pre-download base model outside Docker with `hf_transfer` if HF is slow: `pip3 install hf-transfer && HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/GR00T-N1.7-3B', token='...')"`

---

## Re-Training (v3) — COMPLETE ✓

### What Changed from v2
- **New dataset**: 72 episodes (vs 28) — simple one-direction pick-and-place only (no return loop)
- **Scene**: white table, 30×20×20cm cardboard box (rotated 90°, 20cm facing robot), 8cm cube in front of box
- **Camera**: Logitech C920 on head (~91cm above table), overhead ~45° angle
- Same modality config as v2: state=gripper only, action=single_arm+gripper ABSOLUTE

### Dataset Prep
- `meta/modality.json` created (same v2 mapping)
- `meta/info.json` fixed: `{chunk_index:03d}` → `{episode_chunk:03d}`
- Dataset at: `/root/Isaac-GR00T/lerobot_dataset/` (72 episodes, 7673 frames, 25fps)

### Infrastructure Notes (trained on 2× NVIDIA L4 24GB)
- ZeRO stage 2 OOMs on L4 (22GB/GPU too tight for Adam optimizer states)
- **Fix**: switched to ZeRO stage 3 + gradient checkpointing in `gr00t/configs/training/training_config.py`
- Build Docker image: `bash docker/build.sh` then mount updated training_config.py into container

### v3 Training Results
- Machine: 2× NVIDIA L4 24GB
- Trained in two runs: steps 0→3000, then resumed 3000→6000
- Speed: ~2.2 s/it on 2× L4
- Loss never plateaued (unlike v2) — still improving at step 6000:

| Step range | Avg Loss |
|------------|----------|
| 501–1000   | 0.4122   |
| 1001–1500  | 0.3939   |
| 1501–2000  | 0.3843   |
| 2001–2500  | 0.3723   |
| 2501–3000  | 0.3704   |
| 3001–3500  | 0.3815   | ← LR restart bump on resume
| 3501–4000  | 0.3810   |
| 4001–4500  | 0.3644   |
| 4501–5000  | 0.3622   |
| 5001–5500  | 0.3605   |
| 5501–6000  | 0.3541   |

- **Best checkpoint: `checkpoint-6000`** → uploaded to `https://huggingface.co/tolasing/groot-pick-place-v3`

### v3 Training Command (2× L4, ZeRO stage 3)
```bash
# First set deepspeed_stage=3 and gradient_checkpointing=True in
# gr00t/configs/training/training_config.py, then:

docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /root/Isaac-GR00T/lerobot_dataset:/data/lerobot_dataset \
  -v /root/Isaac-GR00T/checkpoints:/data/checkpoints \
  -v /root/Isaac-GR00T/modality_human_hand.py:/data/modality_human_hand.py \
  -v /root/Isaac-GR00T/gr00t/configs/training/training_config.py:/workspace/gr00t/configs/training/training_config.py \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -e USE_WANDB=0 \
  -e NUM_GPUS=2 \
  -e MAX_STEPS=6000 \
  -e SAVE_STEPS=500 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HF_TOKEN=<your_hf_token> \
  gr00t \
  bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /data/lerobot_dataset \
    --embodiment-tag new_embodiment \
    --modality-config-path /data/modality_human_hand.py \
    --output-dir /data/checkpoints/groot_pick_place_v3
```

**To resume from a checkpoint** (same output dir + higher MAX_STEPS — trainer auto-detects latest checkpoint):
```bash
# Change -e MAX_STEPS=9000 (or desired total), keep --output-dir the same
```

**Upload checkpoint:**
```bash
# Run inside gr00t container or any env with huggingface_hub:
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='<your_hf_token>')
api.create_repo('tolasing/groot-pick-place-v3', exist_ok=True)
api.upload_folder(
    folder_path='/checkpoints/groot_pick_place_v3/checkpoint-6000',
    repo_id='tolasing/groot-pick-place-v3',
    token='<your_hf_token>'
)
"
```

---

## Isaac Sim Eval Setup (L4 Machine) — Already Working

### Infrastructure
- Policy server: `gr00t:latest` Docker image
- Isaac Sim: `isaac-lab-base-gui` Docker container (host network mode, so ZMQ port 5555 is shared)
- Eval script: `/root/groot/scripts/eval_groot_franka.py` (mounted at `/workspace/isaaclab/scripts/` inside container)
- Checkpoint location: `/root/Isaac-GR00T/checkpoints/groot_pick_place/checkpoint-2000/`

### Terminal 1 — Start policy server (gr00t container)
```bash
docker run --rm --gpus all --ipc=host --network host \
  -e HF_TOKEN=<your_hf_token> \
  -v /root/Isaac-GR00T/checkpoints:/checkpoints \
  -v /root/Isaac-GR00T/modality_human_hand.py:/workspace/modality_human_hand.py \
  gr00t:latest \
  python /workspace/gr00t/eval/run_gr00t_server.py \
    --model-path /checkpoints/groot_pick_place_v2/checkpoint-2000 \
    --embodiment-tag new_embodiment \
    --modality-config-path /workspace/modality_human_hand.py \
    --port 5555
```

### Terminal 2 — Run Isaac Sim eval
```bash
docker exec -it isaac-lab-base-gui bash
/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/scripts/eval_groot_franka.py \
  --headless --enable_cameras --num_episodes 50
```

### Observation format sent to server (v2)
```python
obs = {
    "video": {"top": np.array(...)},          # (1, T=1, 720, 1280, 3) uint8
    "state": {
        "gripper": np.array([[[ grip_norm ]]]) # (1, T=1, 1) float32
                                               # normalised: 0.013=closed, 0.5=open
    },
    "language": {
        "annotation.human.task_description": [["pick up the cube and place it inside the cardboard box"]]
    },
}
# result[0] = {"single_arm": (1,16,3), "gripper": (1,16,1)}
```

---

---

## Re-Training (v4) — COMPLETE ✓

### What Changed from v3
- **New dataset**: 81 episodes with camera-calibrated coordinates (wrist in Isaac Sim world frame)
- **Modality config**: `single_arm` restored to state + actions back to `RELATIVE` (now valid since state has matching key)
- **Machine**: 1× NVIDIA L40S 46GB — no ZeRO stage 3 needed (single GPU, 46GB headroom)
- **Dataset fix**: `meta/info.json` path pattern fixed `{chunk_index:03d}` → `{episode_chunk:03d}`

### Known issue: gripper stuck open
All 81 episodes have gripper = 0.04 (fully open) throughout — close threshold in `record_demos.py` never triggered. Model will move arm correctly but not grasp. Fix in v5.

### v4 Training Results
- Machine: `root@e2e-60-207` (1× L40S 46GB)
- Dataset: 81 episodes, 14545 frames, 25fps
- Speed: ~1 it/s on L40S
- Loss curve (avg per 500-step window):

| Step range | Avg Loss |
|------------|----------|
| 10–500     | 0.2772   |
| 501–1000   | 0.0550   |
| 1001–1500  | 0.0440   |
| 1501–2000  | 0.0405   |
| 2001–2500  | 0.0356   |
| 2501–3000  | 0.0331   |
| 3001–3500  | 0.0315   |
| 3501–4000  | 0.0303   |
| 4001–4500  | 0.0274   |
| 4501–5000  | 0.0266   |
| 5001–5500  | 0.0258   |
| 5501–6000  | 0.0249   |

- Plateaued from ~step 2000 onwards
- **Best checkpoint: `checkpoint-5000`** → uploaded to `https://huggingface.co/tolasing/groot-pick-place-v4`

### v4 Training Command (1× L40S)
```bash
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /root/Isaac-GR00T/lerobot_dataset:/data/lerobot_dataset \
  -v /root/Isaac-GR00T/checkpoints:/data/checkpoints \
  -v /root/Isaac-GR00T/modality_human_hand.py:/data/modality_human_hand.py \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -e USE_WANDB=0 \
  -e MAX_STEPS=6000 \
  -e SAVE_STEPS=500 \
  -e HF_TOKEN=<your_hf_token> \
  gr00t \
  bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /data/lerobot_dataset \
    --embodiment-tag new_embodiment \
    --modality-config-path /data/modality_human_hand.py \
    --output-dir /data/checkpoints/groot_pick_place_v4
```

**Notes:**
- Single L40S (46GB): ZeRO stage 2 works fine, no extra DeepSpeed flags needed
- Each checkpoint is ~24GB — monitor disk space, manually remove old ones if needed
- `save_total_limit 5` is set but may not clean up reliably mid-run

---

## Key Files

| File | Purpose |
|------|---------|
| `modality_human_hand.py` | GR00T modality config (v4/v5: state=single_arm+gripper, RELATIVE actions) |
| `lerobot_dataset/meta/modality.json` | Dataset-side modality config |
| `lerobot_dataset/meta/info.json` | Dataset metadata (path uses `episode_chunk`, not `chunk_index`) |
| `examples/finetune.sh` | Finetuning launcher |
| `/root/groot/scripts/eval_groot_franka.py` | Isaac Sim eval script |
| `/root/groot/record_demos.py` | Demo recorder (camera calibration + gripper smoothing) |
| `/root/groot/convert_to_lerobot.py` | Converts demos dir → LeRobot dataset |

---

## Calibration Bug (found during v4 eval) — FIXED in record_demos.py

### Bug 1: compute_transform returned M.T instead of M
`np.linalg.lstsq(A.T, B.T)` solves `A.T @ X = B.T`, which gives `X = M_true.T` — the
transpose of the correct transform. The code applied `M_cal @ pos` (wrong direction) causing
the robot to move ~180° away from the cube (cos_sim ≈ -0.96 at eval).

**Fix** (in `compute_transform`, `/root/groot/record_demos.py`):
```python
# WRONG (old):
M, _, _, _ = np.linalg.lstsq(A.T, B.T, rcond=None)

# CORRECT (fixed):
M_T, _, _, _ = np.linalg.lstsq(A.T, B.T, rcond=None)
M = M_T.T
```

### Bug 2: Only 3 reference points → rank-2 matrix
With 3 points you only get 2 difference vectors. lstsq fills the missing 3rd axis with zeros,
making M nearly singular (singular values: [46.9, 5.3, 0.0]). The resulting wrist_robot values
were completely outside the workspace (x in [-1.1, 1.1] instead of [0.35, 0.70]).

**Fix:** Use **4 reference points** that span all 3 axes independently, spread across the full
workspace to minimise sensitivity to hand-placement errors.

### v4 dataset salvage (no camera available)
The 81 v4 demo JSONs each store `wrist_mp` (raw MediaPipe values). A synthetic calibration was
fitted from the data:
- Anchor: gripper-closed frames → cube position (0.45, -0.15, 0.85) in Isaac Sim
- Axis mapping: MP_z→Isaac_x (neg), MP_x→Isaac_y (neg), MP_y→Isaac_z (neg)
- Resulting M and t saved to `demos/calib.json` and applied to all 91 JSONs

Fixed dataset: `tolasing/groot-lerobot-v5` on HuggingFace (81 eps, 14545 frames).

---

## How to Record New Demos (v5+)

### Prerequisites
- Camera (Logitech C920 or similar) connected at `/dev/video2`
- `hand_landmarker.task` model file in `/root/groot/`
- `compute_transform` bug fixed in `record_demos.py` (already done)
- 4 reference positions chosen and physically marked in your recording space

### Step 0 — Choose 4 reference positions
Place physical markers (tape, objects) in your recording space that correspond to these
Isaac Sim world frame positions. Spread them wide:

| Ref | Isaac Sim position (x, y, z) | Description |
|-----|------------------------------|-------------|
| 1   | (0.35, -0.25, 0.85)         | Front-right, low |
| 2   | (0.70,  0.20, 0.85)         | Back-left, low (max x,y span) |
| 3   | (0.35, -0.25, 1.15)         | Front-right, high (30cm above Ref 1) |
| 4   | (0.70, -0.25, 0.85)         | Back-right, low (same y as Ref 1, same x as Ref 2) |

Update `REF_WORLD` in `record_demos.py` if you change these positions.

### Step 1 — Calibrate (run once per camera position)
```bash
python /root/groot/record_demos.py
# Press C → calibration mode starts
# For each ref: hold wrist still at the marked position → press SPACE to capture
# After all 4 refs: calibration is computed and saved to demos/calib.json
# Verify: all 4 ref errors should print as ~0.0 cm
```

**If errors are large (>5 cm):** recapture — wrist wasn't stable when SPACE was pressed.

**Do not move the camera** after calibrating. If camera moves, re-calibrate and re-record.

### Step 2 — Record demos
```bash
# Still in record_demos.py (or re-run it)
# Overlay shows: grip:open(0.035)  world:(0.45, -0.15, 0.90)
# Verify the world coords look reasonable before recording

# Press SPACE to start recording
# Perform pick-and-place: pick up cube → place in box → return hand to start
# Press SPACE to stop
# Repeat for 50+ episodes
```

**Tips for good demos:**
- Move smoothly and deliberately — avoid jerky motions
- Consistently close fingers around cube and open over box
- Keep the same camera angle across all demos
- Mix up starting hand position slightly for generalisation
- Aim for 3–6 seconds per episode

**Gripper thresholds** (in `record_demos.py`):
```python
GRIPPER_OPEN_THRESH  = 0.042   # smoothed thumb-index > this → open (0.04)
GRIPPER_CLOSE_THRESH = 0.022   # smoothed thumb-index < this → closed (0.00)
```
If gripper never closes during recording, lower `GRIPPER_CLOSE_THRESH` to 0.030.

### Step 3 — Convert to LeRobot dataset
```bash
python /root/groot/convert_to_lerobot.py \
  --demos-dir /root/groot/demos \
  --output-dir /root/Isaac-GR00T/lerobot_dataset \
  --task "pick up the cube and place it inside the cardboard box"
```
Generates `meta/modality.json` automatically. Check `meta/info.json` uses `{episode_chunk:03d}`.

### Step 4 — Upload dataset to HuggingFace (optional, for cloud training)
```bash
docker run --rm \
  -v /root/Isaac-GR00T/lerobot_dataset:/lerobot_dataset \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  gr00t:latest \
  python -c "
from huggingface_hub import HfApi
api = HfApi(token='<your_hf_token>')
api.create_repo('tolasing/groot-lerobot-vN', repo_type='dataset', exist_ok=True)
api.upload_folder(
    folder_path='/lerobot_dataset',
    repo_id='tolasing/groot-lerobot-vN',
    repo_type='dataset',
    token='<your_hf_token>'
)
"
```

### Step 5 — Train
Same command as v4 (single L40S), just change the output dir:
```bash
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /root/Isaac-GR00T/lerobot_dataset:/data/lerobot_dataset \
  -v /root/Isaac-GR00T/checkpoints:/data/checkpoints \
  -v /root/Isaac-GR00T/modality_human_hand.py:/data/modality_human_hand.py \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -e USE_WANDB=0 -e MAX_STEPS=6000 -e SAVE_STEPS=500 \
  -e HF_TOKEN=<your_hf_token> \
  gr00t \
  bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /data/lerobot_dataset \
    --embodiment-tag new_embodiment \
    --modality-config-path /data/modality_human_hand.py \
    --output-dir /data/checkpoints/groot_pick_place_v5
```

---

## References
- GR00T GitHub: https://github.com/NVIDIA/Isaac-GR00T
- Fork: https://github.com/tolasing/Isaac-GR00T (branch: `groot_training`)
- v1 Checkpoint: https://huggingface.co/tolasing/groot-pick-place
- v2 Checkpoint: https://huggingface.co/tolasing/groot-pick-place-v2
- v3 Checkpoint: https://huggingface.co/tolasing/groot-pick-place-v3
- v4 Checkpoint: https://huggingface.co/tolasing/groot-pick-place-v4
- v5 Dataset:    https://huggingface.co/datasets/tolasing/groot-lerobot-v5
