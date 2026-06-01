# GR00T Training Context
> Share this with Claude on the cloud GPU before starting work.

---

## Goal

Train GR00T N1.7 VLA model on human hand pick-and-place demos, evaluate on Franka Panda in Isaac Sim.

**Pipeline:**
```
Phone/webcam demos → MediaPipe hand tracking → LeRobot format → GR00T finetune → Isaac Sim eval
```

---

## What's Already Done (local machine)

| Step | Status | Output |
|------|--------|--------|
| Record demos (Logitech C920) | ✅ | `demos/demo_01.mp4` ... `demo_28.mp4` |
| Hand landmark extraction | ✅ | `demos/demo_XX_landmarks.json` (MediaPipe Tasks API, world coords) |
| Segment into episodes | ✅ | 28 episodes, 15-30s each, full pick-and-place loop |
| Convert to LeRobot v2 format | ✅ | `lerobot_dataset/` |

---

## Task Description

**Pick and place — full loop:**
```
Start: cube at position A, hands visible
  → pick cube from A
  → place cube at B
  → pick cube from B
  → return cube to A
End: cube back at A
```

- One hand (right)
- White table, plain background
- Overhead camera angle, consistent across all demos
- 28 demos total

---

## Dataset Format (LeRobot v2)

```
lerobot_dataset/
  meta/info.json          ← 28 episodes, 8423 frames, 25fps
  meta/episodes.jsonl
  meta/stats.json         ← normalization stats
  meta/tasks.jsonl        ← task: "pick and place box"
  data/chunk-000/
    episode_000000.parquet ... episode_000027.parquet
  videos/chunk-000/observation.images.top/
    episode_000000.mp4 ... episode_000027.mp4
```

**State:** `[wrist_x, wrist_y, wrist_z, gripper]` — 4D, world coordinates in metres
**Action:** delta state `[d_wrist_x, d_wrist_y, d_wrist_z, d_gripper]` — 4D relative EEF

---

## Repo Structure

```
groot/                         ← github.com/tolasing/groot  (branch: groot_training)
├── docker/
│   ├── Dockerfile.groot       ← standalone training container (PyTorch 2.7, CUDA 12.6)
│   │                             clones Isaac-GR00T @ n1.7-release tag
│   └── Dockerfile.lerobot     ← conversion container (already used)
├── finetune_groot.sh          ← finetuning script (run inside groot-training container)
├── record_demos.py            ← demo recorder (local, MediaPipe Tasks API)
├── segment_demos.py           ← episode segmentation tool (local)
└── convert_to_lerobot.py      ← LeRobot v2 converter (local)
```

---

## Hardware

| Machine | GPU | VRAM | Role |
|---------|-----|------|------|
| E2E L40 | L40 | 48 GB | Finetune GR00T (this machine) |
| E2E L4  | L4  | 24 GB | Isaac Sim eval (image saved, ready) |
| Local   | ASUS laptop | — | Data collection (done) |

---

## What To Do on This Machine (L40)

### 1. Clone the repo and get the dataset

```bash
git clone https://github.com/tolasing/groot.git
cd groot
git checkout groot_training
```

Copy the `lerobot_dataset/` folder here (scp from local or use a shared drive).

### 2. Build the training container

```bash
docker build -f docker/Dockerfile.groot -t groot-training .
```

This takes ~15-20 min (flash-attn build).

### 3. Run finetuning

```bash
docker run -it --rm --gpus all \
  -v $(pwd)/lerobot_dataset:/workspace/lerobot_dataset \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  groot-training \
  bash /workspace/groot/finetune_groot.sh
```

Downloads GR00T-N1.7 (~8GB) then starts finetuning.
Expected time: ~1-2 hours on L40.

### 4. After training

```bash
# Copy checkpoint to L4 for Isaac Sim eval
scp -r checkpoints/groot_pick_place user@<l4-ip>:/workspace/checkpoints/
```

---

## Key Facts About GR00T N1.7

- **~3B parameter** VLA model (VLM backbone + Diffusion Transformer head)
- **Default finetuning** tunes projector + diffusion head only — keeps VRAM under ~35GB
- **Action space:** relative end-effector delta (same for human hand and Franka)
- **Inference on L4:** ~4-6 Hz PyTorch (sufficient for Isaac Sim eval)
- **LAPA mode:** GR00T learns latent actions from consecutive video frames
- Model: `nvidia/GR00T-N1.7` on HuggingFace

---

## If Finetuning OOMs

```bash
# Reduce batch size
training.batch_size=8 training.gradient_accumulation_steps=4

# Enable gradient checkpointing
training.gradient_checkpointing=true

# These should not be needed on L40 48GB for default finetuning
```

---

## Next After Finetuning

On the **L4 machine** (Isaac Sim already installed):

```bash
# Terminal 1 — start policy server
python scripts/inference_service.py \
  --model-path /workspace/checkpoints/groot_pick_place/checkpoint_final \
  --embodiment franka \
  --port 8000

# Terminal 2 — run Isaac Sim eval
cd /workspace/isaaclab
python source/standalone/gr00t/eval_gr00t.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --policy-server http://localhost:8000 \
  --num-episodes 50
```

---

## Contacts / Refs

- GR00T N1.7 GitHub: https://github.com/NVIDIA/Isaac-GR00T (tag: n1.7-release)
- GR00T HuggingFace: https://huggingface.co/nvidia/GR00T-N1.7
- LeRobot: https://github.com/huggingface/lerobot
- Repo: https://github.com/tolasing/groot (branch: groot_training)
