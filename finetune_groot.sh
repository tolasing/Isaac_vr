#!/bin/bash
# GR00T N1.7 Finetuning Script
# Run inside groot-training container on L40 48GB
#
# Usage:
#   docker run -it --rm --gpus all \
#     -v $(pwd)/lerobot_dataset:/workspace/lerobot_dataset \
#     -v $(pwd)/checkpoints:/workspace/checkpoints \
#     groot-training bash finetune_groot.sh

set -e

DATASET_DIR="/workspace/lerobot_dataset"
CHECKPOINT_DIR="/workspace/checkpoints"
OUTPUT_DIR="/workspace/checkpoints/groot_pick_place"
MODEL_DIR="${CHECKPOINT_DIR}/GR00T-N1.7"
WANDB_PROJECT="groot-pick-place"

# ── Step 1: Download GR00T N1.7 base model ────────────────────────────────────
if [ ! -d "${MODEL_DIR}" ]; then
    echo "Downloading GR00T-N1.7 from HuggingFace..."
    huggingface-cli download nvidia/GR00T-N1.7 \
        --local-dir "${MODEL_DIR}"
else
    echo "Model already downloaded at ${MODEL_DIR}"
fi

# ── Step 2: Finetune ──────────────────────────────────────────────────────────
echo "Starting finetuning on L40..."

cd /workspace/Isaac-GR00T

python launch_finetune.py \
    --config-name finetune \
    dataset.repo_id="${DATASET_DIR}" \
    dataset.root="${DATASET_DIR}" \
    dataset.video_backend=torchvision_av \
    training.num_epochs=100 \
    training.batch_size=16 \
    training.learning_rate=1e-4 \
    training.mixed_precision=bf16 \
    training.gradient_accumulation_steps=2 \
    model.pretrained_model_name_or_path="${MODEL_DIR}" \
    output_dir="${OUTPUT_DIR}" \
    experiment_name="groot_pick_place_box"

echo "Finetuning complete. Checkpoint saved to ${OUTPUT_DIR}"
echo ""
echo "Next: copy checkpoint to L4 and run Isaac Sim eval"
echo "  scp -r ${OUTPUT_DIR} user@l4-ip:/workspace/checkpoints/"
