# GR00T Policy Server — Quick Start

## 1. Download latest checkpoint from HuggingFace

```bash
docker run --rm \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/Isaac-GR00T/checkpoints:/checkpoints \
  gr00t:latest \
  python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'tolasing/groot-pick-place-v5',
    token='<your_hf_token>',
    local_dir='/checkpoints/groot_pick_place_v5'
)
print('Download complete')
"
```

## 2. Start policy server

```bash
docker run --rm --gpus all --ipc=host --network host \
  -e HF_TOKEN=<your_hf_token> \
  -v /root/Isaac-GR00T/checkpoints/groot_pick_place_v5:/checkpoints \
  -v /root/Isaac-GR00T/modality_human_hand.py:/workspace/modality_human_hand.py \
  gr00t:latest \
  python /workspace/gr00t/eval/run_gr00t_server.py \
    --model-path /checkpoints \
    --embodiment-tag new_embodiment \
    --modality-config-path /workspace/modality_human_hand.py \
    --port 5555
```

Server is ready when port 5555 is listening:
```bash
ss -tlnp | grep 5555
```

## Notes
- Checkpoint path: `/root/Isaac-GR00T/checkpoints/groot_pick_place_v5/`
- HuggingFace repo: `tolasing/groot-pick-place-v5`
- ZMQ port: 5555 (shared with Isaac Sim eval via `--network host`)
- Do NOT set `HF_HUB_ENABLE_HF_TRANSFER=1` — `hf_transfer` not installed in container
