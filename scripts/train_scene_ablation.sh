#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${1:-data/RaindropClarity}"
SCENE_JSON="${2:-${DATA_ROOT}/Drop_scen_pred.json}"
GPU_ID="${GPU_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Expose exactly one physical RTX 5090. Inside PyTorch it is always cuda:0.
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. RTX 5090 training requires a CUDA-enabled PyTorch build.")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"Expected exactly one visible GPU, found {torch.cuda.device_count()}.")
print(f"Using one GPU: {torch.cuda.get_device_name(0)}")
PY

python -u train_raindrop.py --config configs/raindrop_no_scene.yaml \
  --data-root "${DATA_ROOT}" --device cuda:0 --num-workers "${NUM_WORKERS}"
python -u train_raindrop.py --config configs/raindrop_scene.yaml \
  --data-root "${DATA_ROOT}" --scene-json "${SCENE_JSON}" \
  --device cuda:0 --num-workers "${NUM_WORKERS}"

