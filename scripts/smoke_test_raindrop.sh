#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"

if [[ "${DEVICE}" == cuda* ]]; then
  # Expose exactly one physical RTX 5090; it becomes logical cuda:0.
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Use DEVICE=cpu only for a CPU fallback smoke test.")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"Expected exactly one visible GPU, found {torch.cuda.device_count()}.")
print(f"Smoke test GPU: {torch.cuda.get_device_name(0)}")
PY
fi

python -B tests/smoke_test_raindrop.py --device "${DEVICE}"
