#!/usr/bin/env bash
set -euo pipefail

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

# Evaluate the two sequential single-RTX-5090 MSDT runs and build the delta table.
#
# Example:
# DATA_PATH=/data/RainDrop_Train \
# SCENE_VAL_PATH=/data/RainDrop_Train/Drop_scen_pred.json \
# bash scripts/eval_scene_ablation.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU=${GPU:-0}
DATA_PATH=${DATA_PATH:-D:/zhl/data/eccv_dn/RainDrop_Train}
SCENE_VAL_PATH=${SCENE_VAL_PATH:-${DATA_PATH}/Drop_scen_pred.json}
CKPT_ROOT=${CKPT_ROOT:-checkpoints/msdt_1x5090}
NO_SCENE_WEIGHTS=${NO_SCENE_WEIGHTS:-${CKPT_ROOT}/no_scene/model_best.pth}
SCENE_WEIGHTS=${SCENE_WEIGHTS:-${CKPT_ROOT}/scene/model_best.pth}
RESULT_DIR=${RESULT_DIR:-results/msdt_scene_ablation_1x5090}

CUDA_VISIBLE_DEVICES="${GPU}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0), 'capability:', torch.cuda.get_device_capability(0), 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)"

if [[ ! -d "${DATA_PATH}/Drop" || ! -d "${DATA_PATH}/Clear" ]]; then
  echo "Missing paired folders: ${DATA_PATH}/Drop and ${DATA_PATH}/Clear" >&2
  exit 1
fi
if [[ ! -f "${SCENE_VAL_PATH}" ]]; then
  echo "Missing scene label file: ${SCENE_VAL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${NO_SCENE_WEIGHTS}" ]]; then
  echo "Missing no-scene checkpoint: ${NO_SCENE_WEIGHTS}" >&2
  exit 1
fi
if [[ ! -f "${SCENE_WEIGHTS}" ]]; then
  echo "Missing scene checkpoint: ${SCENE_WEIGHTS}" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"

echo "============================================================"
echo "[Eval] no_scene"
echo "Weights: ${NO_SCENE_WEIGHTS}"
echo "============================================================"
CUDA_VISIBLE_DEVICES="${GPU}" python -u eval_raindrop.py \
  --config configs/raindrop_no_scene.yaml \
  --weights "${NO_SCENE_WEIGHTS}" \
  --data-root "${DATA_PATH}" \
  --device cuda:0 \
  --output "${RESULT_DIR}/no_scene.json"

echo "============================================================"
echo "[Eval] scene"
echo "Weights: ${SCENE_WEIGHTS}"
echo "============================================================"
CUDA_VISIBLE_DEVICES="${GPU}" python -u eval_raindrop.py \
  --config configs/raindrop_scene.yaml \
  --weights "${SCENE_WEIGHTS}" \
  --data-root "${DATA_PATH}" \
  --scene-json "${SCENE_VAL_PATH}" \
  --device cuda:0 \
  --output "${RESULT_DIR}/scene.json"

python compare_ablation.py \
  --no-scene "${RESULT_DIR}/no_scene.json" \
  --scene "${RESULT_DIR}/scene.json" \
  --output-dir "${RESULT_DIR}"

echo "RTX 5090 scene ablation evaluation finished: ${RESULT_DIR}"

