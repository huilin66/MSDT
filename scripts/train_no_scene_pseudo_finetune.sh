#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the MSDT no-scene model on a pseudo-augmented flat RainDrop_Train.
# The dataset is assumed to already contain matching Drop/Clear files, including
# generated pseudo pairs such as test_pseudo_*.png.

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU=${GPU:-0}
DATA_PATH=${DATA_PATH:-/data/huilin/scrinvme/huilin/tp/eccv_dn/RainDrop_Train}
CONFIG=${CONFIG:-configs/raindrop_no_scene_pseudo.yaml}
LOAD_WEIGHTS=${LOAD_WEIGHTS:-checkpoints/msdt_1x5090/no_scene/model_best.pth}
OUT_ROOT=${OUT_ROOT:-checkpoints/msdt_no_scene_pseudo_1x5090}
OUTPUT_DIR=${OUTPUT_DIR:-${OUT_ROOT}/no_scene_pseudo_ft}

EPOCHS=${EPOCHS:-80}
STOP_AFTER_EPOCH=${STOP_AFTER_EPOCH:-${EPOCHS}}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-5e-5}
NUM_WORKERS=${NUM_WORKERS:-8}
EVAL_EVERY=${EVAL_EVERY:-10}
SAVE_EVERY=${SAVE_EVERY:-5}
MAX_VAL_IMAGES=${MAX_VAL_IMAGES:-160}
CHANNELS_LAST=${CHANNELS_LAST:-1}
TORCH_COMPILE=${TORCH_COMPILE:-0}
COMPILE_MODE=${COMPILE_MODE:-reduce-overhead}

missing_paths=()
check_dir() {
  local label="$1"
  local path="$2"
  if [[ ! -d "${path}" ]]; then
    missing_paths+=("${label}: ${path}")
  fi
}
check_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    missing_paths+=("${label}: ${path}")
  fi
}

check_file "CONFIG" "${CONFIG}"
check_dir "DATA_PATH" "${DATA_PATH}"
check_dir "DATA_PATH/Drop" "${DATA_PATH}/Drop"
check_dir "DATA_PATH/Clear" "${DATA_PATH}/Clear"
if [[ -n "${LOAD_WEIGHTS}" ]]; then
  check_file "LOAD_WEIGHTS" "${LOAD_WEIGHTS}"
fi
if [[ "${#missing_paths[@]}" -gt 0 ]]; then
  echo "Missing required paths:" >&2
  printf '  - %s\n' "${missing_paths[@]}" >&2
  exit 2
fi

extra_args=()
if [[ -n "${LOAD_WEIGHTS}" ]]; then
  extra_args+=(--load-weights "${LOAD_WEIGHTS}")
fi
if [[ "${CHANNELS_LAST}" == "1" ]]; then
  extra_args+=(--channels-last)
fi
if [[ "${TORCH_COMPILE}" == "1" ]]; then
  extra_args+=(--compile --compile-mode "${COMPILE_MODE}")
fi
if [[ -n "${MAX_VAL_IMAGES}" && "${MAX_VAL_IMAGES}" != "0" ]]; then
  extra_args+=(--max-val-images "${MAX_VAL_IMAGES}")
fi

echo "============================================================"
echo "[MSDT no-scene pseudo finetune]"
echo "GPU=${GPU}"
echo "DATA_PATH=${DATA_PATH}"
echo "CONFIG=${CONFIG}"
echo "LOAD_WEIGHTS=${LOAD_WEIGHTS:-<fresh>}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "epochs=${EPOCHS}, stop_after=${STOP_AFTER_EPOCH}, batch=${BATCH_SIZE}, lr=${LR}"
echo "workers=${NUM_WORKERS}, eval_every=${EVAL_EVERY}, save_every=${SAVE_EVERY}, max_val_images=${MAX_VAL_IMAGES}"
echo "channels_last=${CHANNELS_LAST}, torch_compile=${TORCH_COMPILE}, compile_mode=${COMPILE_MODE}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${GPU}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; x=torch.randn(256,256,device='cuda',dtype=torch.bfloat16); print('GPU:', torch.cuda.get_device_name(0), 'capability:', torch.cuda.get_device_capability(0), 'torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'BF16 test:', float((x@x).mean()))"

CUDA_VISIBLE_DEVICES="${GPU}" python -u train_raindrop.py \
  --config "${CONFIG}" \
  --data-root "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --epochs "${EPOCHS}" \
  --stop-after-epoch "${STOP_AFTER_EPOCH}" \
  --num-workers "${NUM_WORKERS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --eval-every "${EVAL_EVERY}" \
  --save-every "${SAVE_EVERY}" \
  "${extra_args[@]}"

echo "MSDT no-scene pseudo finetune finished: ${OUTPUT_DIR}"
