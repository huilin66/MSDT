#!/usr/bin/env bash
set -euo pipefail

# Windows/Git-Bash and single-GPU compatibility, following the JiT launchers.
export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

# End-to-end memory/functional smoke test on one RTX 5090 32GB.
# Each setting runs two epochs, one train step and one validation image per epoch.
#
# Example:
# DATA_PATH=/data/RainDrop_Train \
# SCENE_TRAIN_PATH=/data/RainDrop_Train/Drop_scen_pred.json \
# bash scripts/smoke_test_raindrop.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU=${GPU:-0}
DATA_PATH=${DATA_PATH:-D:/zhl/data/eccv_dn/RainDrop_Train}
SCENE_TRAIN_PATH=${SCENE_TRAIN_PATH:-${DATA_PATH}/Drop_scen_pred.json}
OUT_ROOT=${OUT_ROOT:-run/msdt_1x5090_smoke}

NUM_WORKERS=${NUM_WORKERS:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-0.0001}
EPOCHS=${EPOCHS:-2}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-1}
MAX_VAL_IMAGES=${MAX_VAL_IMAGES:-1}

CUDA_VISIBLE_DEVICES="${GPU}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; x=torch.randn(256,256,device='cuda',dtype=torch.bfloat16); print('GPU:', torch.cuda.get_device_name(0), 'capability:', torch.cuda.get_device_capability(0), 'torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'BF16 test:', float((x@x).mean()))"

if [[ ! -d "${DATA_PATH}/Drop" || ! -d "${DATA_PATH}/Clear" ]]; then
  echo "Missing paired folders: ${DATA_PATH}/Drop and ${DATA_PATH}/Clear" >&2
  exit 1
fi

if [[ ! -f "${SCENE_TRAIN_PATH}" ]]; then
  echo "Missing scene label file: ${SCENE_TRAIN_PATH}" >&2
  exit 1
fi

run_smoke() {
  local name="$1"
  local config="$2"
  local use_scene="$3"
  local output_dir="${OUT_ROOT}/${name}_b${BATCH_SIZE}"
  local scene_args=()

  if [[ "${use_scene}" == "1" ]]; then
    scene_args+=(--scene-json "${SCENE_TRAIN_PATH}")
  fi

  echo "============================================================"
  echo "[Smoke] ${name}: use_scene_condition=${use_scene}"
  echo "batch=${BATCH_SIZE}, lr=${LR}, epochs=${EPOCHS}"
  echo "train_steps=${MAX_TRAIN_STEPS}, val_images=${MAX_VAL_IMAGES}"
  echo "Output: ${output_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" python -u train_raindrop.py \
    --config "${config}" \
    --data-root "${DATA_PATH}" \
    --output-dir "${output_dir}" \
    --device cuda:0 \
    --epochs "${EPOCHS}" \
    --num-workers "${NUM_WORKERS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --max-train-steps "${MAX_TRAIN_STEPS}" \
    --max-val-images "${MAX_VAL_IMAGES}" \
    "${scene_args[@]}"

  if [[ ! -f "${output_dir}/model_latest.pth" ]]; then
    echo "Smoke test failed: model_latest.pth was not created for ${name}" >&2
    exit 1
  fi
  echo "[PASS] ${name}"
}

run_smoke "no_scene" "configs/raindrop_no_scene.yaml" 0
run_smoke "scene"    "configs/raindrop_scene.yaml"    1

echo "Both RTX 5090 end-to-end smoke tests passed."
