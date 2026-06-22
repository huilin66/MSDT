#!/usr/bin/env bash
set -euo pipefail

# Windows/Git-Bash and single-GPU compatibility, following the JiT launchers.
export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

# Two-way MSDT scene ablation on one RTX 5090 32GB. Runs are sequential.
#
# Example:
# DATA_PATH=/data/RainDrop_Train \
# SCENE_TRAIN_PATH=/data/RainDrop_Train/Drop_scen_pred.json \
# bash scripts/train_scene_ablation.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU=${GPU:-0}
DATA_PATH=${DATA_PATH:-D:/zhl/data/eccv_dn/RainDrop_Train}
SCENE_TRAIN_PATH=${SCENE_TRAIN_PATH:-${DATA_PATH}/Drop_scen_pred.json}
OUT_ROOT=${OUT_ROOT:-checkpoints/msdt_1x5090}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-${EPOCHS:-200}}
STOP_AFTER_EPOCH=${STOP_AFTER_EPOCH:-${TOTAL_EPOCHS}}
NUM_WORKERS=${NUM_WORKERS:-8}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-0.0001}
RUN_MODE=${RUN_MODE:-both}
NO_SCENE_RESUME=${NO_SCENE_RESUME:-}
SCENE_RESUME=${SCENE_RESUME:-}

CUDA_VISIBLE_DEVICES="${GPU}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; x=torch.randn(256,256,device='cuda',dtype=torch.bfloat16); print('GPU:', torch.cuda.get_device_name(0), 'capability:', torch.cuda.get_device_capability(0), 'torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'BF16 test:', float((x@x).mean()))"

if [[ ! -d "${DATA_PATH}/Drop" || ! -d "${DATA_PATH}/Clear" ]]; then
  echo "Missing paired folders: ${DATA_PATH}/Drop and ${DATA_PATH}/Clear" >&2
  exit 1
fi

if [[ "${RUN_MODE}" != "no_scene" && ! -f "${SCENE_TRAIN_PATH}" ]]; then
  echo "Missing scene label file: ${SCENE_TRAIN_PATH}" >&2
  exit 1
fi

run_exp() {
  local name="$1"
  local config="$2"
  local use_scene="$3"
  local resume="$4"
  local output_dir="${OUT_ROOT}/${name}"
  local scene_args=()
  local resume_args=()

  if [[ "${use_scene}" == "1" ]]; then
    scene_args+=(--scene-json "${SCENE_TRAIN_PATH}")
  fi
  if [[ -n "${resume}" ]]; then
    if [[ ! -f "${resume}" ]]; then
      echo "Missing resume checkpoint: ${resume}" >&2
      exit 1
    fi
    resume_args+=(--resume "${resume}")
  fi

  echo "============================================================"
  echo "[Ablation] ${name}: use_scene_condition=${use_scene}"
  echo "GPU=${GPU}, total_epochs=${TOTAL_EPOCHS}, stop_after=${STOP_AFTER_EPOCH}"
  echo "batch=${BATCH_SIZE}, lr=${LR}, workers=${NUM_WORKERS}"
  echo "Data: ${DATA_PATH}"
  echo "Output: ${output_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" python -u train_raindrop.py \
    --config "${config}" \
    --data-root "${DATA_PATH}" \
    --output-dir "${output_dir}" \
    --device cuda:0 \
    --epochs "${TOTAL_EPOCHS}" \
    --stop-after-epoch "${STOP_AFTER_EPOCH}" \
    --num-workers "${NUM_WORKERS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    "${scene_args[@]}" \
    "${resume_args[@]}"
}

case "${RUN_MODE}" in
  no_scene)
    run_exp "no_scene" "configs/raindrop_no_scene.yaml" 0 "${NO_SCENE_RESUME}"
    ;;
  scene)
    run_exp "scene" "configs/raindrop_scene.yaml" 1 "${SCENE_RESUME}"
    ;;
  both)
    run_exp "no_scene" "configs/raindrop_no_scene.yaml" 0 "${NO_SCENE_RESUME}"
    run_exp "scene"    "configs/raindrop_scene.yaml"    1 "${SCENE_RESUME}"
    ;;
  *)
    echo "RUN_MODE must be no_scene, scene, or both; got: ${RUN_MODE}" >&2
    exit 1
    ;;
esac

echo "Requested single-RTX-5090 MSDT run(s) finished."
