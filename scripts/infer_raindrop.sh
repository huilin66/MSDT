#!/usr/bin/env bash
set -euo pipefail

# Single-RTX-5090 inference launcher for RaindropClarity MSDT.
#
# Typical usage:
#   INPUT_PATH=D:/path/to/drop.png RUN_MODE=no_scene bash scripts/infer_raindrop.sh
#   INPUT_PATH=D:/path/to/drop.png RUN_MODE=scene SCENE_ID=3 bash scripts/infer_raindrop.sh
#   INPUT_PATH=D:/path/to/Drop RUN_MODE=scene SCENE_JSON=D:/path/to/Drop_scen_pred.json bash scripts/infer_raindrop.sh
#   INFER_VALIDATION=1 RUN_MODE=both bash scripts/infer_raindrop.sh

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU=${GPU:-0}
RUN_MODE=${RUN_MODE:-scene}

DATA_PATH=${DATA_PATH:-D:/zhl/data/eccv_dn/RainDrop_Train}
SCENE_JSON=${SCENE_JSON:-${DATA_PATH}/Drop_scen_pred.json}

CKPT_ROOT=${CKPT_ROOT:-checkpoints/msdt_1x5090}
NO_SCENE_WEIGHTS=${NO_SCENE_WEIGHTS:-${CKPT_ROOT}/no_scene/model_best.pth}
SCENE_WEIGHTS=${SCENE_WEIGHTS:-${CKPT_ROOT}/scene/model_best.pth}

INPUT_PATH=${INPUT_PATH:-}
OUT_ROOT=${OUT_ROOT:-results/msdt_1x5090_infer}
INFER_VALIDATION=${INFER_VALIDATION:-0}

# 0 means whole-image inference with reflect padding. Use e.g. 512/64 for overlap tiles.
TILE_SIZE=${TILE_SIZE:-0}
TILE_OVERLAP=${TILE_OVERLAP:-64}

# For scene-conditioned single-image/folder inference:
# - set SCENE_ID=0/1/2/3 to force one label for all inputs, or
# - leave SCENE_ID empty and provide SCENE_JSON with filename labels.
SCENE_ID=${SCENE_ID:-}

CUDA_VISIBLE_DEVICES="${GPU}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0), 'capability:', torch.cuda.get_device_capability(0), 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)"

if [[ "${INFER_VALIDATION}" == "1" ]]; then
  if [[ ! -d "${DATA_PATH}/Drop" || ! -d "${DATA_PATH}/Clear" ]]; then
    echo "Missing paired folders for validation inference: ${DATA_PATH}/Drop and ${DATA_PATH}/Clear" >&2
    exit 1
  fi
else
  if [[ -z "${INPUT_PATH}" ]]; then
    echo "Set INPUT_PATH for single-image/folder inference, or set INFER_VALIDATION=1." >&2
    exit 1
  fi
  if [[ ! -e "${INPUT_PATH}" ]]; then
    echo "Missing inference input: ${INPUT_PATH}" >&2
    exit 1
  fi
fi

if [[ -n "${SCENE_ID}" && ! "${SCENE_ID}" =~ ^[0-3]$ ]]; then
  echo "SCENE_ID must be one of 0, 1, 2, 3; got: ${SCENE_ID}" >&2
  exit 1
fi

run_infer() {
  local name="$1"
  local config="$2"
  local weights="$3"
  local use_scene="$4"
  local output_dir="${OUT_ROOT}/${name}"
  local source_args=()
  local scene_args=()

  if [[ ! -f "${weights}" ]]; then
    echo "Missing checkpoint for ${name}: ${weights}" >&2
    exit 1
  fi

  if [[ "${INFER_VALIDATION}" == "1" ]]; then
    source_args+=(--validation --data-root "${DATA_PATH}")
  else
    source_args+=(--input "${INPUT_PATH}")
  fi

  if [[ "${use_scene}" == "1" ]]; then
    if [[ -n "${SCENE_ID}" && "${INFER_VALIDATION}" == "1" ]]; then
      echo "Validation inference uses dataset scene labels; do not set SCENE_ID with INFER_VALIDATION=1." >&2
      exit 1
    fi
    if [[ -n "${SCENE_ID}" ]]; then
      scene_args+=(--scene-id "${SCENE_ID}")
    else
      if [[ ! -f "${SCENE_JSON}" ]]; then
        echo "Missing scene label file for scene-conditioned inference: ${SCENE_JSON}" >&2
        exit 1
      fi
      scene_args+=(--scene-json "${SCENE_JSON}")
    fi
  fi

  mkdir -p "${output_dir}"

  echo "============================================================"
  echo "[Infer] ${name}: use_scene_condition=${use_scene}"
  echo "GPU=${GPU}, tile_size=${TILE_SIZE}, tile_overlap=${TILE_OVERLAP}"
  echo "Weights: ${weights}"
  if [[ "${INFER_VALIDATION}" == "1" ]]; then
    echo "Validation data: ${DATA_PATH}"
  else
    echo "Input: ${INPUT_PATH}"
  fi
  echo "Output: ${output_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" python -u infer_raindrop.py \
    --config "${config}" \
    --weights "${weights}" \
    --output-dir "${output_dir}" \
    --device cuda:0 \
    --tile-size "${TILE_SIZE}" \
    --tile-overlap "${TILE_OVERLAP}" \
    "${source_args[@]}" \
    "${scene_args[@]}"
}

case "${RUN_MODE}" in
  no_scene)
    run_infer "no_scene" "configs/raindrop_no_scene.yaml" "${NO_SCENE_WEIGHTS}" 0
    ;;
  scene)
    run_infer "scene" "configs/raindrop_scene.yaml" "${SCENE_WEIGHTS}" 1
    ;;
  both)
    run_infer "no_scene" "configs/raindrop_no_scene.yaml" "${NO_SCENE_WEIGHTS}" 0
    run_infer "scene"    "configs/raindrop_scene.yaml"    "${SCENE_WEIGHTS}"    1
    ;;
  *)
    echo "RUN_MODE must be no_scene, scene, or both; got: ${RUN_MODE}" >&2
    exit 1
    ;;
esac

echo "Inference finished: ${OUT_ROOT}"
