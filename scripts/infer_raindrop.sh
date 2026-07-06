#!/usr/bin/env bash
set -euo pipefail

# Single-RTX-5090 inference launcher for RaindropClarity MSDT.
#
# Typical usage:
#   INPUT_PATH=D:/path/to/drop.png RUN_MODE=no_scene bash scripts/infer_raindrop.sh
#   INPUT_PATH=D:/path/to/drop.png RUN_MODE=scene SCENE_ID=3 bash scripts/infer_raindrop.sh
#   INPUT_PATH=D:/path/to/Drop RUN_MODE=scene SCENE_JSON=D:/path/to/Drop_scen_pred.json bash scripts/infer_raindrop.sh
#   INFER_VALIDATION=1 RUN_MODE=both bash scripts/infer_raindrop.sh
#   CREATE_SUBMISSION=1 INPUT_PATH=D:/path/to/Drop RUN_MODE=no_scene bash scripts/infer_raindrop.sh
#   VFLIP=1 HFLIP=1 ROT90=1 ROT180=1 ROT270=1 INPUT_PATH=D:/path/to/Drop RUN_MODE=no_scene bash scripts/infer_raindrop.sh
#   SCALE=1,0.75 STRIDE=384 TILE_SIZE=512 INPUT_PATH=D:/path/to/Drop RUN_MODE=no_scene bash scripts/infer_raindrop.sh

export TORCHDYNAMO_DISABLE=1
export USE_LIBUV=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU=${GPU:-0}
RUN_MODE=${RUN_MODE:-no_scene}

DATA_PATH=${DATA_PATH:-D:/zhl/data/eccv_dn}
SCENE_JSON=${SCENE_JSON:-${DATA_PATH}/Drop_scen_pred.json}

CKPT_ROOT=${CKPT_ROOT:-checkpoints/msdt_1x5090}
CKPT_TYPE=${CKPT_TYPE:-last}
case "${CKPT_TYPE}" in
  best)
    DEFAULT_CKPT_NAME="model_best.pth"
    ;;
  last|latest)
    DEFAULT_CKPT_NAME="model_latest.pth"
    ;;
  *)
    echo "CKPT_TYPE must be best, last, or latest; got: ${CKPT_TYPE}" >&2
    exit 1
    ;;
esac
NO_SCENE_WEIGHTS=${NO_SCENE_WEIGHTS:-${CKPT_ROOT}/no_scene/${DEFAULT_CKPT_NAME}}
SCENE_WEIGHTS=${SCENE_WEIGHTS:-${CKPT_ROOT}/scene/${DEFAULT_CKPT_NAME}}

INPUT_PATH=${INPUT_PATH:-${DATA_PATH}/Drop}
OUT_ROOT=${OUT_ROOT:-results/msdt_1x5090_infer}
INFER_VALIDATION=${INFER_VALIDATION:-0}

# Submission packaging. When CREATE_SUBMISSION=1, PNGs are flattened by default,
# zipped, and one row is appended to HISTORY_CSV.
CREATE_SUBMISSION=${CREATE_SUBMISSION:-1}
SUBMISSION_ROOT=${SUBMISSION_ROOT:-submissions/msdt_1x5090}
HISTORY_CSV=${HISTORY_CSV:-${SUBMISSION_ROOT}/submission_history.csv}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
MODEL_NAME=${MODEL_NAME:-}
SUBMISSION_INFO=${SUBMISSION_INFO:-readme.txt}
NOTES=${NOTES:-}
FLATTEN_OUTPUT=${FLATTEN_OUTPUT:-${CREATE_SUBMISSION}}
REMOVE_IMAGES_AFTER_ZIP=${REMOVE_IMAGES_AFTER_ZIP:-0}

# 0 means whole-image inference with reflect padding. Use e.g. 512/64 for overlap tiles.
TILE_SIZE=${TILE_SIZE:-0}
TILE_OVERLAP=${TILE_OVERLAP:-16}
STRIDE=${STRIDE:-}
SCALE=${SCALE:-1}
VFLIP=${VFLIP:-1}
HFLIP=${HFLIP:-0}
ROT90=${ROT90:-1}
ROT180=${ROT180:-0}
ROT270=${ROT270:-0}

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

for flag_name in VFLIP HFLIP ROT90 ROT180 ROT270; do
  flag_value="${!flag_name}"
  if [[ ! "${flag_value}" =~ ^[01]$ ]]; then
    echo "${flag_name} must be 0 or 1; got: ${flag_value}" >&2
    exit 1
  fi
done

if [[ -n "${STRIDE}" && ! "${STRIDE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STRIDE must be a positive integer when set; got: ${STRIDE}" >&2
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
  local submission_args=()
  local prediction_args=()
  local run_model_name="${MODEL_NAME:-msdt_${name}}"

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

  if [[ "${CREATE_SUBMISSION}" == "1" ]]; then
    output_dir="${SUBMISSION_ROOT}/${run_model_name}_${CKPT_TYPE}_${RUN_TAG}"
    submission_args+=(
      --archive-path "${SUBMISSION_ROOT}/${run_model_name}_${CKPT_TYPE}_${RUN_TAG}.zip"
      --history-csv "${HISTORY_CSV}"
      --model-name "${run_model_name}"
      --notes "${NOTES}"
    )
    if [[ -n "${SUBMISSION_INFO}" ]]; then
      submission_args+=(--submission-info "${SUBMISSION_INFO}")
    fi
  fi

  if [[ "${FLATTEN_OUTPUT}" == "1" ]]; then
    submission_args+=(--flatten-output)
  fi

  if [[ "${REMOVE_IMAGES_AFTER_ZIP}" == "1" ]]; then
    submission_args+=(--remove-images-after-zip)
  fi

  if [[ "${VFLIP}" == "1" ]]; then
    prediction_args+=(--vflip)
  fi

  if [[ "${HFLIP}" == "1" ]]; then
    prediction_args+=(--hflip)
  fi

  if [[ "${ROT90}" == "1" ]]; then
    prediction_args+=(--rot90)
  fi

  if [[ "${ROT180}" == "1" ]]; then
    prediction_args+=(--rot180)
  fi

  if [[ "${ROT270}" == "1" ]]; then
    prediction_args+=(--rot270)
  fi

  if [[ -n "${STRIDE}" ]]; then
    prediction_args+=(--stride "${STRIDE}")
  fi

  mkdir -p "${output_dir}"

  echo "============================================================"
  echo "[Infer] ${name}: use_scene_condition=${use_scene}"
  echo "GPU=${GPU}, tile_size=${TILE_SIZE}, tile_overlap=${TILE_OVERLAP}, stride=${STRIDE:-auto}, scale=${SCALE}, vflip=${VFLIP}, hflip=${HFLIP}, rot90=${ROT90}, rot180=${ROT180}, rot270=${ROT270}"
  echo "Weights: ${weights}"
  if [[ "${INFER_VALIDATION}" == "1" ]]; then
    echo "Validation data: ${DATA_PATH}"
  else
    echo "Input: ${INPUT_PATH}"
  fi
  echo "Output: ${output_dir}"
  if [[ "${CREATE_SUBMISSION}" == "1" ]]; then
    echo "Archive: ${SUBMISSION_ROOT}/${run_model_name}_${CKPT_TYPE}_${RUN_TAG}.zip"
    echo "History CSV: ${HISTORY_CSV}"
  fi
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" python -u infer_raindrop.py \
    --config "${config}" \
    --weights "${weights}" \
    --output-dir "${output_dir}" \
    --device cuda:0 \
    --tile-size "${TILE_SIZE}" \
    --tile-overlap "${TILE_OVERLAP}" \
    --scale "${SCALE}" \
    "${source_args[@]}" \
    "${scene_args[@]}" \
    "${prediction_args[@]}" \
    "${submission_args[@]}"
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
