#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/RaindropClarity}"
SCENE_JSON="${2:-${DATA_ROOT}/Drop_scen_pred.json}"
RESULT_DIR="${3:-results/scene_ablation}"

mkdir -p "${RESULT_DIR}"
python eval_raindrop.py --config configs/raindrop_no_scene.yaml \
  --weights checkpoints/raindrop_no_scene/model_best.pth \
  --data-root "${DATA_ROOT}" --output "${RESULT_DIR}/no_scene.json"
python eval_raindrop.py --config configs/raindrop_scene.yaml \
  --weights checkpoints/raindrop_scene/model_best.pth \
  --data-root "${DATA_ROOT}" --scene-json "${SCENE_JSON}" \
  --output "${RESULT_DIR}/scene.json"
python compare_ablation.py --no-scene "${RESULT_DIR}/no_scene.json" \
  --scene "${RESULT_DIR}/scene.json" --output-dir "${RESULT_DIR}"

