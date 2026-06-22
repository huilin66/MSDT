#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/RaindropClarity}"
SCENE_JSON="${2:-${DATA_ROOT}/Drop_scen_pred.json}"

python train_raindrop.py --config configs/raindrop_no_scene.yaml --data-root "${DATA_ROOT}"
python train_raindrop.py --config configs/raindrop_scene.yaml --data-root "${DATA_ROOT}" --scene-json "${SCENE_JSON}"

