#!/usr/bin/env python3
"""Evaluate one MSDT RaindropClarity checkpoint on the fixed validation split."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from raindrop_engine import prepare_datasets, validate_model
from raindrop_metrics import ValidationMetrics
from raindrop_utils import build_model, load_config, load_model_checkpoint, resolve_device, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--scene-json")
    parser.add_argument("--output", help="Metrics JSON path")
    parser.add_argument("--device")
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--tile-overlap", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["training"].get("seed", 1234)))
    device = resolve_device(args.device)
    use_scene = bool(config["experiment"]["use_scene_condition"])
    _, val_dataset, data_info = prepare_datasets(config, args.data_root, args.scene_json)
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    model = build_model(config).to(device)
    load_model_checkpoint(model, args.weights, device, use_scene)
    model.eval()
    metrics = ValidationMetrics(device)
    inference = config.get("inference", {})
    result = validate_model(
        model,
        loader,
        metrics,
        device,
        use_scene,
        tile_size=args.tile_size if args.tile_size is not None else int(inference.get("validation_tile_size", 0)),
        tile_overlap=args.tile_overlap if args.tile_overlap is not None else int(inference.get("tile_overlap", 32)),
    )
    payload = {"model": config["experiment"]["name"], **result, "data": data_info}
    output = Path(args.output or Path(args.weights).with_suffix(".metrics.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

