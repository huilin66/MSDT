#!/usr/bin/env python3
"""Run lossless MSDT inference on one image, a folder, or the fixed validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from raindrop_data import IMAGE_SUFFIXES
from raindrop_engine import prepare_datasets
from raindrop_utils import (
    build_model,
    infer_image,
    load_config,
    load_model_checkpoint,
    resolve_device,
    tensor_to_png,
)


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def load_labels(path: str) -> dict:
    label_path = Path(path)
    if not label_path.is_file():
        raise FileNotFoundError(f"Scene JSON does not exist: {label_path}")
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    if not isinstance(labels, dict):
        raise ValueError("Scene JSON must map filenames to integer labels")
    for filename, label in labels.items():
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label <= 3:
            raise ValueError(f"Illegal scene label for {filename!r}: {label!r}")
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Input image or folder")
    source.add_argument("--validation", action="store_true", help="Infer the fixed validation split")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", help="Required/optional override with --validation")
    parser.add_argument("--scene-json", help="Filename-to-scene mapping for scene-conditioned inference")
    parser.add_argument("--scene-id", type=int, choices=range(4), help="Manual scene label (especially for one image)")
    parser.add_argument("--device")
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--tile-overlap", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    use_scene = bool(config["experiment"]["use_scene_condition"])
    model = build_model(config).to(device)
    load_model_checkpoint(model, args.weights, device, use_scene)
    model.eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inference = config.get("inference", {})
    tile_size = args.tile_size if args.tile_size is not None else int(inference.get("tile_size", 0))
    overlap = args.tile_overlap if args.tile_overlap is not None else int(inference.get("tile_overlap", 32))

    jobs = []
    if args.validation:
        if args.scene_id is not None:
            raise ValueError("Validation inference uses dataset labels; do not pass --scene-id")
        _, dataset, _ = prepare_datasets(config, args.data_root, args.scene_json)
        for sample in dataset.samples:
            relative = Path(sample.sample_id)
            jobs.append((Path(sample.input_path), relative, sample.scene_id))
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Inference input does not exist: {input_path}")
        if input_path.is_file():
            if input_path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported image file: {input_path}")
            paths = [input_path]
            base = input_path.parent
        else:
            paths = sorted(
                path for path in input_path.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            base = input_path
            if not paths:
                raise ValueError(f"No images found below {input_path}")
        labels = None
        if use_scene and args.scene_id is None:
            label_path = args.scene_json or config["data"].get("scene_json")
            if not label_path:
                raise ValueError("Scene-conditioned inference requires --scene-id or --scene-json")
            labels = load_labels(label_path)
        for path in paths:
            scene_id = None
            if use_scene:
                if args.scene_id is not None:
                    scene_id = args.scene_id
                elif path.name not in labels:
                    raise ValueError(f"Scene JSON is missing label for {path.name!r}")
                else:
                    scene_id = labels[path.name]
            jobs.append((path, path.relative_to(base), scene_id))

    with torch.no_grad():
        for input_path, relative, scene_value in jobs:
            image = load_rgb(input_path).to(device)
            scene_id = torch.tensor([scene_value], device=device) if use_scene else None
            restored = infer_image(
                model,
                image,
                scene_id=scene_id,
                tile_size=tile_size,
                tile_overlap=overlap,
            )
            output_path = (output_dir / relative).with_suffix(".png")
            tensor_to_png(restored, str(output_path))
            with Image.open(output_path) as saved:
                expected_size = (image.shape[-1], image.shape[-2])
                if saved.size != expected_size:
                    raise RuntimeError(f"Saved size mismatch for {output_path}: {saved.size} vs {expected_size}")
            print(output_path)


if __name__ == "__main__":
    main()
