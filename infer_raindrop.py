#!/usr/bin/env python3
"""Run lossless MSDT inference on one image, a folder, or the fixed validation split."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

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


HISTORY_FIELDS = [
    "timestamp",
    "model_name",
    "archive_name",
    "checkpoint",
    "config",
    "use_scene",
    "input_mode",
    "input_path",
    "data_root",
    "scene_json",
    "scene_id",
    "num_images",
    "tile_size",
    "tile_overlap",
    "vflip",
    "rot90",
    "output_dir",
    "runtime_seconds",
    "psnr_y",
    "ssim_y",
    "lpips",
    "score",
    "notes",
]

DEFAULT_SUBMISSION_INFO = (
    "runtime per video [s] : 1.00\r\n"
    "CPU[1] / GPU[0] : 0\r\n"
    "Extra Data [1] / No Extra Data [0] : 0\r\n"
    "Other description : MSDT baseline\r\n"
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


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return name.strip("._-") or "msdt"


def default_model_name(weights_path: str, use_scene: bool) -> str:
    path = Path(weights_path)
    parent = path.parent.name
    if parent and parent not in {".", ""}:
        return parent
    return "scene" if use_scene else "no_scene"


def create_archive(image_dir: Path, archive_path: Path, submission_info: str | None = None) -> int:
    # Match submit_jit.py exactly: the archive root contains only flat PNG files
    # from image_dir/*.png, never an outer folder or nested relative paths.
    image_files = sorted(image_dir.glob("*.png"))
    if not image_files:
        raise RuntimeError(
            f"No root-level prediction PNG files under: {image_dir}. "
            "Use --flatten-output for submission ZIPs."
        )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for image_path in image_files:
            archive.write(image_path, arcname=image_path.name)
        if submission_info:
            info_path = Path(submission_info)
            if info_path.is_file():
                archive.write(info_path, arcname=info_path.name)
            elif info_path.name.lower() == "readme.txt":
                archive.writestr("readme.txt", DEFAULT_SUBMISSION_INFO)
            else:
                print(f"Warning: submission info file not found, skipped: {info_path}")
    return len(image_files)


def append_history(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            header = reader.fieldnames or []
            rows = list(reader)
        if header != HISTORY_FIELDS:
            if not set(header).issubset(HISTORY_FIELDS):
                raise RuntimeError(f"Existing history CSV has an incompatible header: {path}")
            with path.open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
                writer.writeheader()
                for old_row in rows:
                    writer.writerow({field: old_row.get(field, "") for field in HISTORY_FIELDS})
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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
    parser.add_argument("--vflip", action="store_true", help="Average original and vertically flipped predictions")
    parser.add_argument("--rot90", action="store_true", help="Average original and 90-degree rotated predictions")
    parser.add_argument("--archive-path", help="Optional ZIP path for generated PNGs")
    parser.add_argument("--history-csv", help="Optional CSV path for submission/inference history")
    parser.add_argument("--model-name", default="", help="Name recorded in CSV; defaults to checkpoint parent")
    parser.add_argument("--submission-info", default="readme.txt", help="Optional file added to the ZIP; empty disables it")
    parser.add_argument("--notes", default="")
    parser.add_argument("--flatten-output", action="store_true", help="Save all PNGs at output-dir root as <stem>.png")
    parser.add_argument("--remove-images-after-zip", action="store_true")
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

    if args.flatten_output:
        output_names = [f"{input_path.stem}.png" for input_path, _, _ in jobs]
        duplicates = sorted(name for name, count in Counter(output_names).items() if count > 1)
        if duplicates:
            raise RuntimeError(
                f"Duplicate output names cannot be flattened into one submission folder: {duplicates[0]}"
            )

    started = time.perf_counter()
    with torch.no_grad():
        for input_path, relative, scene_value in tqdm(jobs, desc="MSDT inference", unit="img"):
            image = load_rgb(input_path).to(device)
            scene_id = torch.tensor([scene_value], device=device) if use_scene else None
            restored = infer_image(
                model,
                image,
                scene_id=scene_id,
                tile_size=tile_size,
                tile_overlap=overlap,
                vflip=args.vflip,
                rot90=args.rot90,
            )
            if args.flatten_output:
                output_path = output_dir / f"{input_path.stem}.png"
            else:
                output_path = (output_dir / relative).with_suffix(".png")
            tensor_to_png(restored, str(output_path))
            with Image.open(output_path) as saved:
                expected_size = (image.shape[-1], image.shape[-2])
                if saved.size != expected_size:
                    raise RuntimeError(f"Saved size mismatch for {output_path}: {saved.size} vs {expected_size}")
    print(f"Saved PNGs: {output_dir}")
    runtime = time.perf_counter() - started

    archive_path = Path(args.archive_path) if args.archive_path else None
    if archive_path is not None:
        archived = create_archive(output_dir, archive_path, args.submission_info or None)
        if archived != len(jobs):
            raise RuntimeError(f"Archive contains {archived} PNGs, expected {len(jobs)}")
        print(f"Archive: {archive_path}")

    if args.history_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = sanitize_name(args.model_name or default_model_name(args.weights, use_scene))
        append_history(
            Path(args.history_csv),
            {
                "timestamp": timestamp,
                "model_name": model_name,
                "archive_name": archive_path.name if archive_path is not None else "",
                "checkpoint": str(Path(args.weights).resolve()),
                "config": str(Path(args.config).resolve()),
                "use_scene": int(use_scene),
                "input_mode": "validation" if args.validation else "input",
                "input_path": "" if args.validation else str(Path(args.input).resolve()),
                "data_root": str(Path(args.data_root).resolve()) if args.data_root else "",
                "scene_json": str(Path(args.scene_json).resolve()) if args.scene_json else "",
                "scene_id": "" if args.scene_id is None else args.scene_id,
                "num_images": len(jobs),
                "tile_size": tile_size,
                "tile_overlap": overlap,
                "vflip": int(args.vflip),
                "rot90": int(args.rot90),
                "output_dir": str(output_dir.resolve()),
                "runtime_seconds": round(runtime, 3),
                "psnr_y": "",
                "ssim_y": "",
                "lpips": "",
                "score": "",
                "notes": args.notes,
            },
        )
        print(f"History CSV: {args.history_csv}")

    if args.remove_images_after_zip:
        if archive_path is None:
            raise ValueError("--remove-images-after-zip requires --archive-path")
        shutil.rmtree(output_dir)
        print(f"Removed image directory: {output_dir}")

    print(f"Runtime: {runtime:.2f}s ({runtime / max(1, len(jobs)):.3f}s/image)")


if __name__ == "__main__":
    main()
