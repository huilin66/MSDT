"""Shared RaindropClarity dataset preparation and validation loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from raindrop_data import (
    RaindropClarityDataset,
    discover_samples,
    load_or_create_split,
    scene_coverage,
)
from raindrop_metrics import ValidationMetrics, aggregate_metrics
from raindrop_utils import infer_image


def prepare_datasets(
    config: Dict[str, Any],
    data_root: Optional[str] = None,
    scene_json: Optional[str] = None,
) -> Tuple[RaindropClarityDataset, RaindropClarityDataset, Dict[str, Any]]:
    data = config["data"]
    use_scene = bool(config["experiment"]["use_scene_condition"])
    root = data_root or data["root"]
    labels_path = scene_json or data.get("scene_json")
    samples, detected_format = discover_samples(
        root,
        data_format=data.get("format", "auto"),
        use_scene_condition=use_scene,
        scene_json=labels_path,
        group_regex=data.get("group_regex", r"(?i)^(day|night)(?:raindrop)?(?:__|[_-])(\d+)"),
        allow_unmatched_flat_groups=bool(data.get("allow_unmatched_flat_groups", False)),
    )
    split = load_or_create_split(
        samples,
        data["split_manifest"],
        float(data.get("val_ratio", 0.1)),
        int(config["training"]["seed"]),
    )
    train_dataset = RaindropClarityDataset(
        samples,
        split["train"],
        training=True,
        patch_size=int(data.get("patch_size", 256)),
        use_scene_condition=use_scene,
    )
    val_dataset = RaindropClarityDataset(
        samples,
        split["val"],
        training=False,
        patch_size=int(data.get("patch_size", 256)),
        use_scene_condition=use_scene,
    )
    info = {
        "format": detected_format,
        "samples": len(samples),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_groups": len(split["train"]),
        "val_groups": len(split["val"]),
        "scene_coverage": scene_coverage(samples) if use_scene else None,
    }
    return train_dataset, val_dataset, info


@torch.no_grad()
def validate_model(
    model: torch.nn.Module,
    loader,
    metrics: ValidationMetrics,
    device: torch.device,
    use_scene: bool,
    tile_size: int = 0,
    tile_overlap: int = 32,
) -> Dict[str, float]:
    model.eval()
    totals = {"PSNR_Y": 0.0, "SSIM_Y": 0.0, "LPIPS": 0.0}
    count = 0
    for batch in loader:
        image = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        scene_id = batch["scene_id"].to(device) if use_scene else None
        restored = infer_image(
            model,
            image,
            scene_id=scene_id,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        current = metrics.compute(restored, target)
        for key in totals:
            totals[key] += current[key]
        count += image.shape[0]
    return aggregate_metrics(totals, count)
