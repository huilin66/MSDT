#!/usr/bin/env python3
"""CPU smoke coverage for both RaindropClarity MSDT ablation paths."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import MSDT
from raindrop_data import (
    RaindropClarityDataset,
    discover_samples,
    load_or_create_split,
    scene_coverage,
)
from raindrop_engine import validate_model
from raindrop_metrics import ValidationMetrics
from raindrop_utils import MultiScaleMSDTLoss, infer_image, tensor_to_png, validate_outputs


def write_image(path: Path, value: int, size=(64, 64)):
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.indices((size[1], size[0]))
    array = np.stack(
        [
            (xx + value) % 256,
            (yy * 2 + value) % 256,
            (xx + yy + value) % 256,
        ],
        axis=2,
    ).astype(np.uint8)
    Image.fromarray(array).save(path)


def make_flat(root: Path):
    labels = {}
    specs = [
        ("Night_00001_bf.png", 0),
        ("Night_00002_rf.png", 1),
        ("Day_00003_bf.png", 2),
        ("Day_00004_rf.png", 3),
        ("Night_00005_bf.png", 0),
        ("Night_00006_rf.png", 1),
        ("Day_00007_bf.png", 2),
        ("Day_00008_rf.png", 3),
    ]
    for index, (name, label) in enumerate(specs):
        write_image(root / "Clear" / name, 20 + index)
        write_image(root / "Drop" / name, 30 + index)
        labels[name] = label
    (root / "Drop_scen_pred.json").write_text(json.dumps(labels), encoding="utf-8")


def make_raw(root: Path):
    specs = [
        ("NightRainDrop", "00001", "00001.png", 0),
        ("NightRainDrop", "00002", "00001.png", 1),
        ("DayRainDrop", "00003", "00001.png", 2),
        ("DayRainDrop", "00004", "00001.png", 3),
    ]
    for index, (dataset, scene, name, label) in enumerate(specs):
        base = root / dataset
        clear_value = 70 + index
        write_image(base / "Clear" / scene / name, clear_value)
        write_image(base / "Drop" / scene / name, clear_value + 10)
        blur_value = clear_value if label in (0, 2) else clear_value + 1
        write_image(base / "Blur" / scene / name, blur_value)


def one_backward(use_scene: bool, item, device: torch.device):
    model = MSDT(num_res=1, use_scene_condition=use_scene).to(device)
    image = item["input"].unsqueeze(0).to(device)
    target = item["target"].unsqueeze(0).to(device)
    scene = item.get("scene_id")
    if scene is not None:
        scene = scene.view(1).to(device)
    outputs = model(image, scene_id=scene) if use_scene else model(image)
    validate_outputs(outputs, image)
    assert [tuple(value.shape[-2:]) for value in outputs] == [(16, 16), (8, 8), (4, 4)]
    loss, _ = MultiScaleMSDTLoss()(outputs, target)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    return float(loss.detach())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA smoke test requested but CUDA is unavailable: {args.device}")
    torch.manual_seed(1234)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1234)
    with tempfile.TemporaryDirectory(prefix="msdt_smoke_") as temporary:
        root = Path(temporary)
        flat, raw = root / "flat", root / "raw"
        make_flat(flat)
        make_raw(raw)

        no_scene, detected = discover_samples(
            str(flat), use_scene_condition=False, scene_json=str(root / "does-not-exist.json")
        )
        assert detected == "flat" and all(sample.scene_id is None for sample in no_scene)
        scene, _ = discover_samples(str(flat), use_scene_condition=True)
        assert scene_coverage(scene) == {0: 2, 1: 2, 2: 2, 3: 2}
        assert [Path(sample.input_path).name for sample in scene] == [sample.filename for sample in scene]

        raw_scene, raw_format = discover_samples(str(raw), use_scene_condition=True)
        assert raw_format == "raw" and scene_coverage(raw_scene) == {0: 1, 1: 1, 2: 1, 3: 1}

        manifest = root / "split.json"
        split = load_or_create_split(no_scene, str(manifest), val_ratio=0.25, seed=1234)
        same_split = load_or_create_split(scene, str(manifest), val_ratio=0.25, seed=1234)
        assert split == same_split and not (set(split["train"]) & set(split["val"]))

        no_train = RaindropClarityDataset(no_scene, split["train"], True, 16, False)
        scene_train = RaindropClarityDataset(scene, split["train"], True, 16, True)
        no_loss = one_backward(False, no_train[0], device)
        scene_loss = one_backward(True, scene_train[0], device)

        base_model = MSDT(num_res=0, use_scene_condition=False).to(device)
        scene_model = MSDT(num_res=0, use_scene_condition=True).to(device)
        incompatible = scene_model.load_state_dict(base_model.state_dict(), strict=False)
        assert not incompatible.unexpected_keys
        assert all(key.startswith("scene_conditioner.") for key in incompatible.missing_keys)
        all_ids = torch.tensor([0, 1, 2, 3], device=device)
        fixed_input = torch.rand(4, 3, 16, 16, device=device)
        base_outputs = base_model(fixed_input)
        four_outputs = scene_model(fixed_input, scene_id=all_ids)
        assert four_outputs[0].shape == (4, 3, 16, 16)
        assert all(torch.equal(a, b) for a, b in zip(base_outputs, four_outputs))
        assert torch.count_nonzero(scene_model.scene_conditioner.mlp[-1].weight) == 0
        assert torch.count_nonzero(scene_model.scene_conditioner.mlp[-1].bias) == 0

        metric = ValidationMetrics(device)
        validation_results = {}
        inference_results = {}
        for mode, samples, train_dataset in (
            ("no_scene", no_scene, no_train),
            ("scene", scene, scene_train),
        ):
            use_scene = mode == "scene"
            val = RaindropClarityDataset(samples, split["val"], False, 16, use_scene)
            model = MSDT(num_res=0, use_scene_condition=use_scene).to(device).eval()
            validation_results[mode] = validate_model(
                model, DataLoader(val, batch_size=1), metric, device, use_scene
            )
            odd = torch.rand(1, 3, 65, 67, device=device)
            sid = torch.tensor([3], device=device) if use_scene else None
            restored = infer_image(model, odd, sid, tile_size=32, tile_overlap=8)
            assert restored.shape == odd.shape
            restored_tta = infer_image(
                model,
                odd,
                sid,
                tile_size=32,
                tile_overlap=8,
                tile_stride=16,
                scales=(1.0, 0.75),
                vflip=True,
                hflip=True,
                rot90=True,
                rot180=True,
                rot270=True,
            )
            assert restored_tta.shape == odd.shape
            output = root / f"{mode}_input.png"
            tensor_to_png(restored, str(output))
            with Image.open(output) as saved:
                assert saved.size == (67, 65) and output.name == f"{mode}_input.png"
            inference_results[mode] = {"filename": output.name, "size": saved.size}

        result = {
            "status": "passed",
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "flat_pairs": len(scene),
            "raw_pairs": len(raw_scene),
            "scene_coverage": scene_coverage(scene),
            "backward_loss": {"no_scene": no_loss, "scene": scene_loss},
            "validation": validation_results,
            "inference": inference_results,
        }
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
