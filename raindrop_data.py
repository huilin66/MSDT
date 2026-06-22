"""RaindropClarity data discovery, grouped splits, and paired augmentation."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageChops
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_FLAT_GROUP_REGEX = r"(?i)^(day|night)(?:raindrop)?(?:__|[_-])(\d+)"


@dataclass(frozen=True)
class RaindropSample:
    sample_id: str
    group_id: str
    filename: str
    input_path: str
    target_path: str
    scene_id: Optional[int] = None


def _image_files(directory: Path) -> Dict[str, Path]:
    return {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def _flat_group(filename: str, regex: str) -> str:
    match = re.match(regex, Path(filename).stem)
    if match is None:
        raise ValueError(
            f"Cannot parse a scene/triplet group from flat filename {filename!r}. "
            "Set data.group_regex to match the dataset naming convention."
        )
    return "_".join(str(part).lower() for part in match.groups())


def _load_scene_json(path: Path, filenames: Iterable[str]) -> Dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Scene-label JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if not isinstance(labels, dict):
        raise ValueError(f"Scene-label JSON must be an object mapping filenames to labels: {path}")
    required = set(filenames)
    missing = sorted(required - set(labels))
    if missing:
        raise ValueError(
            f"Scene-label JSON is missing {len(missing)} input filename(s), e.g. {missing[:5]}"
        )
    parsed: Dict[str, int] = {}
    for name in sorted(required):
        value = labels[name]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3:
            raise ValueError(f"Illegal scene label for {name!r}: {value!r}; expected integer 0..3")
        parsed[name] = value
    return parsed


def _discover_flat(
    root: Path,
    use_scene_condition: bool,
    scene_json: Optional[Path],
    group_regex: str,
) -> List[RaindropSample]:
    drop_dir, clear_dir = root / "Drop", root / "Clear"
    inputs, targets = _image_files(drop_dir), _image_files(clear_dir)
    if not inputs:
        raise ValueError(f"No images found in {drop_dir}")
    missing_gt = sorted(set(inputs) - set(targets))
    orphan_gt = sorted(set(targets) - set(inputs))
    if missing_gt or orphan_gt:
        raise ValueError(
            "Flat Drop/Clear filenames do not match exactly. "
            f"Missing GT: {missing_gt[:5]}; orphan GT: {orphan_gt[:5]}"
        )
    labels: Dict[str, int] = {}
    if use_scene_condition:
        labels = _load_scene_json(scene_json or root / "Drop_scen_pred.json", inputs)
    return [
        RaindropSample(
            sample_id=name,
            group_id=_flat_group(name, group_regex),
            filename=name,
            input_path=str(inputs[name]),
            target_path=str(targets[name]),
            scene_id=labels.get(name),
        )
        for name in sorted(inputs)
    ]


def _raw_roots(root: Path) -> List[Path]:
    if (root / "Drop").is_dir() and (root / "Clear").is_dir():
        return [root]
    found = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / "Drop").is_dir() and (path / "Clear").is_dir()
    ]
    if not found:
        raise ValueError(
            f"Could not find Drop/Clear under {root}. Expected flat Drop/Clear folders or "
            "DayRainDrop/NightRainDrop-style dataset folders."
        )
    return found


def _day_or_night(dataset_root: Path, relative_path: Path) -> str:
    text = f"{dataset_root.name}/{relative_path.as_posix()}".lower()
    has_day, has_night = "day" in text, "night" in text
    if has_day == has_night:
        raise ValueError(
            f"Cannot unambiguously infer day/night for {dataset_root / relative_path}; "
            "include Day or Night in the dataset/scene directory name."
        )
    return "day" if has_day else "night"


def _images_equal(first: Path, second: Path) -> bool:
    with Image.open(first) as image_a, Image.open(second) as image_b:
        a, b = image_a.convert("RGB"), image_b.convert("RGB")
        return a.size == b.size and ImageChops.difference(a, b).getbbox() is None


def _discover_raw(root: Path, use_scene_condition: bool) -> List[RaindropSample]:
    samples: List[RaindropSample] = []
    for dataset_root in _raw_roots(root):
        drop_dir, clear_dir, blur_dir = (
            dataset_root / "Drop",
            dataset_root / "Clear",
            dataset_root / "Blur",
        )
        input_paths = sorted(
            path for path in drop_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not input_paths:
            raise ValueError(f"No images found below {drop_dir}")
        for input_path in input_paths:
            relative = input_path.relative_to(drop_dir)
            if len(relative.parts) < 2:
                raise ValueError(
                    f"Raw format expects Drop/<scene-or-triplet>/<frame>, got {input_path}"
                )
            target_path = clear_dir / relative
            if not target_path.is_file():
                raise FileNotFoundError(f"Missing clean GT for {input_path}: expected {target_path}")
            time_of_day = _day_or_night(dataset_root, relative)
            scene_id: Optional[int] = None
            if use_scene_condition:
                blur_path = blur_dir / relative
                if not blur_path.is_file():
                    raise FileNotFoundError(
                        f"Missing Blur image required to determine raw focus label: {blur_path}"
                    )
                background_focused = _images_equal(blur_path, target_path)
                scene_id = (2 if time_of_day == "day" else 0) + (0 if background_focused else 1)
            group_id = f"{dataset_root.name.lower()}::{relative.parts[0]}"
            sample_id = f"{dataset_root.name}/{relative.as_posix()}"
            samples.append(
                RaindropSample(
                    sample_id=sample_id,
                    group_id=group_id,
                    filename=relative.name,
                    input_path=str(input_path),
                    target_path=str(target_path),
                    scene_id=scene_id,
                )
            )
    return samples


def discover_samples(
    data_root: str,
    data_format: str = "auto",
    use_scene_condition: bool = False,
    scene_json: Optional[str] = None,
    group_regex: str = DEFAULT_FLAT_GROUP_REGEX,
) -> Tuple[List[RaindropSample], str]:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    if data_format not in {"auto", "flat", "raw"}:
        raise ValueError(f"data_format must be auto, flat, or raw; got {data_format!r}")
    if data_format == "auto":
        direct_drop = root / "Drop"
        if direct_drop.is_dir() and any(path.is_dir() for path in direct_drop.iterdir()):
            data_format = "raw"
        elif direct_drop.is_dir():
            data_format = "flat"
        else:
            data_format = "raw"
    if data_format == "flat":
        samples = _discover_flat(
            root,
            use_scene_condition,
            Path(scene_json).expanduser().resolve() if scene_json else None,
            group_regex,
        )
    else:
        samples = _discover_raw(root, use_scene_condition)
    if not samples:
        raise ValueError(f"No paired samples discovered below {root}")
    return samples, data_format


def scene_coverage(samples: Sequence[RaindropSample]) -> Dict[int, int]:
    return dict(sorted(Counter(s.scene_id for s in samples if s.scene_id is not None).items()))


def _group_fingerprint(groups: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(groups)).encode("utf-8")).hexdigest()


def _sample_fingerprint(samples: Sequence[RaindropSample]) -> str:
    records = sorted(f"{sample.group_id}\t{sample.sample_id}" for sample in samples)
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def load_or_create_split(
    samples: Sequence[RaindropSample],
    manifest_path: str,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    groups = sorted({sample.group_id for sample in samples})
    sample_fingerprint = _sample_fingerprint(samples)
    if len(groups) < 2:
        raise ValueError("At least two scene/triplet groups are required for a leak-free train/val split")
    path = Path(manifest_path)
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        train_groups = manifest.get("train_groups")
        val_groups = manifest.get("val_groups")
        if not isinstance(train_groups, list) or not isinstance(val_groups, list):
            raise ValueError(f"Invalid split manifest: {path}")
        if set(train_groups) & set(val_groups):
            raise ValueError(f"Split manifest leaks groups between train and val: {path}")
        if set(train_groups) | set(val_groups) != set(groups):
            raise ValueError(
                f"Split manifest groups no longer match the dataset: {path}. "
                "Use a new manifest path for a changed dataset."
            )
        if manifest.get("sample_fingerprint") != sample_fingerprint:
            raise ValueError(
                f"Split manifest sample list no longer matches the dataset: {path}. "
                "Use a new manifest path for a changed dataset."
            )
        return {"train": train_groups, "val": val_groups}

    shuffled = groups[:]
    random.Random(seed).shuffle(shuffled)
    val_count = min(len(groups) - 1, max(1, round(len(groups) * val_ratio)))
    val_groups = sorted(shuffled[:val_count])
    train_groups = sorted(shuffled[val_count:])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "seed": seed,
        "val_ratio": val_ratio,
        "group_fingerprint": _group_fingerprint(groups),
        "sample_fingerprint": sample_fingerprint,
        "train_groups": train_groups,
        "val_groups": val_groups,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return {"train": train_groups, "val": val_groups}


def _to_tensor_rgb(path: str) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class RaindropClarityDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[RaindropSample],
        groups: Sequence[str],
        training: bool,
        patch_size: int = 256,
        use_scene_condition: bool = False,
    ):
        selected = set(groups)
        self.samples = [sample for sample in samples if sample.group_id in selected]
        if not self.samples:
            raise ValueError("Dataset split contains no samples")
        self.training = training
        self.patch_size = patch_size
        self.use_scene_condition = use_scene_condition
        if use_scene_condition and any(sample.scene_id is None for sample in self.samples):
            raise ValueError("Scene-conditioned dataset contains samples without a scene label")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        input_image, target = _to_tensor_rgb(sample.input_path), _to_tensor_rgb(sample.target_path)
        if input_image.shape != target.shape:
            raise ValueError(
                f"Paired image sizes differ for {sample.sample_id}: "
                f"input {tuple(input_image.shape)}, target {tuple(target.shape)}"
            )
        if self.training:
            input_image, target = self._augment(input_image, target)
        item: Dict[str, object] = {
            "input": input_image,
            "target": target,
            "filename": sample.filename,
            "sample_id": sample.sample_id,
            "group_id": sample.group_id,
        }
        if self.use_scene_condition:
            item["scene_id"] = torch.tensor(sample.scene_id, dtype=torch.long)
        return item

    def _augment(self, input_image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        patch = self.patch_size
        _, height, width = target.shape
        pad_h, pad_w = max(0, patch - height), max(0, patch - width)
        if pad_h or pad_w:
            if height < 2 or width < 2:
                raise ValueError("Reflect padding requires paired images with height and width >= 2")
            input_image = torch.from_numpy(
                np.pad(input_image.permute(1, 2, 0).numpy(), ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            ).permute(2, 0, 1)
            target = torch.from_numpy(
                np.pad(target.permute(1, 2, 0).numpy(), ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            ).permute(2, 0, 1)
            _, height, width = target.shape
        top = random.randint(0, height - patch)
        left = random.randint(0, width - patch)
        input_image = input_image[:, top:top + patch, left:left + patch]
        target = target[:, top:top + patch, left:left + patch]
        if random.random() < 0.5:
            input_image, target = input_image.flip(2), target.flip(2)
        if random.random() < 0.5:
            input_image, target = input_image.flip(1), target.flip(1)
        turns = random.randint(0, 3)
        if turns:
            input_image = torch.rot90(input_image, turns, (1, 2))
            target = torch.rot90(target, turns, (1, 2))
        return input_image.contiguous(), target.contiguous()
