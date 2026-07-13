"""Shared configuration, checkpoint, loss, and inference helpers."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

import losses
from model import MSDT


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def resolve_device(device: Optional[str] = None) -> torch.device:
    selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(selected)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {selected}")
    return result


def build_model(config: Dict[str, Any]) -> MSDT:
    model_config = config.get("model", {})
    return MSDT(
        num_res=int(model_config.get("num_res", 8)),
        use_scene_condition=bool(config["experiment"]["use_scene_condition"]),
        scene_embedding_dim=int(model_config.get("scene_embedding_dim", 32)),
    )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key.removeprefix("module.").removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    expected_use_scene: Optional[bool] = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    unwrap_model(model).load_state_dict(normalize_state_dict(state_dict), strict=True)
    if expected_use_scene is not None and "use_scene_condition" in checkpoint:
        actual = bool(checkpoint["use_scene_condition"])
        if actual != expected_use_scene:
            raise ValueError(
                f"Checkpoint scene mode ({actual}) does not match config ({expected_use_scene})"
            )
    return checkpoint


def gaussian_pyramid(target: torch.Tensor, levels: int = 3) -> List[torch.Tensor]:
    if levels < 1:
        raise ValueError("Pyramid must contain at least one level")
    kernel_1d = target.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    kernel = torch.outer(kernel_1d, kernel_1d)[None, None]
    kernel = kernel.repeat(target.shape[1], 1, 1, 1)
    pyramid = [target]
    current = target
    for _ in range(1, levels):
        current = F.pad(current, (2, 2, 2, 2), mode="reflect")
        current = F.conv2d(current, kernel, groups=target.shape[1])[:, :, ::2, ::2]
        pyramid.append(current)
    return pyramid


def validate_outputs(outputs: Sequence[torch.Tensor], input_image: torch.Tensor) -> Dict[str, float]:
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 3:
        raise RuntimeError("MSDT must return exactly three outputs: full, half, and quarter resolution")
    height, width = input_image.shape[-2:]
    expected = [(height, width), (height // 2, width // 2), (height // 4, width // 4)]
    mins, maxs = [], []
    for index, (output, spatial) in enumerate(zip(outputs, expected)):
        if output.shape[:2] != input_image.shape[:2] or tuple(output.shape[-2:]) != spatial:
            raise RuntimeError(
                f"MSDT output[{index}] has shape {tuple(output.shape)}, expected "
                f"{(input_image.shape[0], input_image.shape[1], *spatial)}"
            )
        if not torch.isfinite(output).all():
            raise FloatingPointError(f"MSDT output[{index}] contains NaN or Inf")
        mins.append(float(output.detach().amin().cpu()))
        maxs.append(float(output.detach().amax().cpu()))
    if min(mins) < -1e4 or max(maxs) > 1e4:
        raise FloatingPointError(f"MSDT output range is unstable: [{min(mins)}, {max(maxs)}]")
    return {"raw_min": min(mins), "raw_max": max(maxs)}


class MultiScaleMSDTLoss(torch.nn.Module):
    """Official three-scale objective and component weights."""
    def __init__(self, fft_weight: float = 0.01, edge_weight: float = 0.05):
        super().__init__()
        self.charbonnier = losses.CharbonnierLoss()
        self.edge = losses.EdgeLoss()
        self.fft = losses.fftLoss()
        self.fft_weight = fft_weight
        self.edge_weight = edge_weight

    def forward(self, outputs: Sequence[torch.Tensor], target: torch.Tensor):
        targets = gaussian_pyramid(target, 3)
        for index, (output, scaled_target) in enumerate(zip(outputs, targets)):
            if output.shape != scaled_target.shape:
                raise RuntimeError(
                    f"Output/GT mismatch at scale {index}: {tuple(output.shape)} vs {tuple(scaled_target.shape)}"
                )
        char = sum(self.charbonnier(output, gt) for output, gt in zip(outputs, targets))
        fft = target.new_tensor(0.0)
        edge = target.new_tensor(0.0)
        if self.fft_weight > 0:
            fft = sum(self.fft(output, gt) for output, gt in zip(outputs, targets))
        if self.edge_weight > 0:
            edge = sum(self.edge(output, gt) for output, gt in zip(outputs, targets))
        total = char + self.fft_weight * fft + self.edge_weight * edge
        return total, {"charbonnier": char.detach(), "fft": fft.detach(), "edge": edge.detach()}


def _reflect_pad(image: torch.Tensor, pad_right: int, pad_bottom: int) -> torch.Tensor:
    if pad_right == 0 and pad_bottom == 0:
        return image
    if image.shape[-2] < 2 or image.shape[-1] < 2:
        raise ValueError("Reflect padding requires image height and width >= 2")
    result = image
    remaining_right, remaining_bottom = pad_right, pad_bottom
    while remaining_right or remaining_bottom:
        step_right = min(remaining_right, result.shape[-1] - 1)
        step_bottom = min(remaining_bottom, result.shape[-2] - 1)
        result = F.pad(result, (0, step_right, 0, step_bottom), mode="reflect")
        remaining_right -= step_right
        remaining_bottom -= step_bottom
    return result


def pad_to_multiple(image: torch.Tensor, multiple: int = 4) -> Tuple[torch.Tensor, Tuple[int, int]]:
    height, width = image.shape[-2:]
    pad_bottom = (-height) % multiple
    pad_right = (-width) % multiple
    return _reflect_pad(image, pad_right, pad_bottom), (height, width)


def _model_forward(model, image: torch.Tensor, scene_id: Optional[torch.Tensor]):
    return model(image, scene_id=scene_id) if scene_id is not None else model(image)


def infer_image(
    model: torch.nn.Module,
    image: torch.Tensor,
    scene_id: Optional[torch.Tensor] = None,
    tile_size: int = 0,
    tile_overlap: int = 32,
    tile_stride: Optional[int] = None,
    multiple: int = 4,
    scales: Sequence[float] = (1.0,),
    vflip: bool = False,
    hflip: bool = False,
    rot90: bool = False,
    rot180: bool = False,
    rot270: bool = False,
) -> torch.Tensor:
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Inference expects BCHW with batch size 1, got {tuple(image.shape)}")
    scale_values = tuple(float(value) for value in scales)
    if not scale_values:
        raise ValueError("scales must contain at least one value")
    if any(value <= 0 for value in scale_values):
        raise ValueError(f"scales must be positive, got {scale_values}")

    def infer_once(input_image: torch.Tensor) -> torch.Tensor:
        padded, original_size = pad_to_multiple(input_image, multiple)
        height, width = padded.shape[-2:]
        if not tile_size or tile_size >= max(height, width):
            outputs = _model_forward(model, padded, scene_id)
            validate_outputs(outputs, padded)
            restored_once = outputs[0]
        else:
            if tile_size % multiple:
                raise ValueError(f"tile_size must be divisible by {multiple}")
            if tile_stride is None:
                if tile_overlap < 0 or tile_overlap >= tile_size:
                    raise ValueError("tile_overlap must be >= 0 and smaller than tile_size")
                stride = tile_size - tile_overlap
            else:
                if tile_stride <= 0 or tile_stride > tile_size:
                    raise ValueError("tile_stride must be > 0 and <= tile_size")
                stride = tile_stride
            if tile_size > height or tile_size > width:
                extra_bottom, extra_right = max(0, tile_size - height), max(0, tile_size - width)
                padded = _reflect_pad(padded, extra_right, extra_bottom)
                height, width = padded.shape[-2:]
            ys = list(range(0, max(1, height - tile_size + 1), stride))
            xs = list(range(0, max(1, width - tile_size + 1), stride))
            if ys[-1] != height - tile_size:
                ys.append(height - tile_size)
            if xs[-1] != width - tile_size:
                xs.append(width - tile_size)
            window_1d = torch.hann_window(tile_size, periodic=False, device=input_image.device, dtype=input_image.dtype)
            window = torch.outer(window_1d, window_1d).clamp_min(1e-3)[None, None]
            accumulation = torch.zeros_like(padded)
            weights = torch.zeros_like(padded[:, :1])
            for top in ys:
                for left in xs:
                    tile = padded[:, :, top:top + tile_size, left:left + tile_size]
                    tile_outputs = _model_forward(model, tile, scene_id)
                    validate_outputs(tile_outputs, tile)
                    accumulation[:, :, top:top + tile_size, left:left + tile_size] += tile_outputs[0] * window
                    weights[:, :, top:top + tile_size, left:left + tile_size] += window
            restored_once = accumulation / weights.clamp_min(1e-6)
        return restored_once[:, :, :original_size[0], :original_size[1]]

    def infer_scaled(input_image: torch.Tensor, scale: float) -> torch.Tensor:
        if math.isclose(scale, 1.0):
            return infer_once(input_image)
        height, width = input_image.shape[-2:]
        scaled_size = (max(2, round(height * scale)), max(2, round(width * scale)))
        scaled = F.interpolate(input_image, size=scaled_size, mode="bilinear", align_corners=False)
        restored_scaled = infer_once(scaled)
        return F.interpolate(restored_scaled, size=(height, width), mode="bilinear", align_corners=False)

    transforms = [(lambda value: value, lambda value: value)]
    if vflip:
        transforms.append((lambda value: value.flip(2), lambda value: value.flip(2)))
    if hflip:
        transforms.append((lambda value: value.flip(3), lambda value: value.flip(3)))
    if rot90:
        transforms.append((
            lambda value: torch.rot90(value, 1, (2, 3)),
            lambda value: torch.rot90(value, -1, (2, 3)),
        ))
    if rot180:
        transforms.append((
            lambda value: torch.rot90(value, 2, (2, 3)),
            lambda value: torch.rot90(value, -2, (2, 3)),
        ))
    if rot270:
        transforms.append((
            lambda value: torch.rot90(value, 3, (2, 3)),
            lambda value: torch.rot90(value, -3, (2, 3)),
        ))

    predictions = []
    for scale in scale_values:
        for apply_transform, invert_transform in transforms:
            predictions.append(invert_transform(infer_scaled(apply_transform(image), scale)))
    restored = torch.stack(predictions, dim=0).mean(dim=0).clamp(0.0, 1.0)
    if not torch.isfinite(restored).all() or restored.min() < 0 or restored.max() > 1:
        raise FloatingPointError("Final inference output failed finite/range checks")
    return restored


def tensor_to_png(image: torch.Tensor, path: str) -> None:
    array = image.detach().squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy()
    array = np.rint(array * 255.0).astype(np.uint8)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(output_path, format="PNG", compress_level=6)


def append_metrics_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
