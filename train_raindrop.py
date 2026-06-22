#!/usr/bin/env python3
"""Train one MSDT RaindropClarity baseline (with or without scene FiLM)."""

from __future__ import annotations

import argparse
import json
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from raindrop_engine import prepare_datasets, validate_model
from raindrop_metrics import ValidationMetrics
from raindrop_utils import (
    MultiScaleMSDTLoss,
    append_metrics_csv,
    build_model,
    load_config,
    load_model_checkpoint,
    resolve_device,
    set_seed,
    unwrap_model,
    validate_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--scene-json")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", help="Resume a complete training checkpoint")
    parser.add_argument("--device", help="e.g. cuda, cuda:1, or cpu")
    parser.add_argument("--epochs", type=int, help="Override epochs (primarily for smoke tests)")
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Stop this invocation early while preserving the full cosine schedule",
    )
    parser.add_argument("--num-workers", type=int, help="Override DataLoader workers")
    parser.add_argument("--batch-size", type=int, help="Override training batch size")
    parser.add_argument("--lr", type=float, help="Override Adam learning rate")
    parser.add_argument("--max-train-steps", type=int, help="Limit train steps per epoch for smoke tests")
    parser.add_argument("--max-val-images", type=int, help="Limit validation images for smoke tests")
    return parser.parse_args()


def _worker_seed(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)


def _checkpoint_state(model, optimizer, scheduler, scaler, epoch, best_score, config, generator):
    state = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "use_scene_condition": bool(config["experiment"]["use_scene_condition"]),
        "config": config,
        "rng_state": {
            "python": __import__("random").getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "loader_generator": generator.get_state(),
        },
    }
    if torch.cuda.is_available():
        state["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()
    return state


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers must be >= 0")
        config["data"]["num_workers"] = args.num_workers
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be > 0")
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        if args.lr <= 0:
            raise ValueError("--lr must be > 0")
        config["optimizer"]["lr"] = args.lr
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError("--max-train-steps must be > 0")
    if args.max_val_images is not None and args.max_val_images <= 0:
        raise ValueError("--max-val-images must be > 0")
    seed = int(config["training"].get("seed", 1234))
    set_seed(seed)
    device = resolve_device(args.device)
    use_scene = bool(config["experiment"]["use_scene_condition"])
    output_dir = Path(args.output_dir or config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, data_info = prepare_datasets(config, args.data_root, args.scene_json)
    if args.max_val_images is not None:
        val_dataset = torch.utils.data.Subset(
            val_dataset, range(min(args.max_val_images, len(val_dataset)))
        )
    print(json.dumps({"device": str(device), **data_info}, ensure_ascii=False))
    generator = torch.Generator().manual_seed(seed)
    workers = int(config["data"].get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"].get("batch_size", 1)),
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_seed,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device)
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_config.get("lr", 1e-4)),
        betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
        eps=float(optimizer_config.get("eps", 1e-8)),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )
    epochs = int(config["training"].get("epochs", 200))
    stop_after_epoch = args.stop_after_epoch if args.stop_after_epoch is not None else epochs
    if stop_after_epoch <= 0 or stop_after_epoch > epochs:
        raise ValueError(
            f"--stop-after-epoch must be in [1, {epochs}], got {stop_after_epoch}"
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(optimizer_config.get("min_lr", 1e-6))
    )
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    amp_dtype_name = str(config["training"].get("amp_dtype", "float16")).lower()
    amp_dtypes = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if amp_dtype_name not in amp_dtypes:
        raise ValueError(f"training.amp_dtype must be float16 or bfloat16, got {amp_dtype_name!r}")
    amp_dtype = amp_dtypes[amp_dtype_name]
    if amp_enabled and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 AMP was requested but the selected CUDA device does not support BF16")
    # BF16 has FP32-like exponent range and does not need gradient scaling.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    print(f"AMP enabled={amp_enabled}, dtype={amp_dtype_name}, grad_scaler={scaler.is_enabled()}")
    criterion = MultiScaleMSDTLoss(
        fft_weight=float(config["loss"].get("fft_weight", 0.01)),
        edge_weight=float(config["loss"].get("edge_weight", 0.05)),
    ).to(device)
    metrics = ValidationMetrics(device)

    start_epoch, best_score = 1, float("-inf")
    if args.resume:
        checkpoint = load_model_checkpoint(model, args.resume, device, use_scene)
        for required in ("optimizer", "scheduler", "scaler", "epoch", "best_score"):
            if required not in checkpoint:
                raise ValueError(f"Resume checkpoint is missing {required!r}: {args.resume}")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        rng = checkpoint.get("rng_state")
        if rng:
            __import__("random").setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            generator.set_state(rng["loader_generator"])
            if device.type == "cuda" and "cuda" in rng:
                torch.cuda.set_rng_state_all(rng["cuda"])
        print(f"Resumed {args.resume} at epoch {start_epoch}")

    if start_epoch > stop_after_epoch:
        raise ValueError(
            f"Checkpoint starts at epoch {start_epoch}, after requested stop epoch {stop_after_epoch}"
        )

    print(
        f"Training epochs {start_epoch}..{stop_after_epoch}; "
        f"cosine schedule remains configured for {epochs} total epochs"
    )
    for epoch in range(start_epoch, stop_after_epoch + 1):
        model.train()
        epoch_loss = 0.0
        train_steps = 0
        smoke_grad_norm = 0.0
        smoke_parameter_delta = 0.0
        started = time.time()
        if args.max_train_steps is not None:
            epoch_step_limit = min(args.max_train_steps, len(train_loader))
            epoch_batches = islice(train_loader, epoch_step_limit)
        else:
            epoch_step_limit = len(train_loader)
            epoch_batches = train_loader
        for batch in tqdm(
            epoch_batches, total=epoch_step_limit, desc=f"train {epoch}/{epochs}"
        ):
            target = batch["target"].to(device, non_blocking=True)
            image = batch["input"].to(device, non_blocking=True)
            scene_id = batch["scene_id"].to(device) if use_scene else None
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                outputs = model(image, scene_id=scene_id) if use_scene else model(image)
                validate_outputs(outputs, image)
                loss, _ = criterion(outputs, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {float(loss)}")
            scaler.scale(loss).backward()
            probe_parameter = None
            probe_before = None
            if args.max_train_steps is not None:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float("inf")
                )
                if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
                    raise FloatingPointError(
                        f"Smoke test produced an invalid gradient norm: {float(grad_norm)}"
                    )
                smoke_grad_norm = max(smoke_grad_norm, float(grad_norm.detach().cpu()))
                for parameter in model.parameters():
                    if parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0:
                        probe_parameter = parameter
                        probe_before = parameter.detach().clone()
                        break
                if probe_parameter is None:
                    raise RuntimeError("Smoke test found no parameter with a non-zero gradient")
            scaler.step(optimizer)
            scaler.update()
            if probe_parameter is not None:
                parameter_delta = float(
                    (probe_parameter.detach() - probe_before).abs().max().cpu()
                )
                if not parameter_delta > 0:
                    raise RuntimeError("Optimizer step did not update the probed model parameter")
                smoke_parameter_delta = max(smoke_parameter_delta, parameter_delta)
            epoch_loss += float(loss.detach().cpu())
            train_steps += 1

        validation = validate_model(
            model,
            val_loader,
            metrics,
            device,
            use_scene,
            tile_size=int(config.get("inference", {}).get("validation_tile_size", 0)),
            tile_overlap=int(config.get("inference", {}).get("tile_overlap", 32)),
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": epoch_loss / train_steps,
            "lr": optimizer.param_groups[0]["lr"],
            **validation,
        }
        append_metrics_csv(output_dir / "metrics.csv", row)
        is_best = validation["Score"] > best_score
        if is_best:
            best_score = validation["Score"]
        state = _checkpoint_state(
            model, optimizer, scheduler, scaler, epoch, best_score, config, generator
        )
        torch.save(state, output_dir / "model_latest.pth")
        if is_best:
            torch.save(state, output_dir / "model_best.pth")
        log_payload = {
            **row,
            "best_score": best_score,
            "train_steps": train_steps,
            "seconds": time.time() - started,
        }
        if args.max_train_steps is not None:
            log_payload.update(
                {
                    "smoke_grad_norm": smoke_grad_norm,
                    "smoke_parameter_delta": smoke_parameter_delta,
                    "optimizer_update_verified": True,
                }
            )
        print(json.dumps(log_payload))


if __name__ == "__main__":
    main()
