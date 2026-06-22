#!/usr/bin/env python3
"""Create CSV/Markdown comparison and Scene-minus-baseline deltas."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-scene", required=True, help="No-scene metrics JSON")
    parser.add_argument("--scene", required=True, help="Scene metrics JSON")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    baseline = json.loads(Path(args.no_scene).read_text(encoding="utf-8"))
    scene = json.loads(Path(args.scene).read_text(encoding="utf-8"))
    metrics = ["PSNR_Y", "SSIM_Y", "LPIPS", "Score"]
    rows = [
        {"model": baseline["model"], **{key: baseline[key] for key in metrics}},
        {"model": scene["model"], **{key: scene[key] for key in metrics}},
        {"model": "Scene - no-scene", **{key: scene[key] - baseline[key] for key in metrics}},
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scene_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *metrics])
        writer.writeheader()
        writer.writerows(rows)
    header = "| model | PSNR_Y | SSIM_Y | LPIPS | Score |\n|---|---:|---:|---:|---:|\n"
    body = "".join(
        f"| {row['model']} | {row['PSNR_Y']:.4f} | {row['SSIM_Y']:.4f} | "
        f"{row['LPIPS']:.4f} | {row['Score']:.4f} |\n"
        for row in rows
    )
    (output_dir / "scene_ablation.md").write_text(header + body, encoding="utf-8")
    print(header + body)


if __name__ == "__main__":
    main()

