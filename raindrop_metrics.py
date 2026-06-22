"""Y-channel PSNR/SSIM and AlexNet LPIPS validation metrics."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def rgb_to_y(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"Expected RGB BCHW input, got {tuple(image.shape)}")
    coefficients = image.new_tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1) / 255.0
    return (image * coefficients).sum(dim=1, keepdim=True) + 16.0 / 255.0


def psnr_y(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(rgb_to_y(prediction), rgb_to_y(target))
    if mse == 0:
        return torch.tensor(float("inf"), device=mse.device)
    return -10.0 * torch.log10(mse)


def ssim_y(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x, y = rgb_to_y(prediction), rgb_to_y(target)
    min_side = min(x.shape[-2:])
    window_size = min(11, min_side if min_side % 2 else min_side - 1)
    if window_size < 3:
        raise ValueError("SSIM requires images with height and width >= 3")
    sigma = 1.5 * window_size / 11.0
    coords = torch.arange(window_size, device=x.device, dtype=x.dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d)[None, None]
    padding = window_size // 2

    def filter_image(value):
        return F.conv2d(F.pad(value, (padding,) * 4, mode="reflect"), kernel)

    mu_x, mu_y = filter_image(x), filter_image(y)
    mu_x_sq, mu_y_sq, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x = filter_image(x * x) - mu_x_sq
    sigma_y = filter_image(y * y) - mu_y_sq
    sigma_xy = filter_image(x * y) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    value = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    )
    return value.mean()


class ValidationMetrics:
    def __init__(self, device: torch.device):
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError(
                "LPIPS is required for validation. Install requirements-raindrop.txt."
            ) from exc
        self.lpips = lpips.LPIPS(net="alex").to(device).eval()
        for parameter in self.lpips.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def compute(self, prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        prediction, target = prediction.clamp(0, 1), target.clamp(0, 1)
        psnr = psnr_y(prediction, target)
        ssim = ssim_y(prediction, target)
        perceptual = self.lpips(prediction * 2 - 1, target * 2 - 1).mean()
        values = {"PSNR_Y": float(psnr.cpu()), "SSIM_Y": float(ssim.cpu()), "LPIPS": float(perceptual.cpu())}
        if not all(torch.isfinite(torch.tensor(value)) for value in values.values()):
            raise FloatingPointError(f"Validation metric contains NaN or Inf: {values}")
        return values


def aggregate_metrics(totals: Dict[str, float], count: int) -> Dict[str, float]:
    if count <= 0:
        raise ValueError("Cannot aggregate an empty validation set")
    result = {key: value / count for key, value in totals.items()}
    result["Score"] = result["PSNR_Y"] + 10.0 * result["SSIM_Y"] - 5.0 * result["LPIPS"]
    return result

