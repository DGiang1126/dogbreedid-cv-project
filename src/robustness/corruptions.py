"""Image corruption functions used in E8 — Gaussian Blur, Gaussian Noise, Occlusion.

Corruption stage: after CenterCrop(224), before ImageNet normalization.
Input:  float32 tensor [C, H, W] with pixel values in [0, 1].
Output: float32 tensor [C, H, W] with pixel values in [0, 1].
"""
from __future__ import annotations

import hashlib
import json
import math

import torch
import torch.nn.functional as F


def _validate_image(image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor) or image.ndim != 3:
        raise ValueError("Expected 3-D tensor [C,H,W]")
    if image.shape[0] != 3 or min(image.shape[1:]) <= 0:
        raise ValueError("Expected a non-empty RGB tensor [3,H,W]")
    if (not image.is_floating_point() or not torch.isfinite(image).all()
            or image.min() < 0 or image.max() > 1):
        raise ValueError("Image must be floating point, finite and in [0,1] before normalization")


def per_image_seed(global_seed: int, image_id: str, corruption_name: str) -> int:
    """Stable uint64 seed from SHA256([global_seed, image_id, corruption_name]).

    JSON framing avoids concatenation collisions. IDs must be stable non-empty
    strings (e.g. dataset-relative paths), never positions within a batch.
    """
    if type(global_seed) is not int or not 0 <= global_seed < 2**64:
        raise ValueError("global_seed must be an integer in [0, 2**64)")
    if not isinstance(image_id, str) or not image_id.strip():
        raise ValueError("image_id must be a non-empty stable string, e.g. a relative path")
    if not isinstance(corruption_name, str) or not corruption_name.strip():
        raise ValueError("corruption_name must be a non-empty string")
    payload = json.dumps([global_seed, image_id, corruption_name], ensure_ascii=True,
                         separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def apply_gaussian_blur(
    image: torch.Tensor,
    kernel_size: int = 7,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Apply Gaussian Blur to a single image tensor [C, H, W] in [0, 1].

    Args:
        image: Float32 tensor [C, H, W], pixel values in [0, 1].
        kernel_size: Odd integer kernel side length. Default 7 (approved).
        sigma: Gaussian standard deviation. Default 1.5 (approved).

    Returns:
        Blurred float32 tensor [C, H, W], pixel values in [0, 1].
    """
    _validate_image(image)
    if type(kernel_size) is not int or kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    if type(sigma) not in (int, float) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    half = kernel_size // 2
    if half >= min(image.shape[1:]):
        raise ValueError("kernel_size is too large for reflection padding of this image")
    coords = torch.arange(kernel_size, dtype=image.dtype, device=image.device) - half
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)  # [k, k]
    C = image.shape[0]
    kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)

    inp = image.unsqueeze(0)  # [1, C, H, W]
    inp = F.pad(inp, [half, half, half, half], mode="reflect")
    out = F.conv2d(inp, kernel_4d, groups=C)  # [1, C, H, W]
    return out.squeeze(0).clamp(0.0, 1.0)


def apply_gaussian_noise(
    image: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.05,
    clamp_range: tuple[float, float] = (0.0, 1.0),
    *,
    global_seed: int,
    image_id: str,
) -> torch.Tensor:
    """Add Gaussian noise to a single image tensor [C, H, W] in [0, 1].

    Args:
        image: Float32 tensor [C, H, W], pixel values in [0, 1].
        mean: Noise mean. Default 0.0 (approved).
        std: Noise standard deviation. Default 0.05 (approved).
        clamp_range: Output pixel clamp bounds. Default (0.0, 1.0).
        global_seed: Seed from cfg.reproducibility.seed.
        image_id: Stable non-empty string identifier, e.g. a relative path.

    Returns:
        Noisy float32 tensor [C, H, W], clamped to clamp_range.
    """
    _validate_image(image)
    if (type(std) not in (int, float) or not math.isfinite(std) or std < 0
            or type(mean) not in (int, float) or not math.isfinite(mean)):
        raise ValueError("mean must be finite and std must be finite and non-negative")
    if (not isinstance(clamp_range, (tuple, list)) or len(clamp_range) != 2 or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in clamp_range
    ) or not 0 <= clamp_range[0] < clamp_range[1] <= 1):
        raise ValueError("clamp_range must have 0 <= lower < upper <= 1")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(per_image_seed(global_seed, image_id, "gaussian_noise"))
    # CPU stream makes the random realization independent of the model device.
    noise = torch.empty(image.shape, dtype=image.dtype, device="cpu").normal_(
        mean=mean, std=std, generator=generator,
    ).to(image.device)
    return (image + noise).clamp(clamp_range[0], clamp_range[1])


def apply_occlusion(
    image: torch.Tensor,
    side_fraction: float = 0.30,
    fill_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5),
    *,
    global_seed: int,
    image_id: str,
) -> torch.Tensor:
    """Apply square occlusion patch to a single image tensor [C, H, W] in [0, 1].

    Position is deterministic per image_id (approved policy: deterministic_per_image).
    Per-image seed derived via SHA-256(global_seed + image_id + corruption_name)
    so it is stable across sessions and independent of processing order.

    Args:
        image: Float32 tensor [C, H, W], pixel values in [0, 1].
        side_fraction: Square side as fraction of min(H, W). Default 0.30 (approved).
        fill_rgb: RGB fill color, each in [0, 1]. Default (0.5, 0.5, 0.5) (approved).
        global_seed: Seed from cfg.reproducibility.seed.
        image_id: Stable non-empty string identifier for deterministic position.

    Returns:
        Occluded float32 tensor [C, H, W].
    """
    _validate_image(image)
    if (type(side_fraction) not in (int, float) or not math.isfinite(side_fraction)
            or not 0.0 < side_fraction < 1.0):
        raise ValueError(f"side_fraction must be in (0,1), got {side_fraction}")
    if not isinstance(fill_rgb, (tuple, list)) or len(fill_rgb) != 3 or any(
        type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
        for v in fill_rgb
    ):
        raise ValueError("fill_rgb must contain exactly three finite numbers in [0,1]")

    _, H, W = image.shape
    side = max(int(min(H, W) * side_fraction), 1)

    rng = torch.Generator()
    rng.manual_seed(per_image_seed(global_seed, image_id, "occlusion"))
    x0 = int(torch.randint(0, W - side + 1, (1,), generator=rng).item())
    y0 = int(torch.randint(0, H - side + 1, (1,), generator=rng).item())
    x1, y1 = x0 + side, y0 + side

    out = image.clone()
    for c, val in enumerate(fill_rgb):
        out[c, y0:y1, x0:x1] = float(val)
    return out

