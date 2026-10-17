"""Image corruption functions used in E8 — Gaussian Blur, Gaussian Noise, Occlusion.

Corruption stage: after CenterCrop(224), before ImageNet normalization.
Input:  float32 tensor [C, H, W] with pixel values in [0, 1].
Output: float32 tensor [C, H, W] with pixel values in [0, 1].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


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
    if image.ndim != 3:
        raise ValueError(f"Expected 3-D tensor [C,H,W], got shape {tuple(image.shape)}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    half = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float32, device=image.device) - half
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
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add Gaussian noise to a single image tensor [C, H, W] in [0, 1].

    Args:
        image: Float32 tensor [C, H, W], pixel values in [0, 1].
        mean: Noise mean. Default 0.0 (approved).
        std: Noise standard deviation. Default 0.05 (approved).
        clamp_range: Output pixel clamp bounds. Default (0.0, 1.0).
        generator: Optional torch.Generator for reproducibility.

    Returns:
        Noisy float32 tensor [C, H, W], clamped to clamp_range.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3-D tensor [C,H,W], got shape {tuple(image.shape)}")
    if std < 0:
        raise ValueError(f"std must be non-negative, got {std}")

    noise = torch.empty_like(image).normal_(mean=mean, std=std, generator=generator)
    return (image + noise).clamp(clamp_range[0], clamp_range[1])


def apply_occlusion(
    image: torch.Tensor,
    side_fraction: float = 0.30,
    fill_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5),
    image_id: int = 0,
) -> torch.Tensor:
    """Apply square occlusion patch to a single image tensor [C, H, W] in [0, 1].

    Position is deterministic per image_id (approved policy: deterministic_per_image).
    Per-image seed derived via SHA-256(global_seed=42 + image_id + corruption_name)
    so it is stable across sessions and independent of processing order.

    Args:
        image: Float32 tensor [C, H, W], pixel values in [0, 1].
        side_fraction: Square side as fraction of min(H, W). Default 0.30 (approved).
        fill_rgb: RGB fill color, each in [0, 1]. Default (0.5, 0.5, 0.5) (approved).
        image_id: Stable integer identifier for deterministic position.

    Returns:
        Occluded float32 tensor [C, H, W].
    """
    import hashlib

    if image.ndim != 3:
        raise ValueError(f"Expected 3-D tensor [C,H,W], got shape {tuple(image.shape)}")
    if not (0.0 < side_fraction < 1.0):
        raise ValueError(f"side_fraction must be in (0,1), got {side_fraction}")

    C, H, W = image.shape
    side = max(int(min(H, W) * side_fraction), 1)

    # Per-image seed: SHA-256(global_seed + image_id + corruption_name)
    # Stable across Python sessions; does not use hash()
    seed_str = f"42:{image_id}:occlusion"
    digest = hashlib.sha256(seed_str.encode()).digest()
    seed_int = int.from_bytes(digest[:4], "big")  # 32-bit, always positive

    rng = torch.Generator()
    rng.manual_seed(seed_int)
    cx = int(torch.randint(side // 2, W - side // 2, (1,), generator=rng).item())
    cy = int(torch.randint(side // 2, H - side // 2, (1,), generator=rng).item())

    x0, y0 = max(cx - side // 2, 0), max(cy - side // 2, 0)
    x1, y1 = min(x0 + side, W), min(y0 + side, H)

    out = image.clone()
    for c, val in enumerate(fill_rgb):
        out[c, y0:y1, x0:x1] = float(val)
    return out

