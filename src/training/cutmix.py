"""CutMix augmentation utilities for the DogBreedID shared training pipeline.

CutMix is applied to training batches only. Validation must use the normal
(non-CutMix) path.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor, nn


def _validate_inputs(images: Tensor, targets: Tensor) -> None:
    if not isinstance(images, Tensor) or images.ndim != 4:
        raise ValueError("images must be a tensor with shape [B, C, H, W].")
    if not isinstance(targets, Tensor) or targets.ndim != 1:
        raise ValueError("targets must be a tensor with shape [B].")
    if images.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Batch mismatch: {images.shape[0]} images but {targets.shape[0]} targets."
        )
    if images.shape[0] == 0:
        raise ValueError("CutMix cannot process an empty batch.")
    if images.shape[2] < 1 or images.shape[3] < 1:
        raise ValueError("Image height and width must be positive.")


def _unchanged(images: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor, float, bool]:
    """Return the standard no-CutMix result using the same target twice."""
    return images, targets, targets, 1.0, False


def cutmix_batch(
    images: Tensor,
    targets: Tensor,
    alpha: float = 1.0,
    probability: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor, float, bool]:
    """Apply CutMix to an NCHW batch.

    Returns ``(mixed_images, targets_a, targets_b, lambda_actual, applied)``.
    ``lambda_actual`` is recomputed from the clipped rectangle's real pixel
    area, not reused from the initially sampled Beta value.

    The caller is responsible for invoking this function on training batches
    only. When CutMix is skipped, ``targets_a == targets_b`` and lambda is 1.
    Randomness follows PyTorch's global RNG and is controlled by the project's
    existing ``set_seed`` policy.
    """
    _validate_inputs(images, targets)
    alpha = float(alpha)
    probability = float(probability)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"alpha must be finite and > 0, got {alpha}.")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"probability must be in [0, 1], got {probability}."
        )

    batch_size, _, height, width = images.shape
    if probability == 0.0 or batch_size < 2:
        return _unchanged(images, targets)
    if torch.rand((), device=images.device).item() >= probability:
        return _unchanged(images, targets)

    # Draw the mixing ratio from Beta(alpha, alpha). Using torch's RNG means
    # the experiment remains reproducible under the shared seed policy.
    beta = torch.distributions.Beta(alpha, alpha)
    lam_sampled = float(beta.sample().item())
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam_sampled))
    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)

    center_x = int(torch.randint(width, size=(), device=images.device).item())
    center_y = int(torch.randint(height, size=(), device=images.device).item())

    x1 = max(center_x - cut_width // 2, 0)
    x2 = min(center_x + (cut_width - cut_width // 2), width)
    y1 = max(center_y - cut_height // 2, 0)
    y2 = min(center_y + (cut_height - cut_height // 2), height)

    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        return _unchanged(images, targets)

    permutation = torch.randperm(batch_size, device=images.device)
    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]

    replaced_area = box_width * box_height
    lambda_actual = 1.0 - (replaced_area / float(height * width))
    lambda_actual = min(1.0, max(0.0, float(lambda_actual)))
    return mixed_images, targets, targets[permutation], lambda_actual, True


def cutmix_criterion_loss(
    criterion: Callable[[Tensor, Tensor], Tensor] | nn.Module,
    logits: Tensor,
    targets_a: Tensor,
    targets_b: Tensor,
    lambda_actual: float,
    applied: bool,
) -> Tensor:
    """Compute ordinary CE or the lambda-weighted two-label CutMix loss."""
    if not applied:
        return criterion(logits, targets_a)
    lam = float(lambda_actual)
    if not math.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError(f"lambda_actual must be finite and in [0, 1], got {lam}.")
    return lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(
        logits, targets_b
    )
