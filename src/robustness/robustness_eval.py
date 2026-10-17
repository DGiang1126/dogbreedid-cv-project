"""Robustness evaluation against the approved corruption protocol — E8.

Usage (synthetic / tooling-only):
    evaluator = RobustnessEvaluator(model, cfg)
    results = evaluator.evaluate(images, labels)   # images: [N, C, H, W] in [0,1]
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.robustness.config import E8Config
from src.robustness.corruptions import (
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_occlusion,
)

# ImageNet normalization constants (applied inside evaluator after corruption)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize [N, 3, H, W] tensor in [0,1] with ImageNet stats."""
    mean = _MEAN.to(images.device)
    std = _STD.to(images.device)
    return (images - mean) / std


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy as fraction in [0, 1]."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


class RobustnessEvaluator:
    """Evaluate accuracy drop under approved corruptions.

    Args:
        model: nn.Module, already in eval mode. Weights never updated.
        cfg: E8Config loaded from e8_robustness.yaml.
    """

    def __init__(self, model: nn.Module, cfg: E8Config) -> None:
        if cfg.official_evaluation_enabled:
            raise RuntimeError("Official evaluation must stay disabled in Task 03.")
        self.model = model
        self.cfg = cfg

    @torch.inference_mode()
    def evaluate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, Any]:
        """Run clean + all enabled corruptions; return metrics dict.

        Args:
            images: Float32 [N, C, H, W] in [0, 1], pre-cropped, NOT normalized.
                    Corruption is applied here before normalization.
            labels: Long [N] class indices.

        Returns:
            Dict with keys: clean_accuracy, per-corruption accuracy + drop, conditions.
            All accuracy values are fractions in [0, 1].
        """
        if self.cfg.restrictions.use_official_test:
            raise RuntimeError("Official Test use is forbidden.")

        self.model.eval()
        device = next(self.model.parameters()).device
        images = images.to(device)
        labels = labels.to(device)

        # ── clean ─────────────────────────────────────────────────────────────
        clean_acc = _accuracy(self.model(_normalize(images)), labels)
        results: dict[str, Any] = {
            "clean_accuracy": clean_acc,
            "conditions": ["clean"],
        }

        # ── Gaussian Blur ─────────────────────────────────────────────────────
        if self.cfg.gaussian_blur.enabled:
            bc = self.cfg.gaussian_blur
            corrupted = torch.stack([
                apply_gaussian_blur(img, kernel_size=bc.kernel_size, sigma=bc.sigma)
                for img in images
            ])
            acc = _accuracy(self.model(_normalize(corrupted)), labels)
            results["gaussian_blur_accuracy"] = acc
            results["gaussian_blur_accuracy_drop"] = clean_acc - acc
            results["conditions"].append("gaussian_blur")

        # ── Gaussian Noise ────────────────────────────────────────────────────
        if self.cfg.gaussian_noise.enabled:
            nc = self.cfg.gaussian_noise
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.cfg.reproducibility.seed)
            corrupted = torch.stack([
                apply_gaussian_noise(
                    img.cpu(), mean=nc.mean, std=nc.std,
                    clamp_range=nc.clamp_range, generator=gen,
                ).to(device)
                for img in images
            ])
            acc = _accuracy(self.model(_normalize(corrupted)), labels)
            results["gaussian_noise_accuracy"] = acc
            results["gaussian_noise_accuracy_drop"] = clean_acc - acc
            results["conditions"].append("gaussian_noise")

        # ── Occlusion ─────────────────────────────────────────────────────────
        if self.cfg.occlusion.enabled:
            oc = self.cfg.occlusion
            corrupted = torch.stack([
                apply_occlusion(
                    img, side_fraction=oc.side_fraction,
                    fill_rgb=oc.fill_rgb, image_id=i,
                )
                for i, img in enumerate(images)
            ])
            acc = _accuracy(self.model(_normalize(corrupted)), labels)
            results["occlusion_accuracy"] = acc
            results["occlusion_accuracy_drop"] = clean_acc - acc
            results["conditions"].append("occlusion")

        return results

