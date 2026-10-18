"""E8 tooling: paired clean/corrupted inference on pre-normalized RGB tensors.

Usage: RobustnessEvaluator(model, cfg).evaluate(images, labels, image_ids)
IDs must come from the dataset (e.g. relative paths), independently of batching.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from src.robustness.config import E8Config
from src.robustness.corruptions import (
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_occlusion,
    per_image_seed,
)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _normalize(images: torch.Tensor) -> torch.Tensor:
    return (images - _MEAN.to(images)) / _STD.to(images)


class RobustnessEvaluator:
    """Run inference only; report accuracies and signed clean-minus-corrupted drops."""

    def __init__(self, model: nn.Module, cfg: E8Config) -> None:
        if (cfg.official_evaluation_enabled or cfg.status != "tooling_only"
                or cfg.restrictions.use_official_test
                or cfg.restrictions.train_new_model or cfg.restrictions.modify_checkpoint):
            raise RuntimeError("E8 Task 03 is tooling-only; Official Test and weight updates are forbidden.")
        self.model = model
        self.cfg = cfg

    def _corrupt(self, images: torch.Tensor, image_ids: Sequence[str], name: str) -> torch.Tensor:
        if name == "clean":
            return images
        cfg = self.cfg
        if name == "gaussian_blur":
            return torch.stack([
                apply_gaussian_blur(img, cfg.gaussian_blur.kernel_size, cfg.gaussian_blur.sigma)
                for img in images
            ])
        if name == "gaussian_noise":
            nc = cfg.gaussian_noise
            return torch.stack([
                apply_gaussian_noise(img, nc.mean, nc.std, nc.clamp_range,
                                     global_seed=cfg.reproducibility.seed, image_id=image_id)
                for img, image_id in zip(images, image_ids)
            ])
        if name == "occlusion":
            oc = cfg.occlusion
            return torch.stack([
                apply_occlusion(img, oc.side_fraction, oc.fill_rgb,
                                global_seed=cfg.reproducibility.seed, image_id=image_id)
                for img, image_id in zip(images, image_ids)
            ])
        raise ValueError(f"Unsupported corruption condition: {name}")

    @torch.inference_mode()
    def evaluate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        image_ids: Sequence[str] | None = None,
        *,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate [N,3,H,W] in [0,1], with aligned labels and stable string IDs.

        Corruption precedes ImageNet normalization. batch_size only controls
        inference chunking; seeds never depend on chunk boundaries or positions.
        Accuracy is in [0,1]; signed accuracy drops can be in [-1,1].
        """
        if self.cfg.restrictions.use_official_test:
            raise RuntimeError("Official Test use is forbidden.")
        if (not isinstance(images, torch.Tensor) or images.ndim != 4
                or images.shape[0] == 0 or images.shape[1] != 3
                or min(images.shape[2:]) <= 0):
            raise ValueError("images must be non-empty [N,3,H,W] RGB tensors")
        if (not images.is_floating_point() or not torch.isfinite(images).all()
                or images.min() < 0 or images.max() > 1):
            raise ValueError("images must be finite floating tensors in [0,1] before normalization")
        n = len(images)
        if (not isinstance(labels, torch.Tensor) or labels.shape != (n,)
                or labels.dtype != torch.long or (labels < 0).any()):
            raise ValueError("labels must be non-negative Long [N] aligned with images")
        if (not isinstance(image_ids, Sequence) or isinstance(image_ids, (str, bytes))
                or len(image_ids) != n):
            raise ValueError("image_ids must contain one stable non-empty string per image")
        for image_id in image_ids:
            per_image_seed(self.cfg.reproducibility.seed, image_id, "validation")
        if len(set(image_ids)) != n:
            raise ValueError("image_ids must be unique within an evaluation")
        if batch_size is None:
            batch_size = n
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self.model.eval()
        reference = next(self.model.parameters(), None)
        if reference is None:
            reference = next(self.model.buffers(), None)
        device = reference.device if reference is not None else images.device
        accuracies = {}
        for name in self.cfg.conditions:
            correct = 0
            for start in range(0, n, batch_size):
                stop = start + batch_size
                batch = images[start:stop].to(device)
                targets = labels[start:stop].to(device)
                corrupted = self._corrupt(batch, image_ids[start:stop], name)
                logits = self.model(_normalize(corrupted))
                if (logits.ndim != 2 or logits.shape[0] != len(batch)
                        or logits.shape[1] == 0 or not torch.isfinite(logits).all()):
                    raise ValueError("model logits must be finite [N,num_classes]")
                if (targets >= logits.shape[1]).any():
                    raise ValueError("labels exceed the model's number of classes")
                correct += int((logits.argmax(dim=1) == targets).sum().item())
            accuracies[name] = correct / n

        clean = accuracies["clean"]
        results: dict[str, Any] = {"clean_accuracy": clean, "conditions": list(self.cfg.conditions)}
        for name, acc in accuracies.items():
            if name != "clean":
                results[f"{name}_accuracy"] = acc
                results[f"{name}_accuracy_drop"] = clean - acc
        return results
