"""Grad-CAM for classification models, including frozen-backbone E4 MobileNetV2.

The function does not update model weights. It uses gradients only to explain one
chosen class score. Do not call it from within torch.inference_mode().
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def find_last_conv2d(model: nn.Module) -> tuple[str, nn.Module]:
    """Return the last named Conv2d module, a sensible default for MobileNetV2."""
    candidates = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    if not candidates:
        raise ValueError("No nn.Conv2d layer was found in the model")
    return candidates[-1]


def resolve_target_layer(model: nn.Module, target_layer: str | nn.Module | None = None) -> nn.Module:
    """Resolve a module or dotted name; default to the last Conv2d layer."""
    if target_layer is None:
        return find_last_conv2d(model)[1]
    if isinstance(target_layer, nn.Module):
        return target_layer
    named = dict(model.named_modules())
    if target_layer not in named:
        candidates = [name for name in named if name.endswith(target_layer)]
        if len(candidates) == 1:
            return named[candidates[0]]
        raise ValueError(f"Target layer {target_layer!r} not found uniquely in model")
    return named[target_layer]


def _extract_logits(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        logits = model_output
    elif isinstance(model_output, dict) and isinstance(model_output.get("logits"), torch.Tensor):
        logits = model_output["logits"]
    elif isinstance(model_output, (tuple, list)) and model_output and isinstance(model_output[0], torch.Tensor):
        logits = model_output[0]
    else:
        raise TypeError("Model output must be a logits Tensor, a dict with 'logits', or a tuple/list starting with Tensor")
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2:
        raise ValueError(f"Expected logits shape [N,C], got {tuple(logits.shape)}")
    return logits


def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
    target_layer: str | nn.Module | None = None,
) -> np.ndarray:
    """Generate one 2D [H,W] heatmap normalized to [0,1].

    Parameters
    ----------
    model:
        Loaded E4 model. The function sets it to evaluation mode and does not train it.
    input_tensor:
        One preprocessed image, shape [1,3,H,W]. It must be on the same device as model.
    target_class:
        Class index 0..C-1. When None, use the model's Top-1 class.
    target_layer:
        Dotted module name or module instance. If omitted, the last Conv2d is used.

    Returns
    -------
    np.ndarray
        Float32 heatmap resized to the input spatial size, shape [H,W], values in [0,1].
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("input_tensor must be a torch.Tensor")
    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError(f"input_tensor must have shape [1,3,H,W], got {tuple(input_tensor.shape)}")
    if input_tensor.shape[1] != 3 or input_tensor.shape[2] < 1 or input_tensor.shape[3] < 1:
        raise ValueError("input_tensor must have 3 channels and non-empty spatial dimensions")
    if not torch.is_floating_point(input_tensor):
        raise TypeError("input_tensor must be floating point (e.g. ToTensor + Normalize)")
    if not torch.isfinite(input_tensor).all():
        raise ValueError("input_tensor contains NaN or Inf")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Grad-CAM requires autograd; call it outside torch.inference_mode()")

    layer = resolve_target_layer(model, target_layer)
    captured: dict[str, torch.Tensor] = {}

    def forward_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("Grad-CAM target layer output must be a Tensor")
        captured["activations"] = output

    handle = layer.register_forward_hook(forward_hook)
    model.eval()
    try:
        # Input requires grad so the graph remains connected even if every model
        # parameter is frozen (the E4 frozen-backbone case).
        x = input_tensor.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = _extract_logits(model(x))
            if logits.shape[0] != 1:
                raise ValueError(f"Grad-CAM expects a single-sample model output, got {tuple(logits.shape)}")
            if not torch.isfinite(logits).all():
                raise ValueError("model logits contain NaN or Inf")
            if "activations" not in captured:
                raise RuntimeError("Grad-CAM hook did not capture the target layer output")
            activations = captured["activations"]
            if activations.ndim != 4:
                raise ValueError(f"Target layer must output [N,K,H,W], got {tuple(activations.shape)}")
            num_classes = logits.shape[1]
            if target_class is None:
                selected_class = int(torch.argmax(logits[0]).item())
            else:
                if isinstance(target_class, bool) or not isinstance(target_class, (int, np.integer)):
                    raise TypeError("target_class must be an integer index or None")
                selected_class = int(target_class)
            if selected_class < 0 or selected_class >= num_classes:
                raise ValueError(f"target_class must be in [0, {num_classes - 1}]")

            gradients = torch.autograd.grad(
                outputs=logits[0, selected_class],
                inputs=activations,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((channel_weights * activations).sum(dim=1, keepdim=True))
            cam = F.interpolate(
                cam,
                size=(int(input_tensor.shape[2]), int(input_tensor.shape[3])),
                mode="bilinear",
                align_corners=False,
            )
            cam = cam[0, 0]
            cam_min = torch.min(cam)
            cam_max = torch.max(cam)
            span = cam_max - cam_min
            if not torch.isfinite(span):
                raise RuntimeError("Grad-CAM produced non-finite values")
            if float(span.detach().cpu()) <= torch.finfo(cam.dtype).eps:
                normalized = torch.zeros_like(cam)
            else:
                normalized = (cam - cam_min) / span
            normalized = torch.clamp(normalized, 0.0, 1.0)
            if not torch.isfinite(normalized).all():
                raise RuntimeError("Normalized Grad-CAM heatmap contains NaN or Inf")
            return normalized.detach().to(dtype=torch.float32, device="cpu").numpy()
    finally:
        handle.remove()
