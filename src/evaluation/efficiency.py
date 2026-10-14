"""Efficiency measurements: parameter count, model size, and inference latency."""
from __future__ import annotations

import io
import time
from typing import Optional

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mib(model: nn.Module) -> float:
    """Serialized state_dict size in MiB (not full checkpoint with optimizer state)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 ** 2)


def measure_latency(
    model: nn.Module,
    device: torch.device,
    input_shape: tuple = (1, 3, 224, 224),
    warmup: int = 20,
    timed: int = 100,
) -> dict[str, float]:
    """Measure inference latency per image.

    Protocol (per task spec):
    - batch_size=1, input [1,3,224,224], model.eval(), torch.inference_mode()
    - warmup 20 runs, timed 100 runs
    - CUDA synchronize before/after each timed run
    - Reports median_ms and mean±std_ms

    Returns dict with keys: median_ms, mean_ms, std_ms
    """
    model.eval()
    dummy = torch.zeros(input_shape, device=device)
    use_cuda = device.type == "cuda"

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(dummy)
        if use_cuda:
            torch.cuda.synchronize()

        latencies: list[float] = []
        for _ in range(timed):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if use_cuda:
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    t = torch.tensor(latencies)
    return {
        "median_ms": float(t.median()),
        "mean_ms": float(t.mean()),
        "std_ms": float(t.std()),
    }


def load_checkpoint_into_model(model: nn.Module, checkpoint_path: str) -> None:
    """Load model_state_dict from checkpoint with strict=True (task requirement)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)


def verify_forward(model: nn.Module, device: torch.device, num_classes: int = 30) -> None:
    """Assert output shape [1, num_classes] and all logits finite."""
    model.eval()
    with torch.inference_mode():
        out = model(torch.zeros(1, 3, 224, 224, device=device))
    assert out.shape == (1, num_classes), f"Expected (1,{num_classes}), got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "Logits contain non-finite values"


def benchmark_model(
    model: nn.Module,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    num_classes: int = 30,
    warmup: int = 20,
    timed: int = 100,
) -> dict:
    """Full benchmark: optionally load checkpoint, count params, measure size and latency.

    Returns a dict with keys:
      checkpoint_loaded, total_params, trainable_params,
      size_mib, median_ms, mean_ms, std_ms
    """
    checkpoint_loaded = False
    if checkpoint_path is not None:
        load_checkpoint_into_model(model, checkpoint_path)
        checkpoint_loaded = True

    model = model.to(device)
    verify_forward(model, device, num_classes)

    total, trainable = count_parameters(model)
    size = model_size_mib(model)
    lat = measure_latency(model, device, warmup=warmup, timed=timed)

    return {
        "checkpoint_loaded": checkpoint_loaded,
        "total_params": total,
        "trainable_params": trainable,
        "size_mib": round(size, 4),
        **{k: round(v, 4) for k, v in lat.items()},
    }
