"""Portable CPU/CUDA/MPS selection and mixed-precision decision."""
import torch


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested, but it is not available.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown device setting: {requested}")
    return torch.device(requested)


def use_mixed_precision(device: torch.device, config: dict) -> bool:
    return device.type == "cuda" and bool(config["training"].get("mixed_precision", False))
