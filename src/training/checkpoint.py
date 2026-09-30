"""Checkpoint save/load. Load only trusted local artifacts."""
from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, metrics: dict, config: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "validation_metrics": metrics,
        "config": config,
    }, path)
