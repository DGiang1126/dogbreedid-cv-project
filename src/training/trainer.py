"""Model-agnostic training loop with correct residual gradient accumulation."""
from __future__ import annotations

import torch

from src.evaluation.metrics import classification_metrics


def _freeze_frozen_batchnorm_stats(model):
    """Do not update running statistics of frozen BatchNorm layers."""
    from torch.nn.modules.batchnorm import _BatchNorm
    for module in model.modules():
        if isinstance(module, _BatchNorm) and not any(p.requires_grad for p in module.parameters(recurse=False)):
            module.eval()


def train_one_epoch(model, loader, criterion, optimizer, device, accumulation_steps=1, scaler=None):
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be at least one")
    model.train()
    _freeze_frozen_batchnorm_stats(model)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    seen = 0
    all_logits, all_targets = [], []
    n_batches = len(loader)
    amp = scaler is not None and scaler.is_enabled() and device.type == "cuda"
    for step, batch in enumerate(loader):
        image = batch["image"].to(device)
        target = batch["label"].to(device).long()
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(image)
            loss = criterion(logits, target)
        group_start = (step // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, n_batches - group_start)
        scaled_loss = loss / group_size
        if amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        if (step + 1) % accumulation_steps == 0 or step + 1 == n_batches:
            if amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        seen += len(target)
        total_loss += loss.item() * len(target)
        all_logits.append(logits.detach().float().cpu())
        all_targets.append(target.detach().cpu())
    if seen == 0:
        raise ValueError("Empty training loader.")
    result = classification_metrics(torch.cat(all_logits), torch.cat(all_targets))
    return {"loss": total_loss / seen, **result}
