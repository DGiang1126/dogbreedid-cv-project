"""Model-agnostic training loop with correct residual gradient accumulation."""
from __future__ import annotations

import math

import torch

from src.evaluation.metrics import classification_metrics
from src.training.cutmix import cutmix_batch, cutmix_criterion_loss


def _freeze_frozen_batchnorm_stats(model):
    """Do not update running statistics of frozen BatchNorm layers."""
    from torch.nn.modules.batchnorm import _BatchNorm
    for module in model.modules():
        if isinstance(module, _BatchNorm) and not any(p.requires_grad for p in module.parameters(recurse=False)):
            module.eval()


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    accumulation_steps=1,
    scaler=None,
    cutmix_config=None,
):
    """Train one epoch; CutMix is opt-in and leaves E0-E3 unchanged.

    If CutMix is enabled, standard classification metrics are measured only on
    batches where CutMix was not applied. Metrics against a single original
    label on a mixed image would not be a valid standard classification metric.
    The result then includes ``metric_samples`` to show how many examples were
    used for those training metrics. Validation uses ``validator.py`` and never
    receives CutMix configuration.
    """
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be at least one")
    cutmix_config = cutmix_config or {}
    cutmix_enabled = bool(cutmix_config.get("enabled", False))
    cutmix_alpha = float(cutmix_config.get("alpha", 1.0))
    cutmix_probability = float(cutmix_config.get("probability", 0.5))
    if cutmix_enabled:
        # Validate fixed CutMix settings before starting the epoch.
        if not math.isfinite(cutmix_alpha) or cutmix_alpha <= 0:
            raise ValueError("CutMix alpha must be finite and greater than zero.")
        if not math.isfinite(cutmix_probability) or not 0.0 <= cutmix_probability <= 1.0:
            raise ValueError("CutMix probability must be in [0, 1].")

    model.train()
    _freeze_frozen_batchnorm_stats(model)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    seen = 0
    metric_seen = 0
    all_logits, all_targets = [], []
    n_batches = len(loader)
    amp = scaler is not None and scaler.is_enabled() and device.type == "cuda"

    for step, batch in enumerate(loader):
        image = batch["image"].to(device)
        target = batch["label"].to(device).long()

        applied = False
        lambda_actual = 1.0
        target_a = target
        target_b = target
        model_image = image
        if cutmix_enabled:
            model_image, target_a, target_b, lambda_actual, applied = cutmix_batch(
                image,
                target,
                alpha=cutmix_alpha,
                probability=cutmix_probability,
            )

        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(model_image)
            loss = cutmix_criterion_loss(
                criterion,
                logits,
                target_a,
                target_b,
                lambda_actual,
                applied,
            ) if cutmix_enabled else criterion(logits, target)

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

        batch_size = len(target)
        seen += batch_size
        total_loss += loss.detach().item() * batch_size

        # Mixed images have two valid labels; do not score them against only
        # target_a. E0-E3 preserve their previous metric path exactly.
        if not cutmix_enabled or not applied:
            all_logits.append(logits.detach().float().cpu())
            all_targets.append(target.detach().cpu())
            metric_seen += batch_size

    if seen == 0:
        raise ValueError("Empty training loader.")

    result = {"loss": total_loss / seen}
    if metric_seen > 0:
        result.update(classification_metrics(torch.cat(all_logits), torch.cat(all_targets)))
    else:
        # This is unusual under the required probability=0.5 and full dataset,
        # but NaN is more honest than presenting invalid metrics as real values.
        result.update({"macro_f1": float("nan"), "top1_accuracy": float("nan"), "top5_accuracy": float("nan")})

    if cutmix_enabled:
        result["metric_samples"] = metric_seen
    return result
