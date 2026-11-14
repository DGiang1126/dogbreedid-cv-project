"""Optimizer builder for Task 07 E10/E11 frozen-backbone models."""
from __future__ import annotations

from typing import Any

import torch

from src.research.extended_models import get_classifier_parameters, validate_frozen_backbone


def build_research_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.AdamW:
    """Create AdamW updating only the classifier head, verifying parameter coverage."""
    model_type = config["model"]["type"]
    validate_frozen_backbone(model, model_type)

    head_params = get_classifier_parameters(model, model_type)
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    head_ids = [id(p) for p in head_params]

    if not head_ids or set(head_ids) != trainable_ids or len(set(head_ids)) != len(head_ids):
        raise RuntimeError("Optimizer parameter groups must cover only and all trainable head parameters.")

    lr = float(config["training"]["classifier_lr"])
    wd = float(config["training"]["weight_decay"])
    return torch.optim.AdamW([{"params": head_params, "lr": lr}], weight_decay=wd)
