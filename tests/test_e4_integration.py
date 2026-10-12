"""E4 integration checks; the real MobileNet smoke test runs in the full repo."""
import pytest
import torch
from torch import nn

from src.models.model_factory import build_optimizer


class _TinyMobileNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(5, 5))
        self.classifier = nn.Sequential(nn.Linear(5, 30))
        for parameter in self.features.parameters():
            parameter.requires_grad = False


def _config(experiment_id):
    return {
        "experiment": {"id": experiment_id},
        "training": {"weight_decay": 1e-4, "classifier_lr": 1e-3},
    }


def test_e4_optimizer_uses_the_same_classifier_only_policy_as_e1():
    e1_model = _TinyMobileNetLike()
    e4_model = _TinyMobileNetLike()
    e1_optimizer = build_optimizer(e1_model, _config("E1"))
    e4_optimizer = build_optimizer(e4_model, _config("E4"))

    for model, optimizer in [(e1_model, e1_optimizer), (e4_model, e4_optimizer)]:
        trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        assert optimizer_ids == trainable_ids
        assert all(not parameter.requires_grad for parameter in model.features.parameters())
        assert all(parameter.requires_grad for parameter in model.classifier.parameters())
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_real_mobilenet_e4_forward_smoke_in_full_repository():
    try:
        from src.models.mobilenet_v2 import build_model
    except (ImportError, ModuleNotFoundError):
        pytest.skip("The unmodified project MobileNetV2 module is not bundled here; run this in repo root.")

    model = build_model({
        "type": "mobilenet_v2",
        "weights": "none",  # Smoke test must not download pretrained weights.
        "backbone_frozen": True,
        "num_classes": 30,
    })
    model.eval()
    with torch.inference_mode():
        logits = model(torch.randn(2, 3, 224, 224))
    assert tuple(logits.shape) == (2, 30)
    assert torch.isfinite(logits).all()
    assert all(not parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
