"""Tests for model input/output contracts."""
import pytest
import torch
from torchvision.models import ResNet18_Weights

from src.models import resnet18 as resnet18_module
from src.models.model_factory import build_model, build_optimizer
from src.utils.config import load_experiment_config


def test_e3_resnet18_contract(monkeypatch):
    requested = {}
    torchvision_builder = resnet18_module.resnet18

    def offline_resnet18(*, weights):
        requested["weights"] = weights
        return torchvision_builder(weights=None)

    monkeypatch.setattr(resnet18_module, "resnet18", offline_resnet18)
    config = load_experiment_config("configs/experiments/e3_resnet18_partial.yaml")
    model = build_model(config)

    assert requested["weights"] is ResNet18_Weights.IMAGENET1K_V1
    assert model.fc.out_features == 30
    assert model(torch.randn(2, 3, 224, 224)).shape == (2, 30)
    for name in ("conv1", "bn1", "layer1", "layer2", "layer3"):
        assert all(not parameter.requires_grad for parameter in getattr(model, name).parameters())
    for name in ("layer4", "fc"):
        assert all(parameter.requires_grad for parameter in getattr(model, name).parameters())

    optimizer = build_optimizer(model, config)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([1e-4, 5e-4])
