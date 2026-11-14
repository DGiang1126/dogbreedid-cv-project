"""Unit tests for Task 07 E10/E11 models, freeze policy, and optimizer."""
from __future__ import annotations

import pytest
import torch
from torchvision.models import EfficientNet_B0_Weights

from src.research import extended_models as em
from src.research.extended_config import load_research_config
from src.research.extended_optimizer import build_research_optimizer


def test_e10_efficientnet_contract_and_freeze(monkeypatch):
    requested = {}
    orig_effnet = em.efficientnet_b0

    def offline_effnet(*, weights):
        requested["weights"] = weights
        return orig_effnet(weights=None)

    monkeypatch.setattr(em, "efficientnet_b0", offline_effnet)
    config = load_research_config("configs/experiments/e10a_efficientnet_no_cutmix.yaml")
    model = em.build_research_model(config)

    assert requested["weights"] is EfficientNet_B0_Weights.IMAGENET1K_V1
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 30)
    assert torch.isfinite(out).all()

    assert all(not p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())

    # Backbone must stay in eval() even when model.train(True) is called.
    model.train(True)
    assert not model.features.training
    for m in model.features.modules():
        assert not m.training
    assert model.classifier.training

    opt = build_research_optimizer(model, config)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    cls_ids = {id(p) for p in model.classifier.parameters()}
    assert opt_ids == cls_ids
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(1e-4)


def test_e11_vit_tiny_contract_and_freeze(monkeypatch):
    requested = {}
    orig_create = em.timm.create_model

    def offline_create(model_name, *, pretrained, num_classes):
        requested["model_name"] = model_name
        requested["pretrained"] = pretrained
        requested["num_classes"] = num_classes
        return orig_create(model_name, pretrained=False, num_classes=num_classes)

    monkeypatch.setattr(em.timm, "create_model", offline_create)
    config = load_research_config("configs/experiments/e11a_vit_no_cutmix.yaml")
    model = em.build_research_model(config)

    assert requested == {
        "model_name": "vit_tiny_patch16_224",
        "pretrained": True,
        "num_classes": 30,
    }
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 30)
    assert torch.isfinite(out).all()

    head_ids = {id(p) for p in model.head.parameters()}
    for p in model.parameters():
        if id(p) in head_ids:
            assert p.requires_grad
        else:
            assert not p.requires_grad

    model.train(True)
    for name, child in model.named_children():
        if name == "head":
            assert child.training
        else:
            assert not child.training

    opt = build_research_optimizer(model, config)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert opt_ids == head_ids
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_tampered_freeze_rejected(monkeypatch):
    orig_effnet = em.efficientnet_b0
    monkeypatch.setattr(em, "efficientnet_b0", lambda *, weights: orig_effnet(weights=None))
    config = load_research_config("configs/experiments/e10b_efficientnet_cutmix.yaml")
    model = em.build_research_model(config)

    # Unfreeze a backbone param -> must raise RuntimeError
    next(model.features.parameters()).requires_grad = True
    with pytest.raises(RuntimeError):
        build_research_optimizer(model, config)
