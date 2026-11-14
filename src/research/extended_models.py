"""Model builders for Task 07 E10 (EfficientNet-B0) and E11 (ViT-Tiny)."""
from __future__ import annotations

import types
from typing import Any

import timm
import torch
import torch.nn as nn
import torchvision
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def _bind_frozen_backbone_train(model: nn.Module, head_attr: str) -> None:
    """Keep frozen backbone submodules in eval() even when model.train(True) is called."""
    def _train_with_frozen_backbone(self: nn.Module, mode: bool = True) -> nn.Module:
        nn.Module.train(self, mode)
        if mode:
            for name, child in self.named_children():
                if name != head_attr:
                    child.eval()
        return self

    model.train = types.MethodType(_train_with_frozen_backbone, model)


def _freeze_except_head(model: nn.Module, head_attr: str) -> None:
    """Freeze all parameters except those belonging to `head_attr`."""
    head = getattr(model, head_attr, None)
    if head is None or not isinstance(head, nn.Module):
        raise ValueError(f"Model has no '{head_attr}' submodule.")

    for param in model.parameters():
        param.requires_grad = False
    for param in head.parameters():
        param.requires_grad = True

    _bind_frozen_backbone_train(model, head_attr)
    model.eval()


def validate_frozen_backbone(model: nn.Module, model_type: str) -> None:
    """Verify only the classifier/head is trainable and backbone remains frozen."""
    head_attr = "classifier" if model_type == "efficientnet_b0" else "head"
    head = getattr(model, head_attr)
    head_param_ids = {id(p) for p in head.parameters()}
    if not head_param_ids:
        raise RuntimeError(f"{model_type} head '{head_attr}' has no parameters.")

    for name, param in model.named_parameters():
        in_head = id(param) in head_param_ids
        if in_head and not param.requires_grad:
            raise RuntimeError(f"Head parameter '{name}' must be trainable.")
        if not in_head and param.requires_grad:
            raise RuntimeError(f"Backbone parameter '{name}' must be frozen.")


def build_efficientnet_b0(num_classes: int = 30, pretrained: bool = True) -> nn.Module:
    """Build torchvision EfficientNet-B0 with frozen backbone and 30-class head."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    if not isinstance(model.classifier, nn.Sequential) or not isinstance(model.classifier[-1], nn.Linear):
        raise TypeError("Unexpected EfficientNet-B0 classifier structure.")
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    _freeze_except_head(model, "classifier")
    validate_frozen_backbone(model, "efficientnet_b0")
    return model


def build_vit_tiny(num_classes: int = 30, pretrained: bool = True) -> nn.Module:
    """Build timm ViT-Tiny (patch16_224) with frozen backbone and 30-class head."""
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    if not isinstance(getattr(model, "head", None), nn.Linear):
        raise TypeError("Unexpected ViT-Tiny head structure; expected nn.Linear.")
    _freeze_except_head(model, "head")
    validate_frozen_backbone(model, "vit_tiny_patch16_224")
    return model


def build_research_model(config: dict[str, Any], force_no_pretrained: bool = False) -> nn.Module:
    """Build E10/E11 model according to validated config."""
    model_cfg = config["model"]
    model_type = model_cfg["type"]
    num_classes = int(model_cfg["num_classes"])
    weights_cfg = str(model_cfg.get("weights", "")).lower()
    pretrained = (not force_no_pretrained) and (weights_cfg not in {"", "none", "null"})

    if model_type == "efficientnet_b0":
        return build_efficientnet_b0(num_classes=num_classes, pretrained=pretrained)
    if model_type == "vit_tiny_patch16_224":
        return build_vit_tiny(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported research model type: {model_type}")


def get_classifier_parameters(model: nn.Module, model_type: str) -> list[nn.Parameter]:
    """Return trainable parameters of the classification head."""
    head_attr = "classifier" if model_type == "efficientnet_b0" else "head"
    head = getattr(model, head_attr)
    return [p for p in head.parameters() if p.requires_grad]


def get_model_provenance(config: dict[str, Any]) -> dict[str, Any]:
    """Return library versions, weight enum/ID, and preprocessing metadata."""
    model_cfg = config["model"]
    model_type = model_cfg["type"]
    if model_type == "efficientnet_b0":
        weight_id = "EfficientNet_B0_Weights.IMAGENET1K_V1"
        source_lib = f"torchvision=={torchvision.__version__}"
    else:
        cfg = timm.get_pretrained_cfg("vit_tiny_patch16_224")
        weight_id = getattr(cfg, "hf_hub_id", "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k")
        source_lib = f"timm=={timm.__version__}"
    return {
        "model_type": model_type,
        "weight_identifier": weight_id,
        "source_library": source_lib,
        "torchvision_version": torchvision.__version__,
        "timm_version": timm.__version__,
        "backbone_frozen": True,
        "normalization": config["preprocessing"]["normalization"],
    }
