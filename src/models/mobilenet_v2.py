"""MobileNetV2 model for DogBreedID experiments E1 and E2.

E1:
    - ImageNet pretrained MobileNetV2
    - Replace classifier with 30-class head
    - Freeze all model.features
    - Train classifier only

E2:
    - ImageNet pretrained MobileNetV2
    - Replace classifier with 30-class head
    - Freeze model.features[:14]
    - Train model.features[14:] + classifier

This module intentionally contains only model architecture/freeze logic.
Optimizer creation is handled by src.models.model_factory.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def _load_mobilenet_v2(pretrained: bool = True) -> nn.Module:
    """Create MobileNetV2 with ImageNet weights when requested.

    Supports both newer torchvision (weights=...) and older torchvision
    (pretrained=...) APIs.
    """
    if not pretrained:
        return models.mobilenet_v2(weights=None)

    # New torchvision API.
    try:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        return models.mobilenet_v2(weights=weights)
    except (AttributeError, TypeError):
        # Compatibility with older torchvision versions.
        return models.mobilenet_v2(pretrained=True)


def _freeze_all_features(model: nn.Module) -> None:
    """E1: freeze every parameter in the MobileNetV2 feature extractor."""
    for parameter in model.features.parameters():
        parameter.requires_grad = False


def _freeze_features_before(model: nn.Module, index: int) -> None:
    """E2: freeze features[:index], train features[index:]."""
    if index < 0 or index > len(model.features):
        raise ValueError(
            f"Invalid frozen feature index={index}; "
            f"MobileNetV2 has {len(model.features)} feature blocks."
        )

    for i, block in enumerate(model.features):
        trainable = i >= index
        for parameter in block.parameters():
            parameter.requires_grad = trainable


def _set_classifier(model: nn.Module, num_classes: int) -> None:
    """Replace MobileNetV2's final Linear layer with a num_classes head."""
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    # torchvision MobileNetV2 classifier:
    # Sequential(Dropout(...), Linear(last_channel, 1000))
    if not isinstance(model.classifier, nn.Sequential):
        raise TypeError("Expected MobileNetV2 classifier to be nn.Sequential.")

    if len(model.classifier) == 0 or not isinstance(model.classifier[-1], nn.Linear):
        raise TypeError("Expected the last classifier layer to be nn.Linear.")

    in_features = model.classifier[-1].in_features

    # Keep the original Dropout and replace only the final classification layer.
    model.classifier[-1] = nn.Linear(in_features, num_classes)


def _validate_trainable_policy(model: nn.Module, experiment_id: str) -> None:
    """Fail fast if the requested E1/E2 freeze policy is incorrect."""
    feature_params = list(model.features.parameters())
    classifier_params = list(model.classifier.parameters())

    if experiment_id == "E1":
        if any(p.requires_grad for p in feature_params):
            raise RuntimeError("E1 freeze policy violated: features must be frozen.")
        if not classifier_params or not all(p.requires_grad for p in classifier_params):
            raise RuntimeError("E1 freeze policy violated: classifier must be trainable.")

    elif experiment_id == "E2":
        frozen = list(model.features[:14].parameters())
        trainable = list(model.features[14:].parameters())

        if any(p.requires_grad for p in frozen):
            raise RuntimeError(
                "E2 freeze policy violated: features[:14] must be frozen."
            )
        if not trainable or not all(p.requires_grad for p in trainable):
            raise RuntimeError(
                "E2 freeze policy violated: features[14:] must be trainable."
            )
        if not classifier_params or not all(p.requires_grad for p in classifier_params):
            raise RuntimeError("E2 freeze policy violated: classifier must be trainable.")

    else:
        raise ValueError(f"Unsupported MobileNetV2 experiment: {experiment_id}")


def build_model(model_config: dict) -> nn.Module:
    """Build MobileNetV2 according to the shared E1/E2 experiment config.

    Expected model_config examples:

    E1:
        {
            "type": "mobilenet_v2",
            "weights": "imagenet",
            "backbone_frozen": True,
            "num_classes": 30,
        }

    E2:
        {
            "type": "mobilenet_v2",
            "weights": "imagenet",
            "frozen_features_before_index": 14,
            "trainable_features_from_index": 14,
            "num_classes": 30,
        }

    Returns:
        torch.nn.Module producing raw logits with shape [B, num_classes].
    """
    if not isinstance(model_config, dict):
        raise TypeError("model_config must be a dict.")

    num_classes = int(model_config.get("num_classes", 30))
    weights_name = str(model_config.get("weights", "imagenet")).lower()

    if weights_name not in {"imagenet", "none", "null"}:
        raise ValueError(
            f"Unsupported MobileNetV2 weights setting: {weights_name}. "
            "Expected 'imagenet' or 'none'."
        )

    use_pretrained = weights_name == "imagenet"
    model = _load_mobilenet_v2(pretrained=use_pretrained)

    _set_classifier(model, num_classes)

    # Detect the experiment from the config rather than hard-coding two
    # separate model implementations.
    if bool(model_config.get("backbone_frozen", False)):
        experiment_id = "E1"
        _freeze_all_features(model)

    elif (
        "frozen_features_before_index" in model_config
        or "trainable_features_from_index" in model_config
    ):
        experiment_id = "E2"

        frozen_index = int(
            model_config.get(
                "trainable_features_from_index",
                model_config.get("frozen_features_before_index", 14),
            )
        )

        # Both config keys are intended to describe the same boundary.
        if (
            "frozen_features_before_index" in model_config
            and "trainable_features_from_index" in model_config
        ):
            frozen_before = int(model_config["frozen_features_before_index"])
            trainable_from = int(model_config["trainable_features_from_index"])
            if frozen_before != trainable_from:
                raise ValueError(
                    "E2 config mismatch: frozen_features_before_index and "
                    "trainable_features_from_index must be equal."
                )

        _freeze_features_before(model, frozen_index)

    else:
        raise ValueError(
            "Cannot determine MobileNetV2 freeze policy. "
            "Expected backbone_frozen=True for E1 or "
            "frozen_features_before_index/trainable_features_from_index for E2."
        )

    _validate_trainable_policy(model, experiment_id)

    # The model returns raw logits. No Softmax is applied here because the
    # shared training pipeline uses CrossEntropyLoss.
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_parameters, trainable_parameters)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Standalone smoke test; does not modify the repository or training outputs.
    torch.manual_seed(42)

    e1_config = {
        "type": "mobilenet_v2",
        "weights": "none",  # Avoid downloading weights for this local smoke test.
        "backbone_frozen": True,
        "num_classes": 30,
    }

    e2_config = {
        "type": "mobilenet_v2",
        "weights": "none",
        "frozen_features_before_index": 14,
        "trainable_features_from_index": 14,
        "num_classes": 30,
    }

    x = torch.randn(2, 3, 224, 224)

    for name, config in [("E1", e1_config), ("E2", e2_config)]:
        model = build_model(config)
        model.eval()

        with torch.no_grad():
            logits = model(x)

        total, trainable = count_parameters(model)

        print(f"{name}:")
        print(f"  output shape      = {tuple(logits.shape)}")
        print(f"  total parameters  = {total:,}")
        print(f"  trainable params  = {trainable:,}")

    print("MobileNetV2 E1/E2 model smoke test PASSED.")
