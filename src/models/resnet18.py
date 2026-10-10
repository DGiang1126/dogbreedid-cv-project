"""ResNet18 model definition/shared setup for E3."""
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(model_config: dict) -> nn.Module:
    """Build ImageNet-pretrained ResNet18 with config-defined freeze policy."""
    if model_config.get("weights") != "imagenet":
        raise ValueError("E3 requires ImageNet pretrained weights.")

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, int(model_config["num_classes"]))

    frozen = set(model_config["frozen_modules"])
    trainable = set(model_config["trainable_modules"])
    expected_frozen = {"conv1", "bn1", "layer1", "layer2", "layer3"}
    expected_trainable = {"layer4", "fc"}
    if frozen != expected_frozen or trainable != expected_trainable:
        raise ValueError("E3 freeze policy must freeze conv1-bn1-layer1-layer2-layer3 and train layer4-fc.")

    for parameter in model.parameters():
        parameter.requires_grad = False
    for module_name in trainable:
        for parameter in getattr(model, module_name).parameters():
            parameter.requires_grad = True
    return model
