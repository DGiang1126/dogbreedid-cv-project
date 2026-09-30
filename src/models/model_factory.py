"""E0–E3 model contract; member model files must expose build_model(model_config)."""
import torch


def build_model(config: dict) -> torch.nn.Module:
    kind = config["model"]["type"]
    module_names = {
        "custom_cnn": "custom_cnn",
        "mobilenet_v2": "mobilenet_v2",
        "resnet18": "resnet18",
    }
    if kind not in module_names:
        raise ValueError(f"Unsupported model type: {kind}")
    from importlib import import_module
    module = import_module("src.models." + module_names[kind])
    builder = getattr(module, "build_model", None)
    if builder is None:
        raise NotImplementedError(
            f"src/models/{module_names[kind]}.py must implement build_model(model_config) "
            "in the model-owner task. Foundation dry-run does not require it."
        )
    model = builder(config["model"])
    if not isinstance(model, torch.nn.Module):
        raise TypeError("The model builder must return torch.nn.Module")
    return model


def build_optimizer(model: torch.nn.Module, config: dict):
    exp = config["experiment"]["id"]
    train = config["training"]
    wd = float(train["weight_decay"])
    if exp == "E0":
        groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                   "lr": float(train["learning_rate"])}]
    elif exp == "E1":
        groups = [{"params": [p for p in model.classifier.parameters() if p.requires_grad],
                   "lr": float(train["classifier_lr"])}]
    elif exp == "E2":
        groups = [
            {"params": [p for p in model.features[14:].parameters() if p.requires_grad],
             "lr": float(train["backbone_lr"])},
            {"params": [p for p in model.classifier.parameters() if p.requires_grad],
             "lr": float(train["classifier_lr"])},
        ]
    elif exp == "E3":
        groups = [
            {"params": [p for p in model.layer4.parameters() if p.requires_grad],
             "lr": float(train["backbone_lr"])},
            {"params": [p for p in model.fc.parameters() if p.requires_grad],
             "lr": float(train["classifier_lr"])},
        ]
    else:
        raise ValueError(f"Unsupported training experiment: {exp}")
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    optimizer_ids = [id(p) for group in groups for p in group["params"]]
    if not optimizer_ids or set(optimizer_ids) != trainable_ids or len(set(optimizer_ids)) != len(optimizer_ids):
        raise ValueError("Optimizer groups do not cover every trainable parameter exactly once; inspect freezes.")
    return torch.optim.AdamW(groups, weight_decay=wd)
