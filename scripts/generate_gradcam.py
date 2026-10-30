#!/usr/bin/env python3
"""Create Grad-CAM overlay for an allowed synthetic/Validation image with E4.

This script does not run Official Test inference. It uses the repository's E4
loader via --model-loader and verifies the approved checkpoint SHA-256 first.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.confusion_matrix import load_class_names, stable_softmax
from src.explainability.gradcam import generate_gradcam


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError("--model-loader must be written as python.module:function")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise TypeError(f"Model loader {spec!r} is not callable")
    return function


def _call_model_loader(
    loader: Callable[..., Any], checkpoint_path: Path, device: torch.device,
    config_path: Path, trust_loader_config_checkpoint: bool,
) -> torch.nn.Module:
    signature = inspect.signature(loader)
    parameters = signature.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    aliases = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint": str(checkpoint_path),
        "ckpt_path": str(checkpoint_path),
        "weights_path": str(checkpoint_path),
        "path": str(checkpoint_path),
        "device": str(device),
        "config_path": str(config_path),
        "config": str(config_path),
        "repo_root": str(REPO_ROOT),
        "project_root": str(REPO_ROOT),
        "root": str(REPO_ROOT),
    }
    kwargs: dict[str, Any] = {}
    if accepts_kwargs:
        kwargs.update({"checkpoint_path": str(checkpoint_path), "device": str(device), "config_path": str(config_path)})
    else:
        kwargs.update({name: value for name, value in aliases.items() if name in parameters})

    checkpoint_param_names = {"checkpoint_path", "checkpoint", "ckpt_path", "weights_path", "path"}
    loader_has_checkpoint_arg = accepts_kwargs or bool(checkpoint_param_names.intersection(parameters))
    if not loader_has_checkpoint_arg and not trust_loader_config_checkpoint:
        raise RuntimeError(
            f"Loader signature {signature} does not accept a checkpoint path. Inspect src/inference/predictor.py; "
            "if it intentionally loads the frozen checkpoint from config, rerun with "
            "--trust-loader-config-checkpoint after verifying that implementation."
        )
    # Required parameters the adapter cannot understand should fail with a helpful message.
    missing_required = [
        name for name, p in parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in kwargs
    ]
    if missing_required:
        raise RuntimeError(f"Cannot call model loader {signature}; unsupported required arguments: {missing_required}")
    loaded = loader(**kwargs)
    if isinstance(loaded, torch.nn.Module):
        model = loaded
    elif isinstance(loaded, (tuple, list)):
        model = next((item for item in loaded if isinstance(item, torch.nn.Module)), None)
        if model is None:
            raise TypeError("Model loader tuple/list did not contain a torch.nn.Module")
    elif isinstance(loaded, dict) and isinstance(loaded.get("model"), torch.nn.Module):
        model = loaded["model"]
    else:
        raise TypeError("Model loader must return a torch.nn.Module, a tuple/list containing one, or {'model': module}")
    return model.to(device).eval()


def _load_config_object(config_path: Path) -> Any:
    """Load E4 config using the repository config loader when available, else YAML."""
    e4_path = (REPO_ROOT / "configs" / "experiments" / "e4_cutmix.yaml").resolve()
    if not e4_path.is_file():
        raise FileNotFoundError(f"E4 experiment config not found: {e4_path}")
    try:
        config_module = importlib.import_module("src.utils.config")
        load_config = getattr(config_module, "load_experiment_config", None)
        if callable(load_config):
            try:
                return load_config(str(e4_path))
            except TypeError:
                sig = inspect.signature(load_config)
                for name in ("config_path", "path", "config_file"):
                    if name in sig.parameters:
                        return load_config(**{name: str(e4_path)})
                raise
    except Exception as exc:
        print(f"E4 config loader fallback to YAML ({type(exc).__name__}: {exc})", file=sys.stderr)
    with e4_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _config_model_name(config_obj: Any) -> str:
    """Best-effort model name extraction from common dict/dataclass config layouts."""
    for section_name in ("model", "experiment", "architecture"):
        section = _obj_get(config_obj, section_name, None)
        if section is not None:
            for key in ("architecture", "name", "model_name", "type", "backbone"):
                value = _obj_get(section, key, None)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    for key in ("architecture", "model_name", "model", "name"):
        value = _obj_get(config_obj, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "mobilenet_v2"


def _build_model_through_factory(config_obj: Any, config_path: Path, device: torch.device) -> torch.nn.Module:
    """Build E4 through repository model_factory; no architecture guessing or retraining."""
    module = importlib.import_module("src.models.model_factory")
    model_name = _config_model_name(config_obj)
    aliases: dict[str, Any] = {
        "config": config_obj, "cfg": config_obj, "experiment_config": config_obj,
        "experiment_cfg": config_obj, "model_config": config_obj, "model_cfg": config_obj,
        "configuration": config_obj, "conf": config_obj,
        "config_path": str(config_path), "config_file": str(config_path),
        "yaml_path": str(config_path), "path": str(config_path),
        "repo_root": str(REPO_ROOT), "project_root": str(REPO_ROOT), "root": str(REPO_ROOT),
        "device": str(device), "num_classes": 30, "num_class": 30,
        "in_channels": 3, "num_channels": 3, "pretrained": False, "weights": None,
        "model_name": model_name, "architecture": model_name, "name": model_name,
    }
    preferred_names = (
        "build_model", "create_model", "get_model", "make_model", "build_classifier",
        "create_classifier", "get_model_from_config", "build_from_config", "model_factory",
    )
    candidates: list[tuple[str, Callable[..., Any]]] = []
    for name in preferred_names:
        obj = getattr(module, name, None)
        if callable(obj):
            candidates.append((f"src.models.model_factory:{name}", obj))
    # Also inspect project-specific factory naming, while still validating signatures/return type.
    for name, obj in inspect.getmembers(module):
        lower_name = name.lower()
        if name.startswith("_") or not callable(obj):
            continue
        if any(token in lower_name for token in ("model", "classifier", "factory")) and any(
            token in lower_name for token in ("build", "create", "get", "make", "factory")
        ):
            if not any(existing_name == f"src.models.model_factory:{name}" for existing_name, _ in candidates):
                candidates.append((f"src.models.model_factory:{name}", obj))
    factory_class = getattr(module, "ModelFactory", None)
    if inspect.isclass(factory_class):
        try:
            instance = factory_class()
        except Exception:
            instance = None
        for owner_name, owner in (("ModelFactory", instance), ("ModelFactory", factory_class)):
            if owner is None:
                continue
            method_names = set(preferred_names)
            method_names.update(
                name for name, method in inspect.getmembers(owner)
                if not name.startswith("_") and callable(method)
                and any(token in name.lower() for token in ("model", "classifier", "factory", "build", "create", "get", "make"))
            )
            for method_name in sorted(method_names):
                method = getattr(owner, method_name, None)
                if callable(method):
                    candidates.append((f"src.models.model_factory.{owner_name}:{method_name}", method))

    attempts: list[str] = []
    seen: set[tuple[str, str]] = set()
    for label, factory in candidates:
        try:
            sig = inspect.signature(factory)
        except (TypeError, ValueError):
            continue
        key = (label, str(sig))
        if key in seen:
            continue
        seen.add(key)
        kwargs: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param_name in aliases:
                kwargs[param_name] = aliases[param_name]
        required_unknown = [
            name for name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and name not in kwargs
        ]
        if required_unknown:
            attempts.append(f"{label}{sig}: unsupported required parameters {required_unknown}")
            continue
        try:
            built = factory(**kwargs)
            if isinstance(built, torch.nn.Module):
                print(f"E4 model factory selected: {label}{sig}")
                return built
            attempts.append(f"{label}{sig}: returned {type(built).__name__}, not nn.Module")
        except Exception as exc:
            attempts.append(f"{label}{sig}: {type(exc).__name__}: {exc}")
        required_positional = [p for p in sig.parameters.values() if p.default is inspect.Parameter.empty and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        if len(required_positional) == 1:
            try:
                built = factory(config_obj)
                if isinstance(built, torch.nn.Module):
                    print(f"E4 model factory selected by positional config: {label}{sig}")
                    return built
                attempts.append(f"{label}{sig}: positional config returned {type(built).__name__}")
            except Exception as exc:
                attempts.append(f"{label}{sig}: positional config failed: {type(exc).__name__}: {exc}")

    details = "\n".join(f" - {item}" for item in attempts) or " - No compatible model factory callables were found."
    raise RuntimeError(
        "Could not instantiate E4 using src/models/model_factory.py. The loader will not guess a different architecture. "
        "Inspect the repository factory API and adapt this call. Attempts:\n" + details
    )


def load_e4_model_native(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Compatibility alias for the verified shared E4 loader; no unsafe fallback."""
    return load_e4_model_auto(checkpoint_path, device)


def load_e4_model_auto(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Use the accepted SHA-verified, weights_only=True, strict shared E4 loader.

    Loader errors propagate. Retrying through a second deserializer could bypass
    the checkpoint or compatibility checks and must never be used as a fallback.
    """
    from src.inference.predictor import load_e4_model

    model, verified_hash = load_e4_model(REPO_ROOT, checkpoint_path, device)
    if verified_hash != "73df0cc1c5b16866d4facc910f3e4d3ab481e3f12f1681a59fa7d61e3845c47b":
        raise RuntimeError("Shared E4 loader returned an unapproved checkpoint SHA-256")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Shared E4 loader did not return a torch model")
    return model.eval()


def _model_state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_preprocessing() -> tuple[transforms.Compose, transforms.Compose]:
    """Return the locked E4 evaluation transform and a matched RGB crop for overlay.

    This reproduces the Task 06 contract: RGB -> Resize(256) -> CenterCrop(224) ->
    ToTensor -> ImageNet normalization. It does not introduce random augmentation.
    """
    crop_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    normalized_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return crop_transform, normalized_transform


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/e9_error_analysis.yaml")
    parser.add_argument("--image", required=True, help="Path to an allowed synthetic/Validation image")
    parser.add_argument("--checkpoint", default=None, help="Defaults to model.checkpoint_path from E9 YAML")
    parser.add_argument("--output", default="docs/task06/e9/gradcam_overlay.png")
    parser.add_argument("--target-class", type=int, default=None, help="Defaults to raw-logit Top-1 class")
    parser.add_argument("--true-label", default=None, help="Optional known Validation ground-truth label for the figure metadata")
    parser.add_argument("--target-layer", default=None, help="Dotted module name; default is last Conv2d")
    parser.add_argument("--model-loader", default="auto", help="auto = verified src.inference.predictor.load_e4_model; or module:function")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--trust-loader-config-checkpoint", action="store_true", help="Only after inspecting loader code and confirming it reads the locked checkpoint")
    parser.add_argument("--split", choices=["synthetic", "validation"], default="validation", help="Official mode intentionally unsupported by this script")
    args = parser.parse_args()

    try:
        config_path = Path(args.config).resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict) or config.get("experiment", {}).get("id") != "E9":
            raise ValueError("Config must be a valid E9 YAML")
        if config.get("model", {}).get("experiment_id") != "E4" or config.get("model", {}).get("seed") != 42:
            raise ValueError("Grad-CAM is locked to E4 seed 42")
        if not math.isclose(float(config.get("e6", {}).get("temperature", float("nan"))), 0.80246283961825, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("E6 Temperature is locked; do not refit or change it")
        if not math.isclose(float(config.get("e7", {}).get("threshold", float("nan"))), 0.5011672918550012, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("E7 threshold is locked; do not retune or change it")
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            device = torch.device(args.device)

        image_path = Path(args.image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        checkpoint_path = Path(args.checkpoint or config["model"]["checkpoint_path"]).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"E4 checkpoint not found: {checkpoint_path}")
        expected_hash = str(config["model"]["checkpoint_sha256"]).lower()
        actual_hash = sha256_file(checkpoint_path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Checkpoint SHA-256 mismatch: expected {expected_hash}, got {actual_hash}. Do not use this checkpoint.")

        if args.model_loader.strip().lower() == "auto":
            model = load_e4_model_auto(checkpoint_path, device)
        else:
            model_loader = import_callable(args.model_loader)
            # A custom loader receives E4's model config, not the E9 analysis config.
            e4_config_path = (REPO_ROOT / "configs" / "experiments" / "e4_cutmix.yaml").resolve()
            model = _call_model_loader(model_loader, checkpoint_path, device, e4_config_path, args.trust_loader_config_checkpoint)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("E4 loader did not return a torch model")

        with Image.open(image_path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
        crop_transform, normalized_transform = build_preprocessing()
        cropped_rgb = crop_transform(rgb)
        tensor = normalized_transform(cropped_rgb).unsqueeze(0).to(device)

        # Normal forward obtains a predicted class; Grad-CAM itself is run outside inference_mode.
        with torch.inference_mode():
            output = model(tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if isinstance(output, dict):
                output = output.get("logits")
            if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1 or output.shape[1] != 30:
                raise ValueError("E4 model output must be logits with shape [1,30]")
            raw_predicted_class = int(torch.argmax(output[0]).item())
            raw_logits = output.detach().cpu().numpy().astype(np.float64)
        selected_class = raw_predicted_class if args.target_class is None else args.target_class
        class_mapping_path = Path(config.get("class_mapping", {}).get("path", "data/metadata/class_mapping.json"))
        if not class_mapping_path.is_absolute():
            class_mapping_path = Path.cwd() / class_mapping_path
        class_names = load_class_names(class_mapping_path, num_classes=30)
        if selected_class < 0 or selected_class >= 30:
            raise ValueError("target-class must be in [0,29]")
        weights_digest_before = _model_state_digest(model)
        heatmap = generate_gradcam(model, tensor, target_class=selected_class, target_layer=args.target_layer)
        weights_digest_after = _model_state_digest(model)
        if weights_digest_before != weights_digest_after:
            raise RuntimeError("Grad-CAM changed model weights; refusing to accept this run")
        if heatmap.shape != (224, 224) or not np.isfinite(heatmap).all() or heatmap.min() < 0 or heatmap.max() > 1:
            raise RuntimeError("Grad-CAM output violates the [224,224] finite [0,1] contract")

        import matplotlib.pyplot as plt
        from matplotlib import colormaps
        import matplotlib.image as mpimg
        from PIL import Image as PILImage
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        heatmap_rgb = (colormaps["jet"](heatmap)[..., :3] * 255).astype(np.uint8)
        heatmap_image = PILImage.fromarray(heatmap_rgb).resize(cropped_rgb.size, PILImage.Resampling.BILINEAR)
        overlay = PILImage.blend(cropped_rgb.convert("RGB"), heatmap_image.convert("RGB"), alpha=0.40)
        overlay.save(output_path)
        heatmap_path = output_path.with_name(output_path.stem + "_heatmap.png")
        heatmap_image.save(heatmap_path)

        temperature = float(config["e6"]["temperature"])
        probs = stable_softmax(raw_logits, temperature)[0]
        confidence = float(probs[selected_class])
        summary = {
            "experiment": "E9",
            "dataset_split": args.split,
            "image_filename": image_path.name,
            "predicted_class_id": raw_predicted_class,
            "predicted_label": class_names[raw_predicted_class],
            "true_label": args.true_label,
            "gradcam_target_class_id": int(selected_class),
            "gradcam_target_label": class_names[int(selected_class)],
            "calibrated_probability_for_target_class": confidence,
            "calibrated_top1_confidence": float(probs.max()),
            "e7_status_for_predicted_class": "ACCEPT" if float(probs.max()) >= float(config["e7"]["threshold"]) else "REJECT",
            "temperature": temperature,
            "threshold": float(config["e7"]["threshold"]),
            "checkpoint_sha256": actual_hash,
            "model_weights_unchanged": weights_digest_before == weights_digest_after,
            "model_state_sha256_before_gradcam": weights_digest_before,
            "model_state_sha256_after_gradcam": weights_digest_after,
            "heatmap_shape": list(heatmap.shape),
            "heatmap_min": float(heatmap.min()),
            "heatmap_max": float(heatmap.max()),
            "overlay_path": str(output_path),
            "heatmap_path": str(heatmap_path),
            "notes": "Grad-CAM is qualitative evidence; it does not prove the causal reason for a prediction.",
        }
        metadata_path = output_path.with_suffix(".json")
        metadata_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Overlay: {output_path}\nHeatmap: {heatmap_path}\nMetadata: {metadata_path}")
    except Exception as exc:
        print(f"Grad-CAM ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
