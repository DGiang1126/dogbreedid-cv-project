import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explainability.gradcam import find_last_conv2d, generate_gradcam, resolve_target_layer


class TinyFrozenCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(5, 7, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(7, num_classes)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


def test_find_and_resolve_last_convolutional_layer():
    model = TinyFrozenCNN()
    name, layer = find_last_conv2d(model)
    assert name == "features.2"
    assert layer is model.features[2]
    assert resolve_target_layer(model, "features.2") is layer


def test_gradcam_runs_with_frozen_parameters_and_does_not_change_weights():
    torch.manual_seed(42)
    model = TinyFrozenCNN().eval()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    x = torch.randn(1, 3, 224, 224)
    cam = generate_gradcam(model, x, target_class=1)
    assert cam.shape == (224, 224)
    assert cam.dtype == np.float32
    assert np.isfinite(cam).all()
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    after = model.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_gradcam_top1_target_and_named_module():
    model = TinyFrozenCNN().eval()
    x = torch.randn(1, 3, 56, 56)
    cam = generate_gradcam(model, x, target_class=None, target_layer="features.2")
    assert cam.shape == (56, 56)


def test_gradcam_rejects_invalid_inputs_and_target_class():
    model = TinyFrozenCNN().eval()
    with pytest.raises(ValueError, match="shape"):
        generate_gradcam(model, torch.randn(3, 224, 224), 0)
    with pytest.raises(ValueError, match="target_class"):
        generate_gradcam(model, torch.randn(1, 3, 32, 32), target_class=99)
    with pytest.raises(ValueError, match="not found"):
        generate_gradcam(model, torch.randn(1, 3, 32, 32), target_class=0, target_layer="no_such_layer")
    with pytest.raises(ValueError, match="NaN or Inf"):
        generate_gradcam(model, torch.full((1, 3, 32, 32), float("nan")), target_class=0)


def test_gradcam_refuses_inference_mode():
    model = TinyFrozenCNN().eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="requires autograd"):
            generate_gradcam(model, x, target_class=0)


APPROVED_SHA = "73df0cc1c5b16866d4facc910f3e4d3ab481e3f12f1681a59fa7d61e3845c47b"


@pytest.mark.parametrize("entrypoint", ["load_e4_model_auto", "load_e4_model_native"])
def test_gradcam_loaders_delegate_to_verified_shared_loader(tmp_path, monkeypatch, entrypoint):
    import scripts.generate_gradcam as cli
    import src.inference.predictor as predictor
    model = TinyFrozenCNN(num_classes=30)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    calls = []
    def shared(root, checkpoint, device):
        calls.append((root, checkpoint, device))
        return model, APPROVED_SHA
    monkeypatch.setattr(predictor, "load_e4_model", shared)
    def forbidden(*args, **kwargs):
        pytest.fail("No native deserializer, pretrained download or guessed model factory is permitted")
    monkeypatch.setattr(cli, "_build_model_through_factory", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    device = torch.device("cpu")
    checkpoint = tmp_path / "fake.pt"
    result = getattr(cli, entrypoint)(checkpoint, device)
    assert result is model and not result.training
    assert calls == [(cli.REPO_ROOT, checkpoint, device)]
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


@pytest.mark.parametrize("error", [ValueError("Checkpoint SHA-256 mismatch"), TypeError("weights_only unsupported"), RuntimeError("strict state_dict mismatch")])
def test_shared_loader_failures_propagate_without_fallback(tmp_path, monkeypatch, error):
    import scripts.generate_gradcam as cli
    import src.inference.predictor as predictor
    calls = []
    def fail(*args):
        calls.append(args)
        raise error
    monkeypatch.setattr(predictor, "load_e4_model", fail)
    with pytest.raises(type(error), match=str(error)):
        cli.load_e4_model_auto(tmp_path / "fake.pt", torch.device("cpu"))
    assert len(calls) == 1


def test_auto_rejects_unapproved_verified_hash(tmp_path, monkeypatch):
    import scripts.generate_gradcam as cli
    import src.inference.predictor as predictor
    monkeypatch.setattr(predictor, "load_e4_model", lambda *args: (TinyFrozenCNN(30), "0" * 64))
    with pytest.raises(RuntimeError, match="unapproved checkpoint"):
        cli.load_e4_model_auto(tmp_path / "fake.pt", torch.device("cpu"))


def test_shared_loader_uses_safe_deserialization_strict_state_and_no_pretrained(tmp_path, monkeypatch):
    import src.inference.predictor as predictor
    import src.models.model_factory as factory
    model = TinyFrozenCNN(30)
    config = {"experiment": {"id": "E4"}, "model": {"type": "mobilenet_v2", "num_classes": 30, "weights": "imagenet"}}
    calls = []
    monkeypatch.setattr(predictor, "verify_checkpoint_hash", lambda path: APPROVED_SHA)
    monkeypatch.setattr(predictor, "_load_yaml", lambda path: config)
    def build(architecture):
        assert architecture["model"]["weights"] == "none"
        assert config["model"]["weights"] == "imagenet"  # Only a deep copy is changed.
        return model
    monkeypatch.setattr(factory, "build_model", build)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    def load(path, *, map_location, weights_only):
        assert weights_only is True and map_location == "cpu"
        calls.append(path)
        return {"model_state_dict": state}
    monkeypatch.setattr(torch, "load", load)
    real_load_state = model.load_state_dict
    def strict_load(values, *, strict):
        assert strict is True
        return real_load_state(values, strict=strict)
    monkeypatch.setattr(model, "load_state_dict", strict_load)
    result, verified_hash = predictor.load_e4_model(ROOT, tmp_path / "fake.pt", torch.device("cpu"))
    assert result is model and verified_hash == APPROVED_SHA
    assert len(calls) == 1 and not model.training


def test_gradcam_cli_auto_with_fake_image_and_checkpoint(tmp_path, monkeypatch):
    import yaml
    from PIL import Image
    import scripts.generate_gradcam as cli
    import src.inference.predictor as predictor
    image = tmp_path / "synthetic.png"
    Image.new("RGB", (320, 280), color=(100, 150, 200)).save(image)
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"fake checkpoint; shared loader mocked")
    config_data = yaml.safe_load((ROOT / "configs/experiments/e9_error_analysis.yaml").read_text(encoding="utf-8"))
    config = tmp_path / "e9.yaml"
    config.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    monkeypatch.setattr(cli, "sha256_file", lambda path: APPROVED_SHA)
    model = TinyFrozenCNN(30).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loader_calls = []
    def loader(root, path, device):
        loader_calls.append((root, path, device))
        return model, APPROVED_SHA
    monkeypatch.setattr(predictor, "load_e4_model", loader)
    # Retain the real Grad-CAM implementation and validate its context and input.
    real_gradcam = cli.generate_gradcam
    def explain(model, tensor, **kwargs):
        assert not torch.is_inference_mode_enabled()
        assert tuple(tensor.shape) == (1, 3, 224, 224)
        expected = (torch.tensor([100., 150., 200.]) / 255 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
        torch.testing.assert_close(tensor[0, :, 0, 0], expected)
        heatmap = real_gradcam(model, tensor, **kwargs)
        assert heatmap.shape == (224, 224) and np.isfinite(heatmap).all()
        assert 0 <= heatmap.min() <= heatmap.max() <= 1
        return heatmap
    monkeypatch.setattr(cli, "generate_gradcam", explain)
    output = tmp_path / "report/overlay.png"
    monkeypatch.setattr(sys, "argv", ["generate_gradcam.py", "--config", str(config), "--image", str(image), "--checkpoint", str(checkpoint), "--output", str(output), "--split", "synthetic", "--device", "cpu", "--model-loader", "auto"])
    assert cli.main() == 0
    assert len(loader_calls) == 1
    assert output.is_file() and output.with_suffix(".json").is_file()
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
