"""Contract tests for the Task 06 demo, independent of private images."""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.gradio_app import DemoController, build_app
from app.inference_service import DemoError, InferenceService, TEMPERATURE, THRESHOLD, _artifacts, read_image
from src.selective.threshold import accept_mask, calibrated_predictions


class TinyClassifier(nn.Module):
    def __init__(self, confidence=0.7):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=1)
        self.register_buffer("scores", torch.zeros(30))
        self.scores[7] = TEMPERATURE * math.log(29 * confidence / (1 - confidence))
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        # Retain a path to the convolution so real Grad-CAM can take gradients.
        contribution = self.conv(x).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return self.scores.unsqueeze(0).expand(x.shape[0], -1) + contribution


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "dog.png"
    Image.fromarray(np.random.default_rng(7).integers(0, 255, (280, 400, 3), dtype=np.uint8)).save(path)
    return path


@pytest.fixture
def service(monkeypatch):
    import app.inference_service as module
    model = TinyClassifier()
    monkeypatch.setattr(module, "load_e4_model", lambda root, checkpoint, device: (model.eval().to(device), "verified"))
    return InferenceService(root=ROOT, device="cpu")


def test_locked_artifacts_and_model_loaded_once(monkeypatch):
    import app.inference_service as module
    calls = []
    monkeypatch.setattr(module, "load_e4_model", lambda root, checkpoint, device: (calls.append(checkpoint) or TinyClassifier().eval(), "verified"))
    service = InferenceService(root=ROOT, device="cpu")
    assert len(calls) == 1
    assert service.temperature == TEMPERATURE and service.threshold == THRESHOLD
    assert len(service.names) == 30 and service.names[7] == "Miniature Poodle"


def test_locked_configuration_rejects_tampering(tmp_path):
    (tmp_path / "data/metadata").mkdir(parents=True)
    (tmp_path / "outputs/metrics/E6").mkdir(parents=True)
    (tmp_path / "outputs/metrics/E7").mkdir(parents=True)
    mapping = json.loads((ROOT / "data/metadata/class_mapping.json").read_text(encoding="utf-8"))
    (tmp_path / "data/metadata/class_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    e6 = json.loads((ROOT / "outputs/metrics/E6/temperature.json").read_text(encoding="utf-8"))
    e7 = json.loads((ROOT / "outputs/metrics/E7/threshold.json").read_text(encoding="utf-8"))
    (tmp_path / "outputs/metrics/E6/temperature.json").write_text(json.dumps(e6), encoding="utf-8")
    e7["threshold"] = 0.5
    (tmp_path / "outputs/metrics/E7/threshold.json").write_text(json.dumps(e7), encoding="utf-8")
    with pytest.raises(DemoError, match="E7"):
        _artifacts(tmp_path)


def test_real_checkpoint_guard_rejects_missing_and_wrong(tmp_path):
    with pytest.raises(DemoError, match="checkpoint"):
        InferenceService(root=ROOT, checkpoint=tmp_path / "missing.pt", device="cpu")
    fake = tmp_path / "fake.pt"
    fake.write_bytes(b"not the approved checkpoint")
    with pytest.raises(DemoError, match="SHA-256"):
        InferenceService(root=ROOT, checkpoint=fake, device="cpu")


def test_prediction_uses_calibration_mapping_and_boundary(service, image_file):
    result = service.predict(image_file)
    assert result["class_id"] == 7 and result["name"] == "Miniature Poodle"
    assert result["confidence"] == pytest.approx(0.7, abs=1e-6)
    assert result["accepted"] is True
    assert len(result["top5"]) == 5
    assert next(iter(result["top5"])) == result["name"]
    assert all(np.isfinite(v) and 0 <= v <= 1 for v in result["top5"].values())
    assert result["latency_ms"] >= 0
    logits = service.model(torch.zeros(1, 3, 224, 224)).detach().numpy()
    probs = calibrated_predictions(logits, TEMPERATURE)["probabilities"][0]
    assert np.isfinite(probs).all() and probs.sum() == pytest.approx(1.0)
    assert accept_mask([THRESHOLD - 1e-10, THRESHOLD, THRESHOLD + 1e-10], THRESHOLD).tolist() == [False, True, True]
    service.model = TinyClassifier(confidence=0.45).eval()
    assert service.predict(image_file)["accepted"] is False


def test_gradcam_real_module_uses_same_model_and_rejects_stale(service, image_file, tmp_path):
    prediction = service.predict(image_file)
    before = {k: v.clone() for k, v in service.model.state_dict().items()}
    overlay = service.explain(image_file, prediction)
    assert overlay.size == (224, 224) and overlay.mode == "RGB"
    assert all(torch.equal(before[k], v) for k, v in service.model.state_dict().items())
    assert all(p.grad is None for p in service.model.parameters())
    other = tmp_path / "other.png"
    Image.new("RGB", (100, 100), "red").save(other)
    with pytest.raises(DemoError, match="đã thay đổi"):
        service.explain(other, prediction)
    with pytest.raises(DemoError, match="dự đoán ảnh trước"):
        service.explain(image_file, None)


def test_invalid_inputs_and_format(tmp_path, monkeypatch):
    with pytest.raises(DemoError, match="tải ảnh"):
        read_image(None)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"broken")
    with pytest.raises(DemoError, match="Không đọc được"):
        read_image(bad)
    gif = tmp_path / "image.gif"
    Image.new("RGB", (20, 20)).save(gif)
    with pytest.raises(DemoError, match="JPG, PNG hoặc WebP"):
        read_image(gif)
    import app.inference_service as module
    monkeypatch.setattr(module, "MAX_FILE_BYTES", 10)
    with pytest.raises(DemoError, match="10 MB"):
        read_image(gif)


def test_pixel_limit_is_checked_before_decode(tmp_path, monkeypatch):
    import app.inference_service as module
    image = tmp_path / "large.png"
    Image.new("RGB", (20, 20)).save(image)
    monkeypatch.setattr(module, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(DemoError, match="Kích thước ảnh"):
        read_image(image)


def test_webp_supported(service, tmp_path):
    image = tmp_path / "example.webp"
    Image.new("RGB", (64, 48), "brown").save(image, "WEBP")
    decoded, _ = read_image(image)
    assert decoded.mode == "RGB" and decoded.size == (64, 48)
    assert service.predict(image)["class_id"] == 7


def test_controller_clears_on_error_and_reset(service, image_file):
    controller = DemoController(service)
    result = controller.predict(str(image_file))
    assert "ACCEPT" in result[0] and result[1] == "Miniature Poodle"
    assert result[-1]["image_digest"]
    errored = controller.predict(None)
    assert "tải ảnh" in errored[0] and errored[-1] is None and errored[4] == {}
    assert controller.reset()[-1] is None and controller.reset()[4] == {}
    service.model = TinyClassifier(confidence=0.45).eval()
    rejected = controller.predict(str(image_file))
    assert "REJECT" in rejected[0] and rejected[1] == "Chưa đủ tự tin để kết luận"
    assert len(rejected[4]) == 5


def test_gradio_ui_constructs():
    app = build_app(startup_error="thiếu checkpoint")
    assert app is not None


def test_upload_validation_clears_preview_and_old_outputs(service, tmp_path, image_file):
    controller = DemoController(service)
    valid = controller.select_image(str(image_file))
    assert valid[0].mode == "RGB"
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    invalid = controller.select_image(str(broken))
    assert invalid[0] is None and "Không đọc được ảnh" in invalid[1]
    assert invalid[2:5] == ("", "", "") and invalid[-1] is None


def test_corrupted_images_through_actual_gradio_http(service, tmp_path, image_file):
    """Catch failures occurring before the controller, not just in read_image."""
    from gradio_client import Client, handle_file
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    truncated = tmp_path / "truncated.png"
    data = image_file.read_bytes()
    truncated.write_bytes(data[:len(data) // 2])
    demo = build_app(service)
    try:
        _, url, _ = demo.queue().launch(server_name="127.0.0.1", share=False,
                                       prevent_thread_lock=True, quiet=True)
        client = Client(url, verbose=False)
        assert client.predict(handle_file(str(image_file)), api_name="/predict")[1] == "Miniature Poodle"
        for image in (broken, truncated):
            result = client.predict(handle_file(str(image)), api_name="/predict")
            assert "Không đọc được ảnh" in result[0]
            assert result[1:4] == ("", "", "")
            assert not result[4] and result[5] is None
        # The same session remains usable after the invalid uploads.
        assert client.predict(handle_file(str(image_file)), api_name="/predict")[1] == "Miniature Poodle"
    finally:
        demo.close(verbose=False)


def test_concurrent_image_change_suppresses_old_result(service, image_file):
    from types import SimpleNamespace
    import gradio as gr
    controller = DemoController(service)
    started, release = threading.Event(), threading.Event()
    original = service.predict

    def slow_predict(path):
        started.set()
        assert release.wait(5)
        return original(path)

    service.predict = slow_predict
    request = SimpleNamespace(session_hash="session-one")
    result = []
    thread = threading.Thread(target=lambda: result.append(controller.predict(str(image_file), request)))
    thread.start()
    assert started.wait(5)
    controller.reset(None, request)
    # Changes in another session must not change this session's generation.
    assert controller._revision(SimpleNamespace(session_hash="session-two")) == 0
    release.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert result[0] == tuple(gr.skip() for _ in range(8))


def test_delayed_upload_event_for_same_image_does_not_suppress_prediction(service, image_file):
    from types import SimpleNamespace
    import gradio as gr
    controller = DemoController(service)
    started, release = threading.Event(), threading.Event()
    original = service.predict

    def slow_predict(path):
        started.set()
        assert release.wait(5)
        return original(path)

    service.predict = slow_predict
    request = SimpleNamespace(session_hash="same-image")
    output = []
    thread = threading.Thread(target=lambda: output.append(controller.predict(str(image_file), request)))
    thread.start()
    assert started.wait(5)
    assert controller.reset(str(image_file), request) == tuple(gr.skip() for _ in range(8))
    release.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert output[0][1] == "Miniature Poodle"
    assert output[0][-1]["image_digest"]
