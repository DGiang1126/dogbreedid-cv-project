"""Contract tests for the Task 06 demo, independent of private images."""
from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch import nn
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.gradio_app import DemoController, build_app
from app.inference_service import (
    FIT_MODE, PADDING_RGB, STANDARD_MODE, TEMPERATURE, THRESHOLD,
    DemoError, InferenceService, _artifacts, read_image,
)
from src.data.transforms import build_transforms
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


def test_standard_is_default_and_bit_exact_with_frozen_e4(service, image_file):
    config = yaml.safe_load((ROOT / "configs/data.yaml").read_text(encoding="utf-8"))
    image, _ = read_image(image_file)
    expected = build_transforms(config, "inference")(image).unsqueeze(0)
    seen = []
    with service.model.register_forward_pre_hook(lambda model, args: seen.append(args[0].clone())):
        implicit = service.predict(image_file)
        explicit = service.predict(image_file, STANDARD_MODE)
    assert len(seen) == 2 and all(torch.equal(tensor, expected) for tensor in seen)
    assert implicit["mode"] == explicit["mode"] == STANDARD_MODE
    assert implicit["image_digest"] == hashlib.sha256(image_file.read_bytes()).hexdigest()
    assert implicit["top5"] == explicit["top5"]
    preview = service.preview(image)
    to_model = transforms.Compose(service.transform.transforms[-2:])
    assert torch.equal(to_model(preview).unsqueeze(0), expected)


def test_standard_preview_uses_decoded_pixels_without_display_metadata(service, tmp_path):
    path = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (300, 240), "blue")
    image.paste("red", (0, 0, 300, 100))
    exif = Image.Exif()
    exif[274] = 6  # The display orientation would rotate the decoded pixels.
    image.save(path, exif=exif)
    decoded, _ = read_image(path)
    assert decoded.getexif().get(274) == 6
    preview = service.preview(decoded)
    assert not preview.getexif()
    np.testing.assert_array_equal(preview, service.preview_transform(decoded))


@pytest.mark.parametrize(("size", "fitted_size"), [
    ((750, 402), (224, 120)),  # Both side markers would be cropped in Standard.
    ((300, 751), (89, 224)),   # Odd padding with markers at the top and bottom.
    ((300, 300), (224, 224)),
    ((12000, 1), (224, 1)),    # Rounding must never create a zero-sized edge.
])
def test_fit_preserves_both_ends_and_aspect_ratio(service, size, fitted_size):
    image = Image.new("RGB", size, "green")
    width, height = size
    if width >= height:
        image.paste("red", (0, 0, width // 4, height))
        image.paste("blue", (width * 3 // 4, 0, width, height))
    else:
        image.paste("red", (0, 0, width, height // 4))
        image.paste("blue", (0, height * 3 // 4, width, height))
    preview = service.preview(image, FIT_MODE)
    assert preview.size == (224, 224) and preview.mode == "RGB"
    y, x = np.where(np.any(np.asarray(preview) != PADDING_RGB, axis=-1))
    fit_width, fit_height = fitted_size
    assert (x.min(), y.min()) == ((224 - fit_width) // 2, (224 - fit_height) // 2)
    assert (x.max() - x.min() + 1, y.max() - y.min() + 1) == fitted_size
    assert preview.getpixel((x.min(), y.min())) == (255, 0, 0)
    assert preview.getpixel((x.max(), y.max())) == (0, 0, 255)


def test_invalid_mode_is_rejected(service, image_file):
    image, _ = read_image(image_file)
    for call in (
        lambda: service.preview(image, "stretch"),
        lambda: service.predict(image_file, "stretch"),
        lambda: service.explain(image_file, None, "stretch"),
    ):
        with pytest.raises(DemoError, match="Chế độ xử lý ảnh không hợp lệ"):
            call()


@pytest.mark.parametrize("mode", [STANDARD_MODE, FIT_MODE])
def test_prediction_uses_calibration_mapping_and_boundary(service, image_file, mode):
    result = service.predict(image_file, mode)
    assert result["class_id"] == 7 and result["name"] == "Miniature Poodle"
    assert result["mode"] == mode
    assert result["confidence"] == pytest.approx(0.7, abs=1e-6)
    assert result["accepted"] is True
    assert len(result["top5"]) == 5
    assert next(iter(result["top5"])) == result["name"]
    assert all(np.isfinite(v) and 0 <= v <= 1 for v in result["top5"].values())
    assert result["latency_ms"] >= 0
    logits = service.model(torch.zeros(1, 3, 224, 224)).detach().numpy()
    probs = calibrated_predictions(logits, TEMPERATURE)["probabilities"][0]
    assert np.isfinite(probs).all() and probs.sum() == pytest.approx(1.0)
    top_ids = [7, 0, 1, 2, 3]
    assert list(result["top5"]) == [service.names[i] for i in top_ids]
    assert list(result["top5"].values()) == pytest.approx(probs[top_ids])
    assert accept_mask([THRESHOLD - 1e-10, THRESHOLD, THRESHOLD + 1e-10], THRESHOLD).tolist() == [False, True, True]
    service.model = TinyClassifier(confidence=0.45).eval()
    assert service.predict(image_file, mode)["accepted"] is False


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
    with pytest.raises(DemoError, match="Chế độ xử lý đã thay đổi"):
        service.explain(image_file, prediction, FIT_MODE)


def test_fit_gradcam_uses_identical_model_tensor_and_preview(service, image_file, monkeypatch):
    from matplotlib import colormaps
    import app.inference_service as module

    generate = module.generate_gradcam
    heatmaps = []

    def record_gradcam(model, tensor, *, target_class):
        assert model is service.model and target_class == 7
        heatmaps.append(generate(model, tensor, target_class=target_class))
        return heatmaps[-1]

    monkeypatch.setattr(module, "generate_gradcam", record_gradcam)
    seen = []
    with service.model.register_forward_pre_hook(lambda model, args: seen.append(args[0].detach().clone())):
        prediction = service.predict(image_file, FIT_MODE)
        overlay = service.explain(image_file, prediction, FIT_MODE)
    assert len(seen) == 2 and seen[0].shape == (1, 3, 224, 224)
    assert torch.equal(seen[0], seen[1])
    image, _ = read_image(image_file)
    preview = service.preview(image, FIT_MODE)
    to_model = transforms.Compose(service.transform.transforms[-2:])
    assert torch.equal(seen[0], to_model(preview).unsqueeze(0))
    colors = colormaps["jet"](heatmaps[0])[..., :3]
    expected = (np.clip(0.55 * (np.asarray(preview, dtype=np.float32) / 255) + 0.45 * colors, 0, 1) * 255)
    assert overlay.size == (224, 224) and overlay.mode == "RGB"
    np.testing.assert_array_equal(np.asarray(overlay), expected.astype(np.uint8))


def test_invalid_inputs_and_format(service, tmp_path, monkeypatch):
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
        service.predict(gif, FIT_MODE)


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
    cleared = controller.clear()
    assert cleared[-1] is None and cleared[7] == {}
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
    assert valid[1].size == (224, 224)
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    invalid = controller.select_image(str(broken), FIT_MODE)
    assert invalid[:2] == (None, None) and "Không đọc được ảnh" in invalid[2]
    assert invalid[3:6] == ("", "", "") and invalid[-1] is None


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
        image, _ = read_image(image_file)
        to_model = transforms.Compose(service.transform.transforms[-2:])
        seen = []
        with service.model.register_forward_pre_hook(lambda model, args: seen.append(args[0].clone())):
            standard = client.predict(handle_file(str(image_file)), api_name="/predict")
            fit = client.predict(handle_file(str(image_file)), mode="fit", api_name="/predict")
        for mode, result, tensor in zip((STANDARD_MODE, FIT_MODE), (standard, fit), seen, strict=True):
            assert result[1] == "Miniature Poodle" and len(result) == 7
            assert torch.equal(tensor, to_model(service.preview(image, mode)).unsqueeze(0))
        assert not torch.equal(seen[0], seen[1])
        for image in (broken, truncated):
            result = client.predict(handle_file(str(image)), FIT_MODE, api_name="/predict")
            assert "Không đọc được ảnh" in result[0]
            assert result[1:4] == ("", "", "")
            assert not result[4] and result[5] is None
        # The same session remains usable after the invalid uploads.
        assert client.predict(handle_file(str(image_file)), api_name="/predict")[1] == "Miniature Poodle"
    finally:
        demo.close(verbose=False)


@pytest.mark.parametrize("change", ["clear", "image", "mode", "replaced", "corrupt"])
@pytest.mark.parametrize("work", ["predict", "explain"])
def test_inflight_result_cannot_restore_invalidated_outputs(service, image_file, monkeypatch, change, work):
    controller = DemoController(service)
    request = SimpleNamespace(session_hash="changing")
    path = str(image_file)
    prediction = controller.predict(path, FIT_MODE, request)[-1]
    started, release = threading.Event(), threading.Event()
    method = "predict_image" if work == "predict" else "explain"
    original = getattr(service, method)

    def slow_inference(*args):
        started.set()
        assert release.wait(10)
        return original(*args)

    monkeypatch.setattr(service, method, slow_inference)
    with ThreadPoolExecutor(max_workers=1) as pool:
        if work == "predict":
            future = pool.submit(controller.predict, path, FIT_MODE, request)
        else:
            future = pool.submit(controller.explain, path, prediction, FIT_MODE, request)
        try:
            assert started.wait(5)
            status = "Tải ảnh lên rồi nhấn **Dự đoán**."
            if change == "clear":
                cleared = controller.clear(request)
                assert len(cleared) == 11 and cleared[:3] == (None, None, None)
                cleared = cleared[3:]
            elif change == "image":
                other = image_file.with_name("other.png")
                Image.new("RGB", (200, 300), "red").save(other)
                selected = controller.select_image(str(other), FIT_MODE, request)
                assert selected[0].size == (200, 300) and selected[1].size == (224, 224)
                cleared = selected[2:]
            elif change == "mode":
                selected = controller.select_image(path, STANDARD_MODE, request)
                assert selected[0].mode == "RGB" and selected[1].size == (224, 224)
                cleared = selected[2:]
            elif change == "replaced":
                Image.new("RGB", (100, 80), "red").save(image_file)
                cleared = controller.select_image(path, FIT_MODE, request)[2:]
                status = "Nội dung ảnh đã thay đổi"
            else:
                image_file.write_bytes(b"broken")
                cleared = controller.select_image(path, FIT_MODE, request)[2:]
                status = "Không đọc được ảnh"
            assert status in cleared[0]
            assert cleared[1:] == ("", "", "", {}, None, "", None)
            assert controller._revision(SimpleNamespace(session_hash="independent"))[0] == 0
        finally:
            release.set()
        assert future.result(timeout=15) == (gr.skip(),) * (8 if work == "predict" else 2)


def test_changing_mode_clears_results_and_requires_new_prediction(service, image_file):
    controller = DemoController(service)
    request = SimpleNamespace(session_hash="mode-change")
    fit = controller.select_image(image_file, FIT_MODE, request)[1]
    prediction = controller.predict(image_file, FIT_MODE, request)[-1]
    selected = controller.select_image(image_file, STANDARD_MODE, request)
    assert not np.array_equal(fit, selected[1])
    assert selected[-1] is None
    overlay, message = controller.explain(image_file, prediction, STANDARD_MODE, request)
    assert overlay is None and "dự đoán lại" in message
    # Returning to the original selection must not reactivate the old result.
    controller.select_image(image_file, FIT_MODE, request)
    overlay, message = controller.explain(image_file, prediction, FIT_MODE, request)
    assert overlay is None and "dự đoán lại" in message
    fresh = controller.predict(image_file, FIT_MODE, request)[-1]
    overlay, _ = controller.explain(image_file, fresh, FIT_MODE, request)
    assert overlay.size == (224, 224)


def test_delayed_upload_event_for_same_image_and_mode_preserves_prediction(service, image_file, monkeypatch):
    controller = DemoController(service)
    request = SimpleNamespace(session_hash="same-selection")
    started, release = threading.Event(), threading.Event()
    original = service.predict_image

    def slow_predict(image, digest, mode):
        started.set()
        assert release.wait(10)
        return original(image, digest, mode)

    monkeypatch.setattr(service, "predict_image", slow_predict)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(controller.predict, image_file, FIT_MODE, request)
        try:
            assert started.wait(5)
            selected = controller.select_image(image_file, FIT_MODE, request)
            assert selected[0].mode == "RGB" and selected[1].size == (224, 224)
            assert selected[2:] == (gr.skip(),) * 8
        finally:
            release.set()
        result = future.result(timeout=15)
    assert result[1] == "Miniature Poodle" and result[-1]["mode"] == FIT_MODE


@pytest.mark.parametrize("replacement", ["valid", "corrupt"])
def test_replaced_file_at_same_path_invalidates_prediction(service, image_file, replacement):
    controller = DemoController(service)
    request = SimpleNamespace(session_hash="replaced-file")
    prediction = controller.predict(image_file, FIT_MODE, request)[-1]
    assert prediction is not None

    if replacement == "valid":
        Image.new("RGB", (100, 80), "red").save(image_file)
    else:
        image_file.write_bytes(b"broken")

    selected = controller.select_image(image_file, FIT_MODE, request)
    if replacement == "valid":
        assert selected[0].size == (100, 80) and selected[1].size == (224, 224)
        assert "dự đoán lại" in selected[2].lower()
    else:
        assert selected[:2] == (None, None) and "Không đọc được ảnh" in selected[2]
    assert selected[3:] == ("", "", "", {}, None, "", None)
    assert controller._revision(request)[0] != prediction["revision"]


@pytest.mark.parametrize("action", ["predict", "clear"])
def test_preview_finishing_late_respects_current_selection(service, image_file, monkeypatch, action):
    controller = DemoController(service)
    request = SimpleNamespace(session_hash="late-preview")
    started, release = threading.Event(), threading.Event()
    original = service.preview

    def slow_preview(image, mode):
        started.set()
        assert release.wait(10)
        return original(image, mode)

    monkeypatch.setattr(service, "preview", slow_preview)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(controller.select_image, image_file, FIT_MODE, request)
        try:
            assert started.wait(5)
            if action == "predict":
                assert controller.predict(image_file, FIT_MODE, request)[-1]["mode"] == FIT_MODE
            else:
                controller.clear(request)
        finally:
            release.set()
        selected = future.result(timeout=15)
    assert selected[2:] == (gr.skip(),) * 8
    if action == "predict":
        assert selected[1].size == (224, 224)
    else:
        assert selected[:2] == (gr.skip(), gr.skip())


def test_gradio_event_contract():
    config = build_app(startup_error="test").get_config_file()
    components = {component["id"]: component for component in config["components"]}
    events = {event["api_name"]: event for event in config["dependencies"] if event["backend_fn"]}
    predict, explain, select, clear = (events[name] for name in ("predict", "explain", "select_image", "clear"))

    def signature(ids):
        return [(components[i]["type"], components[i]["props"].get("label")) for i in ids]

    assert predict["api_visibility"] == "public"
    assert signature(predict["inputs"]) == [
        ("file", "Chọn ảnh JPG, PNG hoặc WebP"), ("radio", "Chế độ xử lý ảnh"),
    ]
    mode = components[predict["inputs"][1]]["props"]
    assert mode["value"] == STANDARD_MODE
    assert mode["choices"] == [("Chuẩn E4 — CenterCrop", STANDARD_MODE), ("Giữ trọn ảnh — Padding", FIT_MODE)]
    assert signature(predict["outputs"]) == [
        ("markdown", None),
        ("textbox", "Giống dự đoán"),
        ("textbox", "Độ tin cậy đã hiệu chỉnh"),
        ("textbox", "Thời gian chạy mô hình"),
        ("label", "5 giống có khả năng cao nhất"),
        ("image", "Grad-CAM trên vùng ảnh mô hình quan sát"),
        ("markdown", None),
        ("state", None),
    ]
    file_id, mode_id = predict["inputs"]
    assert select["inputs"] == [file_id, mode_id]
    assert set(map(tuple, select["targets"])) == {(file_id, "upload"), (file_id, "clear"), (mode_id, "input")}
    assert signature(select["outputs"][:2]) == [
        ("image", "Ảnh của bạn"), ("image", "Ảnh 224 × 224 thực sự đưa vào mô hình"),
    ]
    original_preview, model_preview = (components[i]["props"] for i in select["outputs"][:2])
    assert original_preview["visible"] is True and model_preview["visible"] is False
    assert not any(component["type"] == "accordion" for component in components.values())
    layout_ids = {component["props"].get("elem_id"): i for i, component in components.items()}
    demo_layout = next(node for node in config["layout"]["children"] if node["id"] == layout_ids["dogbreed-demo"])
    demo_children = {node["id"]: node for node in demo_layout["children"]}
    main_id, gradcam_id = (layout_ids[name] for name in ("dogbreed-main", "dogbreed-gradcam"))
    assert list(demo_children).index(gradcam_id) > list(demo_children).index(main_id)
    assert components[gradcam_id]["type"] == "column"
    assert components[gradcam_id]["props"]["visible"] is True
    gradcam_children = {node["id"] for node in demo_children[gradcam_id]["children"]}
    assert set(explain["outputs"]) <= gradcam_children
    assert all(components[i]["props"]["visible"] is True for i in explain["outputs"])
    assert select["outputs"][2:] == predict["outputs"]
    assert clear["inputs"] == [] and clear["outputs"] == [file_id, *select["outputs"]]
    assert explain["inputs"] == [file_id, predict["outputs"][-1], mode_id]
    assert explain["outputs"] == predict["outputs"][5:7]
    assert all(event["api_visibility"] == "private" for event in (select, clear, explain))
    assert select["queue"] is False and clear["queue"] is False
    cancels = {tuple(map(tuple, event["targets"])): event for event in config["dependencies"] if event["cancels"]}
    for event in (select, clear):
        cancellation = cancels[tuple(map(tuple, event["targets"]))]
        assert cancellation["api_visibility"] == "private"
        assert cancellation["cancels"] == [predict["id"], explain["id"]]


# ---------------- Task 07: automatic Energy OOD detection ----------------

from app.inference_service import (  # noqa: E402
    OOD_MESSAGE, OOD_THRESHOLD, STATUS_ACCEPT, STATUS_E7_REJECT, STATUS_OOD_REJECT,
)


class OutOfScopeClassifier(TinyClassifier):
    """All logits very low: confident top-1 for E7 but high energy (out of scope)."""

    def __init__(self):
        super().__init__()
        self.scores[:] = -10.0
        self.scores[7] = 0.0


def _copy_locked_artifacts(dst, edit=None):
    for rel in ("data/metadata/class_mapping.json", "outputs/metrics/E6/temperature.json",
                "outputs/metrics/E7/threshold.json", "outputs/metrics/OOD/energy_threshold.json"):
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        data = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        if edit and rel.endswith("energy_threshold.json"):
            edit(data)
        (dst / rel).write_text(json.dumps(data), encoding="utf-8")


def test_ood_threshold_artifact_is_locked_and_loaded(service):
    artifact = json.loads((ROOT / "outputs/metrics/OOD/energy_threshold.json").read_text(encoding="utf-8"))
    assert artifact["split"] == "calibration" and artifact["ood_200_used"] is False
    assert artifact["threshold"] == OOD_THRESHOLD
    assert service.ood_detector.threshold == OOD_THRESHOLD and service.ood_detector.temperature == 1.0
    # E6/E7 stay untouched by the OOD integration.
    assert service.temperature == TEMPERATURE and service.threshold == THRESHOLD


@pytest.mark.parametrize("edit, missing", [
    (lambda d: d.update(threshold=0.0), False),
    (lambda d: d.update(ood_200_used=True), False),
    (lambda d: d.update(split="ood_200"), False),
    (lambda d: d.update(energy_temperature=TEMPERATURE), False),
    (None, True),
])
def test_ood_artifact_tampering_is_rejected(tmp_path, monkeypatch, edit, missing):
    import app.inference_service as module
    monkeypatch.setattr(module, "load_e4_model", lambda root, checkpoint, device: (TinyClassifier().eval(), "ok"))
    _copy_locked_artifacts(tmp_path, edit)
    if missing:
        (tmp_path / "outputs/metrics/OOD/energy_threshold.json").unlink()
    with pytest.raises(DemoError, match="OOD|cấu hình"):
        InferenceService(root=tmp_path, device="cpu")


@pytest.mark.parametrize("mode", [STANDARD_MODE, FIT_MODE])
def test_three_decision_states_and_energy_diagnostics(service, image_file, mode):
    seen = []
    with service.model.register_forward_hook(lambda m, args, out: seen.append(out.detach().clone())):
        accepted = service.predict(image_file, mode)
    assert accepted["status"] == STATUS_ACCEPT and accepted["ood"]["is_ood"] is False
    logits = seen[0].numpy()
    service.model = TinyClassifier(confidence=0.45).eval()
    e7 = service.predict(image_file, mode)
    assert e7["status"] == STATUS_E7_REJECT and e7["accepted"] is False and e7["ood"]["is_ood"] is False
    service.model = OutOfScopeClassifier().eval()
    ood = service.predict(image_file, mode)
    # OOD has priority even when E7 alone would accept the prediction.
    assert ood["accepted"] is True and ood["status"] == STATUS_OOD_REJECT
    assert ood["ood"]["energy"] > OOD_THRESHOLD and ood["ood"]["threshold"] == OOD_THRESHOLD
    assert accepted["ood"]["energy"] == pytest.approx(-float(torch.logsumexp(torch.tensor(logits), 1)), abs=1e-5)


def test_ood_reject_hides_top1_top5_and_gradcam(service, image_file):
    controller = DemoController(service)
    service.model = OutOfScopeClassifier().eval()
    result = controller.predict(str(image_file))
    status, breed, confidence, latency, top5, overlay, explain_text, state = result
    assert "OOD REJECT" in status and OOD_MESSAGE in status
    assert OOD_MESSAGE == ("Ảnh có thể nằm ngoài phạm vi 30 giống chó được hỗ trợ. "
                           "Hệ thống không đưa ra kết luận phân loại cho ảnh này.")
    assert breed == "" and confidence == "" and top5 == {} and overlay is None and explain_text == ""
    assert latency.endswith("ms") and state["status"] == STATUS_OOD_REJECT
    assert "Miniature Poodle" not in "".join(map(str, result[:7]))
    cam, message = controller.explain(str(image_file), state)
    assert cam is None and "ngoài phạm vi" in message
    with pytest.raises(DemoError, match="ngoài phạm vi"):
        service.explain(image_file, state)


def test_e7_reject_keeps_its_own_meaning(service, image_file):
    service.model = TinyClassifier(confidence=0.45).eval()
    result = DemoController(service).predict(str(image_file))
    assert "E7 REJECT" in result[0] and "OOD" not in result[0] and OOD_MESSAGE not in result[0]
    assert result[1] == "Chưa đủ tự tin để kết luận" and len(result[4]) == 5


def test_ood_outputs_do_not_leak_after_clear_mode_change_or_new_image(service, image_file, tmp_path):
    controller = DemoController(service)
    request = SimpleNamespace(session_hash="ood-session")
    service.model = OutOfScopeClassifier().eval()
    assert "OOD REJECT" in controller.predict(str(image_file), STANDARD_MODE, request)[0]
    cleared = controller.select_image(str(image_file), FIT_MODE, request)
    assert cleared[2] == "Tải ảnh lên rồi nhấn **Dự đoán**." and cleared[-1] is None
    assert controller.clear(request)[-1] is None
    service.model = TinyClassifier().eval()
    other = tmp_path / "dog2.png"
    Image.new("RGB", (300, 300), "brown").save(other)
    controller.select_image(str(other), STANDARD_MODE, request)
    fresh = controller.predict(str(other), STANDARD_MODE, request)
    assert "ACCEPT" in fresh[0] and fresh[1] == "Miniature Poodle" and fresh[-1]["status"] == STATUS_ACCEPT
    overlay, _ = controller.explain(str(other), fresh[-1], STANDARD_MODE, request)
    assert overlay is not None


def test_ood_state_is_isolated_between_sessions(service, image_file):
    controller = DemoController(service)
    a, b = SimpleNamespace(session_hash="A"), SimpleNamespace(session_hash="B")
    service.model = OutOfScopeClassifier().eval()
    ood_state = controller.predict(str(image_file), STANDARD_MODE, a)[-1]
    service.model = TinyClassifier().eval()
    id_state = controller.predict(str(image_file), STANDARD_MODE, b)[-1]
    assert ood_state["status"] == STATUS_OOD_REJECT and id_state["status"] == STATUS_ACCEPT
    assert controller.explain(str(image_file), ood_state, STANDARD_MODE, a)[0] is None
    assert controller.explain(str(image_file), id_state, STANDARD_MODE, b)[0] is not None
