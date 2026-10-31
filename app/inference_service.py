"""Single-image E4/E6/E7 inference and Grad-CAM for the local demo."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from src.data.transforms import build_transforms
from src.explainability.gradcam import generate_gradcam
from src.inference.predictor import load_e4_model
from src.selective.threshold import accept_mask, calibrated_predictions

ROOT = Path(__file__).resolve().parents[1]
TEMPERATURE = 0.80246283961825
THRESHOLD = 0.5011672918550012
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


class DemoError(Exception):
    """An actionable error which can be shown in the interface."""


def _json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as stream:
            result = json.load(stream)
    except (OSError, ValueError) as exc:
        raise DemoError(f"Không đọc được dữ liệu cấu hình: {path.name}.") from exc
    if not isinstance(result, dict):
        raise DemoError(f"Dữ liệu cấu hình không hợp lệ: {path.name}.")
    return result


def _artifacts(root: Path) -> tuple[list[str], float, float]:
    mapping = _json(root / "data/metadata/class_mapping.json")
    if set(mapping) != {str(i) for i in range(30)} or any(
        not isinstance(v, str) or not v.strip() for v in mapping.values()
    ) or len(set(mapping.values())) != 30:
        raise DemoError("Bảng tên giống chó không hợp lệ.")
    e6 = _json(root / "outputs/metrics/E6/temperature.json")
    e7 = _json(root / "outputs/metrics/E7/threshold.json")
    if (e6.get("experiment") != "E6" or e6.get("split") != "calibration"
        or e6.get("model_experiment") != "E4" or e6.get("checkpoint_seed") != 42
        or e6.get("temperature") != TEMPERATURE):
        raise DemoError("Thông số hiệu chỉnh E6 không khớp cấu hình đã khóa.")
    if (e7.get("experiment") != "E7" or e7.get("split") != "calibration"
        or e7.get("temperature") != TEMPERATURE or e7.get("threshold") != THRESHOLD
        or e7.get("accept_operator") != ">="):
        raise DemoError("Ngưỡng E7 không khớp cấu hình đã khóa.")
    return [mapping[str(i)] for i in range(30)], TEMPERATURE, THRESHOLD


def read_image(path: str | Path | None) -> tuple[Image.Image, str]:
    """Validate and decode one JPEG, PNG or WebP without accepting unbounded input."""
    if not path:
        raise DemoError("Vui lòng tải ảnh JPG, PNG hoặc WebP trước.")
    try:
        file = Path(path)
        if not file.is_file():
            raise DemoError("Không tìm thấy ảnh. Vui lòng tải ảnh lên lại.")
        if file.stat().st_size > MAX_FILE_BYTES:
            raise DemoError("Ảnh vượt quá giới hạn 10 MB. Vui lòng chọn ảnh nhỏ hơn.")
        data = file.read_bytes()
        if len(data) > MAX_FILE_BYTES:
            raise DemoError("Ảnh vượt quá giới hạn 10 MB. Vui lòng chọn ảnh nhỏ hơn.")
        import io
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise DemoError("Chỉ hỗ trợ ảnh JPG, PNG hoặc WebP.")
                if image.width < 1 or image.height < 1 or image.width * image.height > MAX_IMAGE_PIXELS:
                    raise DemoError("Kích thước ảnh không hợp lệ hoặc vượt quá 25 triệu điểm ảnh.")
                image.load()
                rgb = image.convert("RGB")
        return rgb, hashlib.sha256(data).hexdigest()
    except DemoError:
        raise
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise DemoError("Không đọc được ảnh. Vui lòng chọn ảnh JPG, PNG hoặc WebP hợp lệ.") from exc


class InferenceService:
    """One loaded model, shared between inference and explanations."""

    def __init__(self, root: Path = ROOT, checkpoint: Path | None = None, device: str = "auto"):
        self.root = Path(root).resolve()
        self.names, self.temperature, self.threshold = _artifacts(self.root)
        if device not in {"auto", "cpu", "cuda"}:
            raise DemoError("Thiết bị phải là auto, cpu hoặc cuda.")
        if device == "cuda" and not torch.cuda.is_available():
            raise DemoError("CUDA không khả dụng. Hãy chọn CPU hoặc kiểm tra PyTorch.")
        self.device = torch.device("cuda" if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else "cpu")
        checkpoint = Path(checkpoint) if checkpoint else self.root / "outputs/checkpoints/E4/seed_42/best.pt"
        try:
            self.model, self.checkpoint_hash = load_e4_model(self.root, checkpoint, self.device)
        except FileNotFoundError as exc:
            raise DemoError("Chưa tìm thấy checkpoint E4. Xem README để đặt đúng tệp best.pt.") from exc
        except Exception as exc:
            raise DemoError("Không thể nạp checkpoint E4; hãy kiểm tra tệp và SHA-256 đã khóa.") from exc
        try:
            with (self.root / "configs/data.yaml").open(encoding="utf-8") as stream:
                data_config = yaml.safe_load(stream)
            self.transform = build_transforms(data_config, "inference")
            evaluation = data_config["preprocessing"]["evaluation"]
            self.preview_transform = transforms.Compose([
                transforms.Resize(evaluation["resize"]),
                transforms.CenterCrop(evaluation["center_crop"]),
            ])
        except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
            raise DemoError("Không thể đọc cấu hình tiền xử lý dùng chung.") from exc
        self.lock = threading.Lock()

    def predict(self, path: str | Path | None) -> dict:
        image, digest = read_image(path)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        try:
            with self.lock, torch.inference_mode():
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                start = time.perf_counter()
                logits = self.model(tensor)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                latency_ms = (time.perf_counter() - start) * 1000
            if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, 30):
                raise ValueError("E4 output must have shape [1,30]")
            result = calibrated_predictions(logits.detach().cpu().numpy(), self.temperature)
            probabilities = result["probabilities"][0]
            top1 = int(result["predictions"][0])
            confidence = float(result["confidences"][0])
            accepted = bool(accept_mask([confidence], self.threshold)[0])
            indices = np.argsort(-probabilities, kind="stable")[:5]
            return {"class_id": top1, "name": self.names[top1], "confidence": confidence,
                    "accepted": accepted, "latency_ms": latency_ms,
                    "top5": {self.names[int(i)]: float(probabilities[i]) for i in indices},
                    "image_digest": digest}
        except DemoError:
            raise
        except Exception as exc:
            raise DemoError("Không thể dự đoán ảnh này. Vui lòng thử lại.") from exc

    def explain(self, path: str | Path | None, prediction: dict | None) -> Image.Image:
        if not prediction:
            raise DemoError("Hãy dự đoán ảnh trước khi xem Grad-CAM.")
        image, digest = read_image(path)
        if digest != prediction.get("image_digest"):
            raise DemoError("Ảnh đã thay đổi. Hãy dự đoán lại trước khi xem Grad-CAM.")
        try:
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with self.lock:
                heatmap = generate_gradcam(self.model, tensor, target_class=int(prediction["class_id"]))
            # Show the actual centered crop seen by the model, aligned to the heatmap.
            crop = np.asarray(self.preview_transform(image), dtype=np.float32) / 255.0
            from matplotlib import colormaps
            colors = colormaps["jet"](heatmap)[..., :3]
            overlay = np.clip(0.55 * crop + 0.45 * colors, 0, 1)
            return Image.fromarray((overlay * 255).astype(np.uint8), "RGB")
        except Exception as exc:
            raise DemoError("Không thể tạo Grad-CAM cho ảnh này. Vui lòng thử lại.") from exc
