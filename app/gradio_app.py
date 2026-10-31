"""Gradio interface for the locked DogBreedID model."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import gradio as gr

from app.inference_service import DemoError, InferenceService, read_image

LOG = logging.getLogger(__name__)

CSS = """
.gradio-container {max-width: 1120px !important; margin: auto;}
.hero {padding: 28px 30px; border-radius: 22px; background: linear-gradient(120deg,#173a38,#25645a); color: white; margin-bottom: 10px;}
.hero h1 {color: white !important; margin: 0 0 8px !important; font-size: 2rem !important;}
.hero p {color: #e2f1eb !important; margin: 0; line-height: 1.65;}
.explain-note {color: #5c6d66; font-size: .9rem;}
footer {visibility: hidden;}
"""


class DemoController:
    """Translate service results to UI updates without sharing session state."""

    def __init__(self, service: InferenceService | None, startup_error: str | None = None):
        self.service = service
        self.startup_error = startup_error
        self._session_lock = threading.Lock()
        self._sessions = {}

    def _revision(self, request, advance=False, prediction_path=None):
        key = getattr(request, "session_hash", None)
        if not key:
            return None
        with self._session_lock:
            now = time.monotonic()
            if len(self._sessions) > 1000:
                self._sessions = {k: v for k, v in self._sessions.items() if now - v[1] < 3600}
            previous = self._sessions.get(key, (0, now, None))
            revision = previous[0] + int(advance)
            path = str(prediction_path) if prediction_path is not None else previous[2]
            self._sessions[key] = (revision, now, path)
            return revision

    def reset(self, path=None, request: gr.Request = None):
        key = getattr(request, "session_hash", None)
        if key:
            with self._session_lock:
                now = time.monotonic()
                previous = self._sessions.get(key, (0, now, None))
                # A delayed upload event for the image already being predicted
                # must not erase or suppress that image's prediction.
                if path and previous[2] == str(path):
                    return tuple(gr.skip() for _ in range(8))
                self._sessions[key] = (previous[0] + 1, now, None)
        return ("Tải ảnh lên rồi nhấn **Dự đoán**.", "", "", "", {}, None, "", None)

    def select_image(self, path, request: gr.Request = None):
        cleared = self.reset(path, request)
        revision = self._revision(request)
        if path is None:
            return (None, *cleared)
        try:
            image, _ = read_image(path)
            if revision is not None and revision != self._revision(request):
                return tuple(gr.skip() for _ in range(9))
            return (image, *cleared)
        except DemoError as exc:
            if revision is not None and revision != self._revision(request):
                return tuple(gr.skip() for _ in range(9))
            return (None, f"⚠️ {exc}", "", "", "", {}, None, "", None)

    def predict(self, path, request: gr.Request = None):
        revision = self._revision(request, advance=True, prediction_path=path)
        if self.service is None:
            return (f"⚠️ {self.startup_error or 'Không thể khởi tạo mô hình.'}", "", "", "", {}, None, "", None)
        try:
            prediction = self.service.predict(path)
            if revision is not None and revision != self._revision(request):
                return tuple(gr.skip() for _ in range(8))
            if prediction["accepted"]:
                status = "✅ **ACCEPT** · Mô hình đủ tự tin để đưa ra dự đoán."
                breed = prediction["name"]
            else:
                status = "⚠️ **REJECT** · Mô hình chưa đủ tự tin để đưa ra kết luận. Top-5 chỉ để tham khảo."
                breed = "Chưa đủ tự tin để kết luận"
            return (status, breed, f"{prediction['confidence'] * 100:.2f}%",
                    f"{prediction['latency_ms']:.1f} ms", prediction["top5"], None, "", prediction)
        except DemoError as exc:
            if revision is not None and revision != self._revision(request):
                return tuple(gr.skip() for _ in range(8))
            return (f"⚠️ {exc}", "", "", "", {}, None, "", None)
        except Exception:
            LOG.exception("Unexpected prediction error")
            if revision is not None and revision != self._revision(request):
                return tuple(gr.skip() for _ in range(8))
            return ("⚠️ Có lỗi khi dự đoán. Vui lòng thử lại.", "", "", "", {}, None, "", None)

    def explain(self, path, prediction, request: gr.Request = None):
        revision = self._revision(request)
        if self.service is None:
            return None, f"⚠️ {self.startup_error or 'Không thể khởi tạo mô hình.'}"
        try:
            overlay = self.service.explain(path, prediction)
            if revision is not None and revision != self._revision(request):
                return gr.skip(), gr.skip()
            return overlay, (f"Vùng ảnh ảnh hưởng đến dự đoán **{prediction['name']}** "
                             "trên vùng cắt 224 × 224 mà mô hình quan sát. "
                             "Đây là hình hỗ trợ quan sát, không khẳng định nguyên nhân dự đoán.")
        except DemoError as exc:
            if revision is not None and revision != self._revision(request):
                return gr.skip(), gr.skip()
            return None, f"⚠️ {exc}"
        except Exception:
            LOG.exception("Unexpected Grad-CAM error")
            if revision is not None and revision != self._revision(request):
                return gr.skip(), gr.skip()
            return None, "⚠️ Có lỗi khi tạo Grad-CAM. Vui lòng thử lại."


def build_app(service: InferenceService | None = None, startup_error: str | None = None) -> gr.Blocks:
    controller = DemoController(service, startup_error)
    with gr.Blocks(title="DogBreedID", analytics_enabled=False) as demo:
        gr.HTML('<div class="hero"><h1>DogBreedID</h1><p>Khám phá 30 giống chó qua ảnh · Mô hình học sâu được hiệu chỉnh độ tin cậy</p></div>')
        gr.Markdown("Tải lên một ảnh có một chú chó là chủ thể chính. Hệ thống chỉ phân loại trong **30 giống đã học**.")
        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("### 1 · Chọn ảnh")
                # File accepts the upload without decoding. Our validator can
                # report broken images before any image-preview processing.
                picture = gr.File(label="Chọn ảnh JPG, PNG hoặc WebP", type="filepath", file_count="single")
                preview = gr.Image(label="Ảnh của bạn", type="pil", interactive=False, height=300)
                gr.Markdown("JPG, PNG hoặc WebP · Tối đa 10 MB · Ảnh rõ, một chú chó ở vị trí trung tâm sẽ dễ quan sát hơn.")
                with gr.Row():
                    predict_button = gr.Button("Dự đoán", variant="primary")
                    clear_button = gr.ClearButton(value="Xóa ảnh và kết quả")
            with gr.Column(scale=5):
                gr.Markdown("### 2 · Kết quả")
                status = gr.Markdown("Tải ảnh lên rồi nhấn **Dự đoán**.")
                breed = gr.Textbox(label="Giống dự đoán", interactive=False)
                with gr.Row():
                    confidence = gr.Textbox(label="Độ tin cậy đã hiệu chỉnh", interactive=False)
                    latency = gr.Textbox(label="Thời gian chạy mô hình", info="Chỉ đo lượt chạy mô hình; không bao gồm tải và xử lý ảnh.", interactive=False)
                top5 = gr.Label(label="5 giống có khả năng cao nhất", num_top_classes=5)
        gr.Markdown("### 3 · Quan sát vùng ảnh ảnh hưởng đến dự đoán")
        gr.Markdown("Grad-CAM giúp quan sát mô hình tập trung vào đâu khi đưa ra dự đoán; hình này không chứng minh nguyên nhân dự đoán.", elem_classes="explain-note")
        explain_button = gr.Button("Xem Grad-CAM")
        overlay = gr.Image(label="Grad-CAM trên vùng ảnh mô hình quan sát", type="pil", interactive=False, height=330)
        explain_text = gr.Markdown("")
        gr.Markdown("**Lưu ý:** Kết quả chỉ để tham khảo. REJECT nghĩa là mô hình chưa đủ tự tin, không xác định được giống ngoài 30 lớp hoặc ảnh có chó hay không.")

        outputs = [status, breed, confidence, latency, top5, overlay, explain_text, state]
        picture.change(fn=controller.select_image, inputs=[picture], outputs=[preview, *outputs], queue=False, api_name=False)
        predict_button.click(fn=controller.predict, inputs=[picture], outputs=outputs, api_name="predict")
        explain_button.click(fn=controller.explain, inputs=[picture, state], outputs=[overlay, explain_text], api_name=False)
        clear_button.add([picture, preview, breed, confidence, latency, top5, overlay, explain_text, state])
    return demo


def create_app(root: Path | None = None, checkpoint: Path | None = None, device: str = "auto") -> gr.Blocks:
    try:
        kwargs = {"checkpoint": checkpoint, "device": device}
        if root is not None:
            kwargs["root"] = root
        service = InferenceService(**kwargs)
        return build_app(service)
    except DemoError as exc:
        LOG.error("Could not start model: %s", exc)
        return build_app(startup_error=str(exc))
