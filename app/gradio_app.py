"""Gradio interface for the locked DogBreedID model."""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import gradio as gr

from app.inference_service import (
    FIT_MODE, OOD_MESSAGE, STANDARD_MODE, STATUS_OOD_REJECT, DemoError, InferenceService, read_image,
)

LOG = logging.getLogger(__name__)
READY = "Tải ảnh lên rồi nhấn **Dự đoán**."

CSS = """
.gradio-container:has(#dogbreed-demo) {
    max-width: 1280px !important;
    margin-inline: auto;
}
.gradio-container:has(#dogbreed-demo) footer { display: none; }

#dogbreed-demo {
    --dogbreed-teal: #0f766e;
    --dogbreed-teal-hover: #115e59;
    --dogbreed-tint: color-mix(in srgb, var(--dogbreed-teal) 9%, var(--background-fill-primary));
    --body-text-color-subdued: color-mix(in srgb, var(--body-text-color) 76%, var(--background-fill-primary));
    --checkbox-background-color-selected: var(--dogbreed-teal);
    --checkbox-border-color-selected: var(--dogbreed-teal);
    --stat-background-fill: var(--dogbreed-teal);
    gap: 16px;
}
#dogbreed-demo .dogbreed-hero.block {
    padding: 16px clamp(16px, 3vw, 28px);
    border-radius: 16px;
    background: #115e59;
}
#dogbreed-demo .dogbreed-hero h1 {
    margin: 0;
    font-size: clamp(1.65rem, 3vw, 2.15rem);
    line-height: 1.15;
    color: #fff;
}
#dogbreed-demo .dogbreed-hero p { margin: 5px 0 0; color: #e6fffb; }
#dogbreed-main {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 16px;
}
#dogbreed-demo .dogbreed-panel {
    gap: 12px;
    padding: clamp(12px, 2vw, 20px);
    border: 1px solid var(--border-color-primary);
    border-radius: 16px;
    background: var(--background-fill-primary);
}
#dogbreed-demo .dogbreed-title h2 { margin: 0; font-size: 1.2rem; }
#dogbreed-demo .dogbreed-note p {
    margin: 0;
    color: var(--body-text-color-subdued);
    font-size: 0.875rem;
    line-height: 1.5;
}
#dogbreed-demo .dogbreed-status.block {
    padding: 10px 12px;
    border-inline-start: 3px solid var(--dogbreed-teal);
    border-radius: 8px;
    background: var(--dogbreed-tint);
}
#dogbreed-demo .dogbreed-status p { margin: 0; }
#dogbreed-demo .form:has(> :is(.dogbreed-mode, .dogbreed-top1, .dogbreed-metric)),
#dogbreed-demo .dogbreed-mode.block,
#dogbreed-demo .dogbreed-top1.block,
#dogbreed-demo .dogbreed-metric.block {
    padding: 0;
    border: 0;
    background: transparent;
    box-shadow: none;
}
#dogbreed-demo .form:has(> .dogbreed-metric) { gap: 12px; }
#dogbreed-demo .dogbreed-top1 label,
#dogbreed-demo .dogbreed-metric label {
    border: 0;
    background: transparent;
    box-shadow: none;
}
#dogbreed-demo .dogbreed-top1 textarea {
    padding: 4px 0;
    border: 0;
    background: transparent;
    font-size: clamp(1.2rem, 2.4vw, 1.65rem);
    font-weight: 700;
}
#dogbreed-demo .dogbreed-metric textarea {
    padding: 4px 0;
    border: 0;
    background: transparent;
    font-weight: 600;
}
#dogbreed-demo .dogbreed-top1 textarea,
#dogbreed-demo .dogbreed-metric textarea {
    color: var(--body-text-color);
    opacity: 1;
}
#dogbreed-demo .dogbreed-top5:has(.output-class) { padding-top: 20px; }
#dogbreed-demo .dogbreed-top5 .output-class { display: none; }
#dogbreed-actions {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    align-items: stretch;
    gap: 12px;
}
#dogbreed-actions button {
    min-height: 44px;
    padding-inline: 8px;
    white-space: normal;
}
#dogbreed-predict {
    background: var(--dogbreed-teal);
    border-color: var(--dogbreed-teal);
    color: #fff;
}
#dogbreed-predict:hover {
    background: var(--dogbreed-teal-hover);
    border-color: var(--dogbreed-teal-hover);
}
#dogbreed-demo .dogbreed-explain-button { min-height: 44px; }
#dogbreed-demo :is(#dogbreed-original, #dogbreed-overlay) img {
    object-fit: contain;
    object-position: center;
}
#dogbreed-demo button:focus-visible {
    outline: 2px solid var(--body-text-color);
    outline-offset: 2px;
}
@media (max-width: 720px) {
    #dogbreed-main { grid-template-columns: minmax(0, 1fr); }
}
"""


class DemoController:
    """Translate service results to UI updates without sharing session state."""

    def __init__(self, service: InferenceService | None, startup_error: str | None = None):
        self.service = service
        self.startup_error = startup_error
        self._session_lock = threading.Lock()
        self._sessions = {}

    def _revision(self, request, *, advance=False, prediction_input=None, selection=None,
                  expected_revision=None):
        key = getattr(request, "session_hash", None)
        if not key:
            return None, None
        while True:
            with self._session_lock:
                now = time.monotonic()
                if len(self._sessions) > 1000:
                    self._sessions = {k: v for k, v in self._sessions.items() if now - v[1] < 3600}
                revision, _, previous_input = self._sessions.get(key, (0, now, None))
                digest = None
                if selection and previous_input and selection[:2] == previous_input[:2]:
                    # Path and mode can defer invalidation until the file is read,
                    # but cannot preserve outputs without the verified digest.
                    if selection[2] is None:
                        return revision, "verify"
                    digest = previous_input[2]
                if digest is None or digest.done():
                    match = None
                    previous_digest = digest.result() if digest is not None else None
                    if previous_digest is not None:
                        match = "same" if previous_digest == selection[2] else "changed"
                    if advance and match != "same":
                        # A slow validation must not invalidate a newer action.
                        if expected_revision is not None and revision != expected_revision:
                            return expected_revision, None
                        revision += 1
                        previous_input = prediction_input
                    self._sessions[key] = (revision, now, previous_input)
                    return revision, match
            # A matching path and mode may still be decoding. Wait for its
            # actual digest without blocking Clear or other sessions.
            digest.result()

    @staticmethod
    def _empty_results(status=READY):
        return (status, "", "", "", {}, None, "", None)

    def _is_current(self, request, revision):
        return revision == self._revision(request)[0]

    def clear(self, request: gr.Request = None):
        """Clear UI and invalidate any prediction still running."""
        self._revision(request, advance=True)
        return (None, None, None, *self._empty_results())

    def select_image(self, path, mode=STANDARD_MODE, request: gr.Request = None):
        candidate = (str(path), mode, None) if path else None
        revision, match = self._revision(request, advance=True, selection=candidate)
        selection = None
        image = model_input = None
        cleared = self._empty_results()
        try:
            if path:
                image, digest = read_image(path)
                selection = (str(path), mode, digest)
        except DemoError as exc:
            cleared = self._empty_results(f"⚠️ {exc}")
        if match == "verify":
            revision, match = self._revision(request, advance=True, selection=selection,
                                             expected_revision=revision)
        if match == "changed":
            cleared = self._empty_results("⚠️ Nội dung ảnh đã thay đổi. Hãy dự đoán lại.")
        try:
            if image is not None and self.service is not None:
                model_input = self.service.preview(image, mode)
        except DemoError as exc:
            cleared = self._empty_results(f"⚠️ {exc}")
        current, latest_match = self._revision(request, selection=selection)
        preserve = match == "same"
        if current != revision:
            if latest_match != "same":
                return (gr.skip(),) * (2 + len(cleared))
            preserve = True
        return (image, model_input, *((gr.skip(),) * len(cleared) if preserve else cleared))

    def predict(self, path, mode=STANDARD_MODE, request: gr.Request = None):
        if self.service is None:
            self._revision(request, advance=True)
            return self._empty_results(f"⚠️ {self.startup_error or 'Không thể khởi tạo mô hình.'}")
        digest_ready = Future()
        prediction_input = (str(path), mode, digest_ready) if path else None
        revision, _ = self._revision(request, advance=True, prediction_input=prediction_input)
        try:
            digest = None
            try:
                image, digest = read_image(path)
            finally:
                digest_ready.set_result(digest)
            prediction = self.service.predict_image(image, digest, mode)
            prediction["revision"] = revision
            latency = f"{prediction['latency_ms']:.1f} ms"
            if prediction.get("status") == STATUS_OOD_REJECT:
                # Out of scope: no Top-1, confidence, Top-5 or Grad-CAM is shown.
                updates = (f"🚫 **OOD REJECT** · {OOD_MESSAGE}", "", "", latency, {}, None, "", prediction)
            else:
                if prediction["accepted"]:
                    status = "✅ **ACCEPT** · Mô hình đủ tự tin để đưa ra dự đoán."
                    breed = prediction["name"]
                else:
                    status = "⚠️ **E7 REJECT** · Mô hình chưa đủ tự tin để đưa ra kết luận. Top-5 chỉ để tham khảo."
                    breed = "Chưa đủ tự tin để kết luận"
                updates = (status, breed, f"{prediction['confidence'] * 100:.2f}%",
                           latency, prediction["top5"], None, "", prediction)
        except DemoError as exc:
            updates = self._empty_results(f"⚠️ {exc}")
        except Exception:
            LOG.exception("Unexpected prediction error")
            updates = self._empty_results("⚠️ Có lỗi khi dự đoán. Vui lòng thử lại.")
        return updates if self._is_current(request, revision) else (gr.skip(),) * len(updates)

    def explain(self, path, prediction, mode=STANDARD_MODE, request: gr.Request = None):
        revision, _ = self._revision(request)
        if self.service is None:
            return None, f"⚠️ {self.startup_error or 'Không thể khởi tạo mô hình.'}"
        try:
            if revision is not None and prediction and prediction.get("revision") != revision:
                raise DemoError("Hãy dự đoán lại ảnh trước khi xem Grad-CAM.")
            overlay = self.service.explain(path, prediction, mode)
            updates = (overlay, f"Vùng ảnh ảnh hưởng đến dự đoán **{prediction['name']}** "
                       "trên ảnh 224 × 224 mà mô hình quan sát. "
                       "Đây là hình hỗ trợ quan sát, không khẳng định nguyên nhân dự đoán.")
        except DemoError as exc:
            updates = (None, f"⚠️ {exc}")
        except Exception:
            LOG.exception("Unexpected Grad-CAM error")
            updates = (None, "⚠️ Có lỗi khi tạo Grad-CAM. Vui lòng thử lại.")
        return updates if self._is_current(request, revision) else (gr.skip(), gr.skip())


def build_app(service: InferenceService | None = None, startup_error: str | None = None) -> gr.Blocks:
    controller = DemoController(service, startup_error)
    with gr.Blocks(title="DogBreedID", analytics_enabled=False) as demo:
        state = gr.State(None)
        with gr.Column(elem_id="dogbreed-demo", min_width=0):
            gr.Markdown("# DogBreedID\n\nNhận diện 30 giống chó qua ảnh.", elem_classes="dogbreed-hero")
            with gr.Row(elem_id="dogbreed-main"):
                with gr.Column(elem_classes="dogbreed-panel", min_width=0):
                    gr.Markdown("## 01 · Ảnh đầu vào", elem_classes="dogbreed-title")
                    # File leaves decoding and error handling to our validator.
                    picture = gr.File(label="Chọn ảnh JPG, PNG hoặc WebP", type="filepath", file_count="single",
                                      height=124)
                    gr.Markdown("Chọn ảnh có một chú chó là chủ thể chính. Tối đa 10 MB và 25 triệu điểm ảnh.",
                                elem_classes="dogbreed-note")
                    mode = gr.Radio(
                        choices=[("Chuẩn E4 — CenterCrop", STANDARD_MODE), ("Giữ trọn ảnh — Padding", FIT_MODE)],
                        value=STANDARD_MODE, label="Chế độ xử lý ảnh",
                        elem_classes="dogbreed-mode",
                    )
                    gr.Markdown("Padding đang thử nghiệm, chưa được xác nhận qua đánh giá chính thức.",
                                elem_classes="dogbreed-note")
                    with gr.Row(elem_id="dogbreed-actions"):
                        predict_button = gr.Button("Dự đoán", variant="primary", min_width=0, elem_id="dogbreed-predict")
                        clear_button = gr.Button("Xóa ảnh và kết quả", variant="secondary", min_width=0)
                    preview = gr.Image(label="Ảnh của bạn", type="pil", interactive=False,
                                       height="clamp(230px, 70vw, 300px)", elem_id="dogbreed-original",
                                       buttons=["fullscreen"], alt_text="Ảnh gốc đã tải lên")
                    model_preview = gr.Image(label="Ảnh 224 × 224 thực sự đưa vào mô hình", type="pil",
                                             format="png", interactive=False, visible=False)
                with gr.Column(elem_classes="dogbreed-panel", min_width=0):
                    gr.Markdown("## 02 · Kết quả phân loại", elem_classes="dogbreed-title")
                    status = gr.Markdown(READY, elem_classes="dogbreed-status")
                    breed = gr.Textbox(label="Giống dự đoán", interactive=False,
                                       elem_classes="dogbreed-top1", placeholder="—")
                    with gr.Row():
                        confidence = gr.Textbox(label="Độ tin cậy đã hiệu chỉnh", interactive=False,
                                                min_width=0,
                                                elem_classes="dogbreed-metric", placeholder="—")
                        latency = gr.Textbox(label="Thời gian chạy mô hình", interactive=False,
                                             min_width=0,
                                             elem_classes="dogbreed-metric", placeholder="—")
                    top5 = gr.Label(label="5 giống có khả năng cao nhất", num_top_classes=5,
                                    elem_classes="dogbreed-top5")
                    gr.Markdown("Thời gian chỉ tính lượt chạy mô hình. Mỗi ảnh được tự động kiểm tra "
                                "ngoài phạm vi (Energy OOD): OOD REJECT nghĩa là ảnh có thể không thuộc "
                                "30 giống được hỗ trợ. E7 REJECT nghĩa là ảnh qua kiểm tra OOD "
                                "nhưng dự đoán chưa đủ tin cậy.",
                                elem_classes="dogbreed-note")

            with gr.Column(elem_id="dogbreed-gradcam", elem_classes="dogbreed-panel", min_width=0):
                gr.Markdown("## 03 · Giải thích dự đoán bằng Grad-CAM", elem_classes="dogbreed-title")
                gr.Markdown("Grad-CAM làm nổi bật vùng ảnh ảnh hưởng đến dự đoán. "
                            "Hình minh họa hỗ trợ quan sát, không xác định nguyên nhân.",
                            elem_classes="dogbreed-note")
                explain_button = gr.Button("Xem Grad-CAM", elem_classes="dogbreed-explain-button")
                overlay = gr.Image(label="Grad-CAM trên vùng ảnh mô hình quan sát", type="pil", interactive=False,
                                   height="clamp(240px, 78vw, 520px)", elem_id="dogbreed-overlay",
                                   buttons=["fullscreen"], alt_text="Bản đồ Grad-CAM của dự đoán")
                explain_text = gr.Markdown("")

        outputs = [status, breed, confidence, latency, top5, overlay, explain_text, state]
        predict_event = predict_button.click(fn=controller.predict, inputs=[picture, mode],
                                             outputs=outputs, api_name="predict")
        explain_event = explain_button.click(fn=controller.explain, inputs=[picture, state, mode],
                                             outputs=[overlay, explain_text], api_visibility="private")
        cancel_inference = [predict_event, explain_event]
        gr.on([picture.upload, picture.clear, mode.input], fn=controller.select_image, inputs=[picture, mode],
              outputs=[preview, model_preview, *outputs], queue=False, api_visibility="private",
              api_name="select_image", trigger_mode="multiple", cancels=cancel_inference)
        clear_button.click(fn=controller.clear, outputs=[picture, preview, model_preview, *outputs],
                           queue=False, api_visibility="private", show_progress="hidden", cancels=cancel_inference)
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
