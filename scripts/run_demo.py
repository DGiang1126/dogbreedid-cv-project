"""Launch DogBreedID locally; never expose a public Gradio share link."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.gradio_app import CSS, create_app


def main():
    parser = argparse.ArgumentParser(description="Chạy ứng dụng DogBreedID trên máy cá nhân.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Đường dẫn đến E4 best.pt đã được phê duyệt.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app = create_app(checkpoint=args.checkpoint, device=args.device)
    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port, share=False,
        max_file_size="10mb", css=CSS, inbrowser=False,
    )


if __name__ == "__main__":
    main()
