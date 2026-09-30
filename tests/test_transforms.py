"""Frozen transform tests using a synthetic image."""
import torch
from PIL import Image

from src.data.transforms import build_transforms


CONFIG = {
    "preprocessing": {
        "train": {
            "random_resized_crop": {"size": 224, "scale": [0.85, 1.0], "ratio": [0.9, 1.1]},
            "horizontal_flip_p": 0.5,
            "color_jitter": {"brightness": 0.15, "contrast": 0.15, "saturation": 0.15, "hue": 0.05},
            "rotation_degrees": 10,
        },
        "evaluation": {"resize": 256, "center_crop": 224},
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
}


def test_shapes_and_eval_determinism():
    image = Image.new("RGB", (320, 260), "yellow")
    assert build_transforms(CONFIG, "train")(image).shape == (3, 224, 224)
    fn = build_transforms(CONFIG, "validation")
    first, second = fn(image), fn(image)
    assert first.shape == (3, 224, 224)
    assert torch.equal(first, second)
    assert torch.equal(first, build_transforms(CONFIG, "calibration")(image))
