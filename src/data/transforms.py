"""Transforms exactly derived from the frozen configs/data.yaml."""
from torchvision import transforms as T


def build_transforms(config: dict, split: str):
    p = config["preprocessing"]
    norm = T.Normalize(mean=p["normalization"]["mean"], std=p["normalization"]["std"])
    if split == "train":
        t = p["train"]
        crop = t["random_resized_crop"]
        jitter = t["color_jitter"]
        return T.Compose([
            T.RandomResizedCrop(
                crop["size"], scale=tuple(crop["scale"]), ratio=tuple(crop["ratio"])
            ),
            T.RandomHorizontalFlip(p=t["horizontal_flip_p"]),
            T.ColorJitter(**jitter),
            T.RandomRotation(degrees=t["rotation_degrees"]),
            T.ToTensor(),
            norm,
        ])
    if split in {"validation", "calibration", "official_test", "inference"}:
        ev = p["evaluation"]
        return T.Compose([
            T.Resize(ev["resize"]),
            T.CenterCrop(ev["center_crop"]),
            T.ToTensor(),
            norm,
        ])
    raise ValueError(f"Unknown transform split: {split}")
