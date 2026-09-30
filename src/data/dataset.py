"""Shared dataset interface, backed only by committed split CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

REQUIRED_COLUMNS = {"relative_path", "class_id", "breed_name", "source_split"}


class DogBreedDataset(Dataset):
    """Sample: {'image': Tensor[C,H,W], 'label': int, 'relative_path': str}."""

    def __init__(self, csv_path: str | Path, images_dir: str | Path, transform):
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        if transform is None:
            raise ValueError("An explicit train/evaluation transform is required.")
        self.transform = transform
        frame = pd.read_csv(self.csv_path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Invalid split {self.csv_path}; missing columns: {missing}")
        if frame.empty or frame.relative_path.isna().any() or frame.class_id.isna().any():
            raise ValueError("The split must contain valid, non-empty records.")
        if frame.relative_path.duplicated().any():
            raise ValueError("Duplicate relative_path in split CSV.")
        if not frame["class_id"].between(0, 29).all():
            raise ValueError("class_id must be in [0, 29].")
        self.rows = frame.to_dict("records")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        name = str(row["relative_path"]).replace("\\", "/")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe image path: {name}")
        image_path = self.images_dir / relative
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return {
            "image": tensor,
            "label": int(row["class_id"]),
            "relative_path": name,
        }
