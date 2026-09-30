"""Synthetic data contract tests; do not touch Official Test."""
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import transforms as T

from src.data.dataset import DogBreedDataset


def test_dataset_reads_committed_csv_shape(tmp_path):
    root = tmp_path / "Images"
    folder = root / "n000001-Eskimo_dog"
    folder.mkdir(parents=True)
    Image.new("RGB", (32, 20), "red").save(folder / "dog.jpg")
    csv_file = tmp_path / "train.csv"
    pd.DataFrame([{
        "relative_path": "n000001-Eskimo_dog/dog.jpg", "class_id": 0,
        "breed_name": "Eskimo Dog", "source_split": "official_train",
    }]).to_csv(csv_file, index=False)
    ds = DogBreedDataset(csv_file, root, T.ToTensor())
    assert len(ds) == 1
    sample = ds[0]
    assert sample["image"].shape == (3, 20, 32)
    assert sample["label"] == 0
    assert sample["relative_path"] == "n000001-Eskimo_dog/dog.jpg"


def test_dataset_rejects_duplicate_paths(tmp_path):
    csv_file = tmp_path / "train.csv"
    row = {"relative_path": "a.jpg", "class_id": 0,
           "breed_name": "Eskimo Dog", "source_split": "official_train"}
    pd.DataFrame([row, row]).to_csv(csv_file, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        DogBreedDataset(csv_file, tmp_path, T.ToTensor())
