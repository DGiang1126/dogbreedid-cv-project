"""Unit tests for Task 07 E10/E11 configs, CutMix scope, runner guards, and checkpointing."""
from __future__ import annotations

import yaml
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.train_research_models import run_training
from src.research.extended_config import load_research_config, research_output_files
from src.research.extended_models import build_research_model
from src.research.extended_optimizer import build_research_optimizer
from src.training.checkpoint import save_checkpoint
from src.training.trainer import train_one_epoch
from src.training.validator import validate_one_epoch
from src.utils.paths import split_file


class _SyntheticDogDataset(Dataset):
    def __init__(self, n: int = 8):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 224, 224),
            "label": idx % 30,
            "relative_path": f"synthetic_{idx}.jpg",
        }


def test_all_four_configs_valid_and_ab_consistent():
    c10a = load_research_config("configs/experiments/e10a_efficientnet_no_cutmix.yaml")
    c10b = load_research_config("configs/experiments/e10b_efficientnet_cutmix.yaml")
    c11a = load_research_config("configs/experiments/e11a_vit_no_cutmix.yaml")
    c11b = load_research_config("configs/experiments/e11b_vit_cutmix.yaml")

    assert c10a["augmentation"]["cutmix"]["enabled"] is False
    assert c10b["augmentation"]["cutmix"]["enabled"] is True
    assert c10a["preprocessing"] == c10b["preprocessing"]
    assert c10a["model"]["weights"] == c10b["model"]["weights"]

    assert c11a["augmentation"]["cutmix"]["enabled"] is False
    assert c11b["augmentation"]["cutmix"]["enabled"] is True
    assert c11a["preprocessing"] == c11b["preprocessing"]
    assert c11a["model"]["weights"] == c11b["model"]["weights"]


def test_invalid_configs_rejected(tmp_path):
    bad_freeze = tmp_path / "bad_freeze.yaml"
    bad_freeze.write_text(
        yaml.safe_dump({
            "experiment": {"id": "E10-A", "name": "bad", "seed": 42},
            "model": {"type": "efficientnet_b0", "source": "torchvision", "backbone_frozen": False, "num_classes": 30},
            "training": {"classifier_lr": 0.001},
            "augmentation": {"cutmix": {"enabled": False, "alpha": 1.0, "probability": 0.5}},
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="backbone_frozen"):
        load_research_config(bad_freeze)

    bad_cutmix = tmp_path / "bad_cutmix.yaml"
    bad_cutmix.write_text(
        yaml.safe_dump({
            "experiment": {"id": "E10-B", "name": "bad", "seed": 42},
            "model": {"type": "efficientnet_b0", "source": "torchvision", "backbone_frozen": True, "num_classes": 30},
            "training": {"classifier_lr": 0.001},
            "augmentation": {"cutmix": {"enabled": False, "alpha": 1.0, "probability": 0.5}},
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cutmix"):
        load_research_config(bad_cutmix)


def test_synthetic_smoke_cutmix_checkpoint_reload(tmp_path):
    config = load_research_config("configs/experiments/e10b_efficientnet_cutmix.yaml")
    model = build_research_model(config, force_no_pretrained=True)
    optimizer = build_research_optimizer(model, config)
    criterion = torch.nn.CrossEntropyLoss()

    loader = DataLoader(_SyntheticDogDataset(8), batch_size=4, shuffle=False)
    train_res = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        torch.device("cpu"),
        accumulation_steps=1,
        cutmix_config=config["augmentation"]["cutmix"],
    )
    val_res = validate_one_epoch(model, loader, criterion, torch.device("cpu"))
    assert "metric_samples" in train_res
    assert 0.0 <= val_res["macro_f1"] <= 1.0

    ckpt_path = tmp_path / "best.pt"
    save_checkpoint(ckpt_path, model, optimizer, 1, val_res, config)
    reloaded = build_research_model(config, force_no_pretrained=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(ckpt["model_state_dict"], strict=True)
    reloaded_val = validate_one_epoch(reloaded, loader, criterion, torch.device("cpu"))
    assert reloaded_val["macro_f1"] == pytest.approx(val_res["macro_f1"])


def test_no_overwrite_and_official_test_blocked(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Development code cannot request split"):
        split_file("official_test")

    paths = research_output_files("E10-A", 42, output_root=tmp_path)
    paths["history"].parent.mkdir(parents=True, exist_ok=True)
    paths["history"].write_text("epoch\n1\n", encoding="utf-8")

    # Even if dataset exists, FileExistsError must fire before training starts.
    monkeypatch.setattr("scripts.train_research_models.dataset_root", lambda d=None: tmp_path)
    monkeypatch.setattr("scripts.train_research_models.DogBreedDataset", lambda *a, **kw: _SyntheticDogDataset(4))
    with pytest.raises(FileExistsError, match="Do not overwrite"):
        run_training(
            "configs/experiments/e10a_efficientnet_no_cutmix.yaml",
            device_arg="cpu",
            output_root=tmp_path,
        )
