"""Test one generic epoch without committing or reading dataset images."""
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.training.checkpoint import save_checkpoint
from src.training.early_stopping import EarlyStopping
from src.training.trainer import train_one_epoch
from src.training.validator import validate_one_epoch
from src.utils.config import deep_merge


class TinyDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return {"image": torch.ones(3, 4, 4) * (index + 1) / 10,
                "label": index % 2,
                "relative_path": f"test_{index}.jpg"}


def test_generic_training_validation_checkpoint(tmp_path):
    loader = DataLoader(TinyDataset(), batch_size=2)
    model = nn.Sequential(nn.Flatten(), nn.Linear(48, 30))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model[1].weight.detach().clone()
    # 3 batches, accumulation_steps=2: must step on final incomplete group.
    train = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"), accumulation_steps=2)
    val = validate_one_epoch(model, loader, criterion, torch.device("cpu"))
    assert not torch.equal(before, model[1].weight)
    assert 0 <= train["macro_f1"] <= 1 and 0 <= val["top5_accuracy"] <= 1
    target = tmp_path / "best.pt"
    save_checkpoint(target, model, optimizer, 1, val, {"experiment": {"id": "E0"}})
    assert target.is_file()


def test_early_stopping_and_config_merge():
    stopper = EarlyStopping(patience=2)
    assert stopper.step(0.2) == (True, False)
    assert stopper.step(0.2) == (False, False)
    assert stopper.step(0.2) == (False, True)
    assert deep_merge({"training": {"seed": 42, "lr": 1}}, {"training": {"lr": 2}}) == {
        "training": {"seed": 42, "lr": 2}
    }
