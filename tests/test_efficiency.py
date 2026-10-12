"""Unit tests for src/evaluation/efficiency.py."""
from __future__ import annotations

import io

import torch
import torch.nn as nn
import pytest

from src.evaluation.efficiency import (
    count_parameters,
    model_size_mib,
    measure_latency,
    verify_forward,
    benchmark_model,
)


class _TinyModel(nn.Module):
    """Minimal 30-class model for fast testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 30)

    def forward(self, x):
        return self.fc(x.mean(dim=[-2, -1]))


@pytest.fixture
def tiny():
    return _TinyModel()


def test_count_parameters_total_and_trainable(tiny):
    total, trainable = count_parameters(tiny)
    expected = sum(p.numel() for p in tiny.parameters())
    assert total == expected
    assert trainable == expected


def test_count_parameters_frozen():
    model = _TinyModel()
    for p in model.parameters():
        p.requires_grad = False
    total, trainable = count_parameters(model)
    assert total > 0
    assert trainable == 0


def test_model_size_mib_positive(tiny):
    size = model_size_mib(tiny)
    assert size > 0


def test_model_size_mib_unit():
    """state_dict size < full checkpoint (no optimizer state)."""
    model = _TinyModel()
    size = model_size_mib(model)
    assert isinstance(size, float)
    assert 0 < size < 10


def test_measure_latency_keys(tiny):
    device = torch.device("cpu")
    lat = measure_latency(tiny, device, warmup=2, timed=5)
    assert set(lat.keys()) == {"median_ms", "mean_ms", "std_ms"}
    for v in lat.values():
        assert v >= 0


def test_verify_forward_ok(tiny):
    """Should not raise for correct output."""
    verify_forward(tiny, torch.device("cpu"), num_classes=30)


def test_verify_forward_wrong_classes():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 10)
        def forward(self, x):
            return self.fc(x.mean(dim=[-2, -1]))

    with pytest.raises(AssertionError):
        verify_forward(Wrapper(), torch.device("cpu"), num_classes=30)


def test_benchmark_model_no_checkpoint(tiny):
    result = benchmark_model(tiny, torch.device("cpu"), checkpoint_path=None,
                             num_classes=30, warmup=2, timed=5)
    assert result["checkpoint_loaded"] is False
    assert result["total_params"] > 0
    assert result["trainable_params"] > 0
    assert result["size_mib"] > 0
    assert result["median_ms"] >= 0
    assert result["mean_ms"] >= 0
    assert result["std_ms"] >= 0


def test_benchmark_model_with_checkpoint(tiny, tmp_path):
    ckpt = tmp_path / "best.pt"
    torch.save({"model_state_dict": tiny.state_dict()}, ckpt)

    model2 = _TinyModel()
    result = benchmark_model(model2, torch.device("cpu"),
                             checkpoint_path=str(ckpt),
                             num_classes=30, warmup=2, timed=5)
    assert result["checkpoint_loaded"] is True
    assert result["total_params"] > 0

