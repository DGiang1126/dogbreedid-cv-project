import math

import pytest
import torch
from torch import nn

from src.training.cutmix import cutmix_batch, cutmix_criterion_loss
from src.training.trainer import train_one_epoch


def test_cutmix_preserves_shape_and_lambda_matches_replaced_area():
    torch.manual_seed(123)
    # Image 0 is zero and image 1 is one, so replaced pixels can be counted.
    images = torch.stack([
        torch.zeros(3, 32, 32),
        torch.ones(3, 32, 32),
        torch.full((3, 32, 32), 2.0),
        torch.full((3, 32, 32), 3.0),
    ])
    targets = torch.tensor([0, 1, 2, 3])

    mixed, target_a, target_b, lam, applied = cutmix_batch(
        images, targets, alpha=1.0, probability=1.0
    )

    assert applied is True
    assert mixed.shape == images.shape
    assert target_a.shape == target_b.shape == targets.shape
    assert 0.0 <= lam <= 1.0
    # For sample zero, the pixels that differ from the original zero image
    # form the pasted rectangle (unless the paired sample also happens to be 0;
    # the known seed/4-sample permutation keeps this test deterministic).
    replaced_fraction = (mixed[0] != images[0]).float().mean().item()
    assert lam == pytest.approx(1.0 - replaced_fraction, abs=1e-6)


def test_probability_zero_returns_original_batch_and_labels():
    images = torch.randn(4, 3, 16, 16)
    targets = torch.tensor([0, 1, 2, 3])
    mixed, target_a, target_b, lam, applied = cutmix_batch(
        images, targets, alpha=1.0, probability=0.0
    )
    assert applied is False
    assert mixed is images
    assert target_a is targets and target_b is targets
    assert lam == 1.0
    torch.testing.assert_close(mixed, images)


def test_single_sample_batch_skips_cutmix():
    images = torch.randn(1, 3, 8, 8)
    targets = torch.tensor([2])
    mixed, target_a, target_b, lam, applied = cutmix_batch(
        images, targets, alpha=1.0, probability=1.0
    )
    assert not applied
    assert lam == 1.0
    torch.testing.assert_close(mixed, images)
    torch.testing.assert_close(target_a, targets)
    torch.testing.assert_close(target_b, targets)


@pytest.mark.parametrize(
    "alpha, probability",
    [(0.0, 0.5), (-1.0, 0.5), (1.0, -0.1), (1.0, 1.1)],
)
def test_invalid_cutmix_parameters_raise(alpha, probability):
    with pytest.raises(ValueError):
        cutmix_batch(torch.randn(2, 3, 8, 8), torch.tensor([0, 1]), alpha, probability)


def test_cutmix_loss_is_weighted_and_backward_is_finite():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.1, 0.8, -0.2]], requires_grad=True)
    target_a = torch.tensor([0, 1])
    target_b = torch.tensor([2, 0])
    criterion = nn.CrossEntropyLoss()
    lam = 0.75

    actual = cutmix_criterion_loss(criterion, logits, target_a, target_b, lam, True)
    expected = lam * criterion(logits, target_a) + (1.0 - lam) * criterion(logits, target_b)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual)
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_no_cutmix_loss_matches_ordinary_cross_entropy():
    logits = torch.randn(3, 4, requires_grad=True)
    targets = torch.tensor([0, 1, 2])
    criterion = nn.CrossEntropyLoss()
    actual = cutmix_criterion_loss(criterion, logits, targets, targets, 1.0, False)
    expected = criterion(logits, targets)
    torch.testing.assert_close(actual, expected)


def test_trainer_keeps_clean_metrics_when_cutmix_probability_is_zero():
    torch.manual_seed(42)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 30))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "image": torch.randn(4, 3, 8, 8),
        "label": torch.tensor([0, 1, 2, 1]),
    }
    loader = [batch, batch]
    result = train_one_epoch(
        model=model,
        loader=loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        accumulation_steps=1,
        scaler=None,
        cutmix_config={"enabled": True, "alpha": 1.0, "probability": 0.0},
    )
    assert math.isfinite(result["loss"])
    assert result["metric_samples"] == 8
    assert math.isfinite(result["macro_f1"])
    assert math.isfinite(result["top1_accuracy"])
    assert math.isfinite(result["top5_accuracy"])
