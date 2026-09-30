"""Macro-F1 across all 30 frozen classes, Top-1 and Top-5."""
import pytest
import torch

from src.evaluation.metrics import classification_metrics


def test_perfect_30_class_metrics():
    logits = torch.eye(30) * 10
    labels = torch.arange(30)
    result = classification_metrics(logits, labels)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["top1_accuracy"] == pytest.approx(1.0)
    assert result["top5_accuracy"] == pytest.approx(1.0)


def test_invalid_output_shape():
    with pytest.raises(ValueError, match="logits"):
        classification_metrics(torch.zeros(2, 29), torch.tensor([0, 1]))
