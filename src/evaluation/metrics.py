"""Classification metrics: Macro-F1, Top-1, Top-5 (fraction in [0,1])."""
import numpy as np
import torch
from sklearn.metrics import f1_score


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 30) -> dict:
    if logits.ndim != 2 or logits.shape[1] != num_classes:
        raise ValueError(f"Expected logits of shape [N,{num_classes}], got {tuple(logits.shape)}")
    targets = targets.long().view(-1)
    if len(targets) != len(logits) or len(targets) == 0:
        raise ValueError("Logit/target length mismatch or empty inputs.")
    if targets.min().item() < 0 or targets.max().item() >= num_classes:
        raise ValueError("Target is outside the fixed class mapping.")
    predicted = logits.argmax(dim=1)
    top_k = logits.topk(k=min(5, num_classes), dim=1).indices
    return {
        "macro_f1": float(f1_score(
            targets.cpu().numpy(), predicted.cpu().numpy(),
            average="macro", labels=np.arange(num_classes), zero_division=0,
        )),
        "top1_accuracy": float((predicted == targets).float().mean().item()),
        "top5_accuracy": float((top_k == targets[:, None]).any(dim=1).float().mean().item()),
    }
