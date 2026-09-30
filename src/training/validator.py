"""One deterministic validation loop shared by E0–E3."""
import torch

from src.evaluation.metrics import classification_metrics


@torch.inference_mode()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    seen = 0
    all_logits, all_targets = [], []
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["label"].to(device).long()
        logits = model(image)
        loss = criterion(logits, target)
        seen += len(target)
        total_loss += loss.item() * len(target)
        all_logits.append(logits.detach().float().cpu())
        all_targets.append(target.detach().cpu())
    if seen == 0:
        raise ValueError("Empty validation loader.")
    result = classification_metrics(torch.cat(all_logits), torch.cat(all_targets))
    return {"loss": total_loss / seen, **result}
