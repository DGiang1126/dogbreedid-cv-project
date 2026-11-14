"""Task 07 OOD-200: Energy-based OOD detection for the closed-set E4 classifier."""

from .energy_detector import (
    DEFAULT_ENERGY_TEMPERATURE,
    OOD_REJECT,
    PASS_OOD_CHECK,
    EnergyOODDetector,
    energy_score,
    is_ood,
    ood_decisions,
    select_energy_threshold,
    validate_logits,
)
from .ood_metrics import id_selective_metrics, ood_auroc, ood_rejection_metrics, score_summary

__all__ = [
    "DEFAULT_ENERGY_TEMPERATURE", "OOD_REJECT", "PASS_OOD_CHECK", "EnergyOODDetector",
    "energy_score", "is_ood", "ood_decisions", "select_energy_threshold", "validate_logits",
    "id_selective_metrics", "ood_auroc", "ood_rejection_metrics", "score_summary",
]
