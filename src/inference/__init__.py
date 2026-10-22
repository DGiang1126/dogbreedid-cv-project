"""Task 04 final inference utilities for DogBreedID."""

from .predictor import (
    InferenceContractError,
    assert_task04_split_allowed,
    collect_predictions,
    compute_task1_metrics,
    load_raw_predictions,
    run_final_inference,
    save_raw_predictions,
    sha256_file,
    validate_raw_predictions,
    verify_checkpoint_hash,
)

__all__ = [
    "InferenceContractError",
    "assert_task04_split_allowed",
    "collect_predictions",
    "compute_task1_metrics",
    "load_raw_predictions",
    "run_final_inference",
    "save_raw_predictions",
    "sha256_file",
    "validate_raw_predictions",
    "verify_checkpoint_hash",
]
