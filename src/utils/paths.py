"""Project-relative data/output locations. Do not hard-code local paths."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def dataset_root(override: str | Path | None = None) -> Path:
    if override:
        path = Path(override)
    elif os.environ.get("DOGBREEDID_DATA_ROOT"):
        path = Path(os.environ["DOGBREEDID_DATA_ROOT"])
    else:
        path = PROJECT_ROOT / "data/raw/stanford_dogs"
    path = path.expanduser().resolve()
    if not (path / "Images").is_dir():
        raise FileNotFoundError(f"Expected the Images directory inside: {path}")
    return path


def split_file(name: str) -> Path:
    # Official Test is deliberately not exposed by the development runner.
    if name not in {"train", "validation", "calibration"}:
        raise ValueError(f"Development code cannot request split: {name}")
    result = PROJECT_ROOT / "data/splits" / f"{name}.csv"
    if not result.is_file():
        raise FileNotFoundError(f"Missing committed split: {result}")
    return result


def output_files(experiment_id: str, seed: int) -> dict[str, Path]:
    part = Path(experiment_id) / f"seed_{seed}"
    root = PROJECT_ROOT / "outputs"
    return {
        "best_checkpoint": root / "checkpoints" / part / "best.pt",
        "last_checkpoint": root / "checkpoints" / part / "last.pt",
        "history": root / "logs" / part / "history.csv",
        "metadata": root / "logs" / part / "run_metadata.json",
        "validation_metrics": root / "metrics" / part / "validation.json",
    }
