"""Entry point for dataset preparation. Implementation should call logic from src/."""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "stanford_dogs"
DEFAULT_CLASS_MAPPING = PROJECT_ROOT / "data" / "metadata" / "class_mapping.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

SPLIT_SEED = 42


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def normalize_name(text: str) -> str:
    """
    Normalize breed names so values such as:

        "Standard Poodle"
        "standard_poodle"
        "standard-poodle"

    can be matched consistently.
    """
    text = unicodedata.normalize("NFKD", text).casefold()
    return "".join(char for char in text if char.isalnum())


def mat_value_to_string(value) -> str:
    """
    Convert a MATLAB cell/string loaded by scipy.io.loadmat
    into a normal Python string.
    """
    while isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(
                f"Expected a scalar MATLAB cell, got shape={value.shape}"
            )
        value = value.reshape(-1)[0]

    if isinstance(value, bytes):
        return value.decode("utf-8")

    return str(value)


def load_mat_file_list(mat_path: Path) -> list[str]:
    """
    Load the official Stanford Dogs file_list variable
    from train_list.mat or test_list.mat.
    """
    data = loadmat(mat_path)

    if "file_list" not in data:
        available = [
            key for key in data.keys()
            if not key.startswith("__")
        ]
        raise KeyError(
            f"{mat_path} does not contain 'file_list'. "
            f"Available keys: {available}"
        )

    raw_file_list = np.ravel(data["file_list"])

    file_list = [
        mat_value_to_string(item).replace("\\", "/")
        for item in raw_file_list
    ]

    return file_list


def load_class_mapping(mapping_path: Path) -> dict[int, str]:
    """
    Read class_mapping.json and validate Class IDs.
    """
    with mapping_path.open("r", encoding="utf-8") as file:
        raw_mapping = json.load(file)

    mapping = {
        int(class_id): breed_name
        for class_id, breed_name in raw_mapping.items()
    }

    expected_ids = list(range(30))

    if sorted(mapping.keys()) != expected_ids:
        raise ValueError(
            "class_mapping.json must contain exactly Class IDs 0..29."
        )

    if len(set(mapping.values())) != 30:
        raise ValueError(
            "class_mapping.json contains duplicate breed names."
        )

    return mapping


# ---------------------------------------------------------------------
# Stanford folder mapping
# ---------------------------------------------------------------------

def build_folder_mapping(
    images_dir: Path,
    class_mapping: dict[int, str],
) -> dict[str, dict]:
    """
    Match our project breed names with actual Stanford Dogs folders.

    Example:
        "Standard Poodle"
        ->
        n02113799-standard_poodle
    """
    actual_folders = [
        folder
        for folder in images_dir.iterdir()
        if folder.is_dir()
    ]

    folder_by_normalized_name = {}

    for folder in actual_folders:
        if "-" not in folder.name:
            continue

        _, breed_part = folder.name.split("-", 1)
        normalized = normalize_name(breed_part)

        if normalized in folder_by_normalized_name:
            raise ValueError(
                f"Duplicate normalized breed folder: {normalized}"
            )

        folder_by_normalized_name[normalized] = folder.name

    selected = {}

    missing_breeds = []

    for class_id, breed_name in class_mapping.items():
        normalized = normalize_name(breed_name)

        folder_name = folder_by_normalized_name.get(normalized)

        if folder_name is None:
            missing_breeds.append(breed_name)
            continue

        selected[folder_name] = {
            "class_id": class_id,
            "breed_name": breed_name,
        }

    if missing_breeds:
        raise ValueError(
            "Could not match the following breeds to Stanford Dogs folders:\n"
            + "\n".join(f"  - {breed}" for breed in missing_breeds)
        )

    if len(selected) != 30:
        raise ValueError(
            f"Expected 30 selected folders, found {len(selected)}."
        )

    return selected


# ---------------------------------------------------------------------
# Build metadata
# ---------------------------------------------------------------------

def build_dataframe(
    file_list: list[str],
    selected_folders: dict[str, dict],
    source_split: str,
) -> pd.DataFrame:
    """
    Filter the official Stanford file list to only our fixed 30 breeds.
    """
    records = []

    for relative_path in file_list:
        path = Path(relative_path)

        if not path.parts:
            continue

        folder_name = path.parts[0]

        if folder_name not in selected_folders:
            continue

        class_info = selected_folders[folder_name]

        records.append(
            {
                "relative_path": path.as_posix(),
                "class_id": class_info["class_id"],
                "breed_name": class_info["breed_name"],
                "source_split": source_split,
            }
        )

    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        raise ValueError(
            f"No selected images found for {source_split}."
        )

    return dataframe


# ---------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------

def audit_images(
    dataframe: pd.DataFrame,
    images_dir: Path,
) -> list[dict]:
    """
    Check that images exist and can be opened.

    Important:
    corrupt/missing files are NOT silently removed.
    """
    issues = []

    for row in dataframe.itertuples(index=False):
        image_path = images_dir / row.relative_path

        if not image_path.exists():
            issues.append(
                {
                    "relative_path": row.relative_path,
                    "issue": "missing_file",
                }
            )
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()

        except Exception as error:
            issues.append(
                {
                    "relative_path": row.relative_path,
                    "issue": f"corrupt_or_unreadable: {error}",
                }
            )

    return issues


def check_no_overlap(named_dataframes: dict[str, pd.DataFrame]) -> None:
    """
    Verify that no image appears in more than one split.
    """
    names = list(named_dataframes.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a = names[i]
            name_b = names[j]

            paths_a = set(
                named_dataframes[name_a]["relative_path"]
            )
            paths_b = set(
                named_dataframes[name_b]["relative_path"]
            )

            overlap = paths_a & paths_b

            if overlap:
                examples = sorted(overlap)[:10]

                raise ValueError(
                    f"DATA LEAKAGE: {name_a} and {name_b} "
                    f"share {len(overlap)} images.\n"
                    f"Examples: {examples}"
                )


# ---------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------

def create_development_splits(
    official_train: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split Official Train:

        80% Train
        10% Validation
        10% Calibration

    using stratification by class_id.
    """

    # Step 1:
    # 80% Train
    # 20% temporary
    train_df, temp_df = train_test_split(
        official_train,
        test_size=0.20,
        random_state=seed,
        stratify=official_train["class_id"],
    )

    # Step 2:
    # Split temporary 20% equally:
    # 10% Validation
    # 10% Calibration
    validation_df, calibration_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["class_id"],
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        calibration_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------
# Audit summary
# ---------------------------------------------------------------------

def create_audit_summary(
    class_mapping: dict[int, str],
    selected_folders: dict[str, dict],
    official_train: pd.DataFrame,
    official_test: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
) -> pd.DataFrame:

    folder_by_class_id = {
        info["class_id"]: folder
        for folder, info in selected_folders.items()
    }

    records = []

    for class_id, breed_name in class_mapping.items():
        records.append(
            {
                "class_id": class_id,
                "breed_name": breed_name,
                "stanford_folder": folder_by_class_id[class_id],
                "official_train_count": int(
                    (official_train["class_id"] == class_id).sum()
                ),
                "official_test_count": int(
                    (official_test["class_id"] == class_id).sum()
                ),
                "train_count": int(
                    (train_df["class_id"] == class_id).sum()
                ),
                "validation_count": int(
                    (validation_df["class_id"] == class_id).sum()
                ),
                "calibration_count": int(
                    (calibration_df["class_id"] == class_id).sum()
                ),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

def save_split(dataframe: pd.DataFrame, output_path: Path) -> None:
    """
    Sort before saving so generated files remain deterministic.
    """
    dataframe = dataframe.sort_values(
        ["class_id", "relative_path"]
    ).reset_index(drop=True)

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the fixed 30-class DogBreedID Stanford Dogs dataset."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to data/raw/stanford_dogs",
    )

    parser.add_argument(
        "--class-mapping",
        type=Path,
        default=DEFAULT_CLASS_MAPPING,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SPLIT_SEED,
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    images_dir = dataset_root / "Images"

    train_mat = dataset_root / "train_list.mat"
    test_mat = dataset_root / "test_list.mat"

    # --------------------------------------------------------------
    # Validate required files
    # --------------------------------------------------------------

    required_paths = [
        images_dir,
        train_mat,
        test_mat,
        args.class_mapping,
    ]

    missing_paths = [
        path for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        print("ERROR: Missing required files/directories:")

        for path in missing_paths:
            print(f"  - {path}")

        sys.exit(1)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("DogBreedID — Dataset Preparation")
    print("=" * 70)

    print(f"Dataset root : {dataset_root}")
    print(f"Images       : {images_dir}")
    print(f"Split seed   : {args.seed}")
    print()

    # --------------------------------------------------------------
    # Class mapping
    # --------------------------------------------------------------

    print("[1/7] Loading class mapping...")

    class_mapping = load_class_mapping(
        args.class_mapping
    )

    print(f"      Selected classes: {len(class_mapping)}")

    # --------------------------------------------------------------
    # Match Stanford folders
    # --------------------------------------------------------------

    print("[2/7] Matching Stanford Dogs folders...")

    selected_folders = build_folder_mapping(
        images_dir,
        class_mapping,
    )

    for folder_name, info in sorted(
        selected_folders.items(),
        key=lambda item: item[1]["class_id"],
    ):
        print(
            f"      {info['class_id']:02d} "
            f"{info['breed_name']:<30} "
            f"-> {folder_name}"
        )

    # --------------------------------------------------------------
    # Official lists
    # --------------------------------------------------------------

    print("\n[3/7] Reading official Stanford splits...")

    official_train_files = load_mat_file_list(
        train_mat
    )

    official_test_files = load_mat_file_list(
        test_mat
    )

    print(
        f"      Full Official Train: "
        f"{len(official_train_files)} images"
    )

    print(
        f"      Full Official Test : "
        f"{len(official_test_files)} images"
    )

    official_train = build_dataframe(
        official_train_files,
        selected_folders,
        "official_train",
    )

    official_test = build_dataframe(
        official_test_files,
        selected_folders,
        "official_test",
    )

    print(
        f"      Selected Official Train: "
        f"{len(official_train)} images"
    )

    print(
        f"      Selected Official Test : "
        f"{len(official_test)} images"
    )

    # --------------------------------------------------------------
    # Check every class appears
    # --------------------------------------------------------------

    expected_class_ids = set(range(30))

    train_class_ids = set(
        official_train["class_id"].unique()
    )

    test_class_ids = set(
        official_test["class_id"].unique()
    )

    if train_class_ids != expected_class_ids:
        raise ValueError(
            "Not all 30 classes are present in Official Train."
        )

    if test_class_ids != expected_class_ids:
        raise ValueError(
            "Not all 30 classes are present in Official Test."
        )

    # --------------------------------------------------------------
    # Audit raw images
    # --------------------------------------------------------------

    print("\n[4/7] Auditing selected images...")

    all_selected_images = pd.concat(
        [official_train, official_test],
        ignore_index=True,
    )

    issues = audit_images(
        all_selected_images,
        images_dir,
    )

    issues_path = args.output_dir / "audit_issues.csv"

    if issues:
        issues_df = pd.DataFrame(issues)
        issues_df.to_csv(
            issues_path,
            index=False,
            encoding="utf-8",
        )

        print(
            f"ERROR: Found {len(issues)} image issues."
        )

        print(
            f"See: {issues_path}"
        )

        print(
            "Splits were NOT generated because the raw dataset "
            "must be reviewed first."
        )

        sys.exit(1)

    if issues_path.exists():
        issues_path.unlink()

    print(
        f"      Checked {len(all_selected_images)} images: OK"
    )

    # --------------------------------------------------------------
    # Development split
    # --------------------------------------------------------------

    print("\n[5/7] Creating Train/Validation/Calibration split...")

    train_df, validation_df, calibration_df = (
        create_development_splits(
            official_train,
            args.seed,
        )
    )

    print(f"      Train       : {len(train_df)}")
    print(f"      Validation  : {len(validation_df)}")
    print(f"      Calibration : {len(calibration_df)}")
    print(f"      Official Test: {len(official_test)}")

    # --------------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------------

    print("\n[6/7] Checking data leakage...")

    check_no_overlap(
        {
            "train": train_df,
            "validation": validation_df,
            "calibration": calibration_df,
            "official_test": official_test,
        }
    )

    print("      No overlap found.")

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    print("\n[7/7] Saving split files...")

    save_split(
        train_df,
        args.output_dir / "train.csv",
    )

    save_split(
        validation_df,
        args.output_dir / "validation.csv",
    )

    save_split(
        calibration_df,
        args.output_dir / "calibration.csv",
    )

    save_split(
        official_test,
        args.output_dir / "official_test.csv",
    )

    audit_summary = create_audit_summary(
        class_mapping=class_mapping,
        selected_folders=selected_folders,
        official_train=official_train,
        official_test=official_test,
        train_df=train_df,
        validation_df=validation_df,
        calibration_df=calibration_df,
    )

    audit_summary.to_csv(
        args.output_dir / "dataset_audit.csv",
        index=False,
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 70)

    print(f"Output directory: {args.output_dir}")

    print()
    print("Generated:")
    print("  - train.csv")
    print("  - validation.csv")
    print("  - calibration.csv")
    print("  - official_test.csv")
    print("  - dataset_audit.csv")

    print()
    print("Class distribution:")
    print(
        audit_summary[
            [
                "class_id",
                "breed_name",
                "official_train_count",
                "train_count",
                "validation_count",
                "calibration_count",
                "official_test_count",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()