"""Shared E0–E3 entrypoint: data smoke-check now; full train after model owners implement builders."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import DogBreedDataset
from src.data.transforms import build_transforms
from src.models.model_factory import build_model, build_optimizer
from src.training.checkpoint import save_checkpoint
from src.training.early_stopping import EarlyStopping
from src.training.trainer import train_one_epoch
from src.training.validator import validate_one_epoch
from src.utils.config import load_experiment_config
from src.utils.device import resolve_device, use_mixed_precision
from src.utils.logger import append_history, run_metadata, write_json
from src.utils.paths import dataset_root, output_files, split_file
from src.utils.seed import set_seed, seed_worker


def main():
    parser = argparse.ArgumentParser(description="DogBreedID shared E0–E3 runner")
    parser.add_argument("--config", required=True, help="configs/experiments/e0_...yaml")
    parser.add_argument("--dataset-root", default=None, help="Root containing Images/; default: data/raw/stanford_dogs or DOGBREEDID_DATA_ROOT")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0, help="Default 0 is portable on Windows and Colab")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Only check Train and Validation loaders; no model/training.")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive; num-workers cannot be negative")
    effective = int(config["training"]["effective_batch_size"])
    if effective % args.batch_size != 0:
        parser.error(f"Physical batch-size {args.batch_size} must divide frozen effective batch {effective}.")
    accumulation = effective // args.batch_size
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    root = dataset_root(args.dataset_root)
    train_ds = DogBreedDataset(split_file("train"), root / "Images", build_transforms(config, "train"))
    val_ds = DogBreedDataset(split_file("validation"), root / "Images", build_transforms(config, "validation"))
    train_gen = torch.Generator().manual_seed(seed)
    val_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker, generator=train_gen,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker, generator=val_gen,
    )
    print(f"Experiment={config['experiment']['id']}; seed={seed}")
    print(f"Train={len(train_ds)}, Validation={len(val_ds)}, effective_batch={effective}, accumulation={accumulation}")
    if args.dry_run:
        for name, loader in [("train", train_loader), ("validation", val_loader)]:
            sample = next(iter(loader))
            print(f"{name}: image={tuple(sample['image'].shape)}; label={tuple(sample['label'].shape)}; example={sample['relative_path'][0]}")
        print("DATA DRY-RUN PASSED. Official Test was not accessed.")
        return
    device = resolve_device(args.device)
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler_cfg = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=scheduler_cfg["mode"], factor=float(scheduler_cfg["factor"]),
        patience=int(scheduler_cfg["patience"]), min_lr=float(scheduler_cfg["min_lr"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision(device, config))
    stopper = EarlyStopping(patience=int(config["training"]["early_stopping_patience"]))
    paths = output_files(config["experiment"]["id"], seed)
    if paths["history"].exists():
        raise FileExistsError(f"Run history already exists. Do not overwrite an experiment silently: {paths['history']}")
    write_json(paths["metadata"], run_metadata(config, device, args.batch_size, accumulation))
    for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
        training = train_one_epoch(model, train_loader, criterion, optimizer, device, accumulation, scaler)
        validation = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(validation["macro_f1"])
        improved, should_stop = stopper.step(validation["macro_f1"])
        row = {"epoch": epoch,
               **{f"train_{key}": value for key, value in training.items()},
               **{f"val_{key}": value for key, value in validation.items()},
               "learning_rate": optimizer.param_groups[0]["lr"]}
        append_history(paths["history"], row)
        if improved:
            save_checkpoint(paths["best_checkpoint"], model, optimizer, epoch, validation, config)
            write_json(paths["validation_metrics"], {"best_epoch": epoch, **validation})
        save_checkpoint(paths["last_checkpoint"], model, optimizer, epoch, validation, config)
        print(f"Epoch {epoch:02d} | train loss={training['loss']:.4f} | val macro-F1={validation['macro_f1']:.4f} | best={stopper.best:.4f}")
        if should_stop:
            print(f"Early stopping after {epoch} epochs (patience={stopper.patience}).")
            break
    print("Finished. Only Train/Validation have been used.")


if __name__ == "__main__":
    main()
