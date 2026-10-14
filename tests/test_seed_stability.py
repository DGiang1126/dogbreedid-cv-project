"""Tests for strict, non-destructive E5 tooling using temporary artifacts only."""
from __future__ import annotations

import copy
import csv
import json
import math
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts import run_seed_stability as cli
from src.evaluation import seed_stability as e5
from src.utils.config import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_E1 = ROOT / "configs/experiments/e1_mobilenet_frozen.yaml"
CONFIG_E2 = ROOT / "configs/experiments/e2_mobilenet_partial.yaml"


@pytest.fixture
def config():
    return load_experiment_config(CONFIG_E1)


def _small_config(config):
    result = copy.deepcopy(config)
    result["training"]["max_epochs"] = 5
    result["training"]["early_stopping_patience"] = 2
    return result


def _write_run(root: Path, config: dict, seed: int, *,
               macro=(0.2, 0.4, 0.4, 0.4), top1=(0.3, 0.5, 0.5, 0.5),
               sha="0123456789abcdef0123456789abcdef01234567",
               metadata_overrides=None, metric_overrides=None,
               history_rows=None, empty=()):
    """Write a valid synthetic run fixture, never into the repository outputs."""
    paths = e5.artifact_paths(config["experiment"]["id"], seed, root)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    for key, path in paths.items():
        if key not in empty:
            path.write_bytes(b"fixture")
    run_config = e5.config_for_seed(config, seed)
    metadata = {
        "experiment_id": config["experiment"]["id"], "seed": seed,
        "git_commit": sha, "config": run_config,
        "physical_batch_size": 8, "gradient_accumulation_steps": 4,
        "effective_batch_size": 32,
    }
    metadata.update(metadata_overrides or {})
    metrics = {"best_epoch": 2, "macro_f1": macro[1], "top1_accuracy": top1[1],
               "top5_accuracy": 0.8, "loss": 1.25}
    metrics.update(metric_overrides or {})
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    paths["validation"].write_text(json.dumps(metrics), encoding="utf-8")
    rows = history_rows
    if rows is None:
        top5 = [0.7] + [0.8] * max(0, len(macro) - 1)
        losses = [1.5, 1.25] + [1.3] * max(0, len(macro) - 2)
        rows = [
            {"epoch": i, "val_macro_f1": macro[i - 1], "val_top1_accuracy": top1[i - 1],
             "val_top5_accuracy": top5[i - 1], "val_loss": losses[i - 1]}
            for i in range(1, len(macro) + 1)
        ]
    if "history" not in empty:
        with paths["history"].open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "val_macro_f1", "val_top1_accuracy", "val_top5_accuracy", "val_loss"])
            writer.writeheader()
            writer.writerows(rows)
    return paths


def test_seed_validation_rejects_bool_and_unsupported_values():
    for value in (True, False, 0, 1, 43, "42", None):
        with pytest.raises(ValueError):
            e5.validate_seed(value)
    assert [e5.validate_seed(s) for s in (42, 123, 2026)] == [42, 123, 2026]


def test_config_for_seed_is_immutable_and_changes_only_seed_fields(config):
    original = copy.deepcopy(config)
    actual = e5.config_for_seed(config, 123)
    expected = copy.deepcopy(config)
    expected["experiment"]["seed"] = 123
    expected["training"]["seed"] = 123
    assert actual == expected
    assert config == original
    assert actual is not config
    assert actual["data"]["split_seed"] == 42


def test_seed_config_is_accepted_by_unchanged_shared_loader(tmp_path, config):
    override = e5.config_for_seed(config, 2026)
    # The runner's loader expects an experiment overlay, so write the exact
    # experiment-level source and let it merge the unchanged base/data files.
    experiment = {"experiment": override["experiment"], "model": override["model"],
                  "training": override["training"]}
    path = tmp_path / "outside_configs" / "seed_2026.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    loaded = load_experiment_config(path)
    assert loaded["experiment"]["seed"] == loaded["training"]["seed"] == 2026
    assert loaded["data"]["split_seed"] == 42
    assert loaded["model"] == config["model"]


@pytest.mark.parametrize("bad", [
    lambda c: c["data"].__setitem__("split_seed", 123),
    lambda c: c["experiment"].__setitem__("id", "E5"),
])
def test_config_for_seed_rejects_nonapproved_config(bad, config):
    changed = copy.deepcopy(config)
    bad(changed)
    with pytest.raises(ValueError):
        e5.config_for_seed(changed, 42)


def test_artifact_paths_preserve_original_experiment_and_seed(tmp_path):
    paths = e5.artifact_paths("E4", 2026, tmp_path)
    assert paths["best_checkpoint"] == tmp_path / "outputs/checkpoints/E4/seed_2026/best.pt"
    assert paths["last_checkpoint"] == tmp_path / "outputs/checkpoints/E4/seed_2026/last.pt"
    assert paths["history"] == tmp_path / "outputs/logs/E4/seed_2026/history.csv"
    assert paths["metadata"] == tmp_path / "outputs/logs/E4/seed_2026/run_metadata.json"
    assert paths["validation"] == tmp_path / "outputs/metrics/E4/seed_2026/validation.json"
    with pytest.raises(ValueError):
        e5.artifact_paths("E5", 42, tmp_path)


def test_pair_must_be_distinct_original_experiments():
    with pytest.raises(ValueError, match="distinct"):
        e5.plan_runs([CONFIG_E1, CONFIG_E1])
    with pytest.raises(ValueError, match="exactly two"):
        e5.plan_runs([CONFIG_E1])


@pytest.mark.parametrize("config_path,section,field,value", [
    (CONFIG_E1, "training", "classifier_lr", 0.002),
    (CONFIG_E1, "experiment", "seed", 123),
    (CONFIG_E1, "preprocessing", "train", {"horizontal_flip_p": 0.0}),
    (ROOT / "configs/experiments/e4_cutmix.yaml", "augmentation", "cutmix",
     {"enabled": True, "alpha": 2.0, "probability": 0.5}),
])
def test_official_entrypoints_reject_external_known_id_overlays(
        tmp_path, config_path, section, field, value):
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw.setdefault(section, {})[field] = value
    overlay_path = tmp_path / f"modified_{config_path.name}"
    overlay_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical frozen config"):
        e5.load_original_config(overlay_path)
    with pytest.raises(ValueError, match="canonical frozen config"):
        e5.plan_runs([overlay_path, CONFIG_E2])
    with pytest.raises(ValueError, match="canonical frozen config"):
        e5.run_seed(overlay_path, 42, dry_run=True)


def test_official_entrypoints_accept_canonical_configs_and_reject_e5_placeholder():
    assert e5.load_original_config(CONFIG_E1)["experiment"]["id"] == "E1"
    with pytest.raises(ValueError):
        e5.load_original_config(ROOT / "configs/experiments/e5_seed_stability.yaml")


def test_read_run_and_summary_use_sample_standard_deviation(tmp_path, config):
    vals = {42: (0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4),
            123: (0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
            2026: (0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6)}
    tops = {42: (0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
            123: (0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6),
            2026: (0.5, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7)}
    for seed in e5.SEEDS:
        _write_run(tmp_path, config, seed, macro=vals[seed], top1=tops[seed])
    one = e5.read_run(config, 42, tmp_path)
    assert one["best_epoch"] == 2 and one["macro_f1"] == 0.4
    # Pair collection requires six runs, so add a second independent config.
    config2 = load_experiment_config(CONFIG_E2)
    for seed in e5.SEEDS:
        _write_run(tmp_path, config2, seed, macro=vals[seed], top1=tops[seed])
    summary = e5.collect_summary([CONFIG_E1, CONFIG_E2], tmp_path)
    assert summary["std_ddof"] == 1
    for row in summary["summary"]:
        assert row["macro_f1_mean"] == pytest.approx(0.5)
        assert row["macro_f1_std"] == pytest.approx(0.1)
        assert row["top1_mean"] == pytest.approx(0.6)
        assert row["top1_std"] == pytest.approx(0.1)


def test_write_summary_emits_expected_csv_json_and_refuses_replacement(tmp_path):
    summary = {
        "seeds": [42, 123, 2026], "std_ddof": 1,
        "runs": [{"experiment_id": "E1", "seed": s} for s in (42, 123, 2026)],
        "summary": [{"experiment_id": "E1", "configuration": "mobilenet_v2_frozen", "n": 3,
                     "macro_f1_mean": .5, "macro_f1_std": .1, "top1_mean": .6, "top1_std": .1}],
    }
    csv_path, json_path = e5.write_summary(summary, tmp_path)
    with csv_path.open(encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    json_result = json.loads(json_path.read_text(encoding="utf-8"))
    assert csv_rows == [{"experiment_id": "E1", "configuration": "mobilenet_v2_frozen", "n": "3",
                         "macro_f1_mean": "0.5", "macro_f1_std": "0.1", "top1_mean": "0.6", "top1_std": "0.1"}]
    assert json_result["std_ddof"] == 1 and [r["seed"] for r in json_result["runs"]] == [42, 123, 2026]
    before = (csv_path.read_bytes(), json_path.read_bytes())
    with pytest.raises(FileExistsError):
        e5.write_summary(summary, tmp_path)
    assert (csv_path.read_bytes(), json_path.read_bytes()) == before


def test_summary_refuses_missing_seed_or_missing_artifact(tmp_path, config):
    config = _small_config(config)
    for seed in e5.SEEDS[:-1]:
        _write_run(tmp_path, config, seed)
    with pytest.raises(ValueError, match="seed_2026"):
        e5.collect_summary([CONFIG_E1, CONFIG_E2], tmp_path)
    paths = _write_run(tmp_path, config, 2026, empty=("last_checkpoint",))
    with pytest.raises(ValueError, match="missing/empty"):
        e5.read_run(config, 2026, tmp_path)
    paths["last_checkpoint"].write_bytes(b"nonempty")
    paths["best_checkpoint"].write_bytes(b"")
    with pytest.raises(ValueError, match="missing/empty"):
        e5.read_run(config, 2026, tmp_path)


@pytest.mark.parametrize("case", [
    "wrong_experiment", "wrong_seed", "wrong_config", "invalid_sha", "wrong_batch",
    "nonfinite", "out_of_range", "top5_below_top1", "best_epoch", "best_metrics",
    "nonconsecutive_history", "incomplete_history", "history_after_stop",
])
def test_read_run_rejects_invalid_metadata_metrics_or_history(tmp_path, config, case):
    config = _small_config(config)
    kwargs = {}
    rows = None
    if case == "wrong_experiment": kwargs["metadata_overrides"] = {"experiment_id": "E2"}
    elif case == "wrong_seed": kwargs["metadata_overrides"] = {"seed": 123}
    elif case == "wrong_config":
        changed = e5.config_for_seed(config, 42)
        changed["model"]["backbone_frozen"] = False
        kwargs["metadata_overrides"] = {"config": changed}
    elif case == "invalid_sha": kwargs["sha"] = "unavailable"
    elif case == "wrong_batch": kwargs["metadata_overrides"] = {"effective_batch_size": 16}
    elif case == "nonfinite": kwargs["metric_overrides"] = {"macro_f1": float("nan")}
    elif case == "out_of_range": kwargs["metric_overrides"] = {"top1_accuracy": 1.2}
    elif case == "top5_below_top1": kwargs["metric_overrides"] = {"top5_accuracy": 0.3}
    elif case == "best_epoch": kwargs["metric_overrides"] = {"best_epoch": 3}
    elif case == "best_metrics": kwargs["metric_overrides"] = {"loss": 0.7}
    elif case == "nonconsecutive_history":
        rows = [{"epoch": 1, "val_macro_f1": .2, "val_top1_accuracy": .3, "val_top5_accuracy": .7, "val_loss": 1.5},
                {"epoch": 3, "val_macro_f1": .4, "val_top1_accuracy": .5, "val_top5_accuracy": .8, "val_loss": 1.25}]
    elif case == "incomplete_history":
        rows = [{"epoch": 1, "val_macro_f1": .2, "val_top1_accuracy": .3, "val_top5_accuracy": .7, "val_loss": 1.5},
                {"epoch": 2, "val_macro_f1": .4, "val_top1_accuracy": .5, "val_top5_accuracy": .8, "val_loss": 1.25}]
    elif case == "history_after_stop":
        rows = [{"epoch": i, "val_macro_f1": .4 if i >= 2 else .2,
                 "val_top1_accuracy": .5 if i >= 2 else .3,
                 "val_top5_accuracy": .8 if i >= 2 else .7,
                 "val_loss": 1.25 if i >= 2 else 1.5} for i in range(1, 6)]
    _write_run(tmp_path, config, 42, history_rows=rows, **kwargs)
    with pytest.raises(ValueError):
        e5.read_run(config, 42, tmp_path)


def test_read_run_rejects_empty_or_missing_history_rows(tmp_path, config):
    config = _small_config(config)
    rows = [{"epoch": 1, "val_macro_f1": .2}]
    paths = _write_run(tmp_path, config, 42, history_rows=rows)
    with pytest.raises(ValueError):
        e5.read_run(config, 42, tmp_path)
    paths["history"].write_text("epoch,val_macro_f1,val_top1_accuracy,val_top5_accuracy,val_loss\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        e5.read_run(config, 42, tmp_path)


def test_read_run_rejects_top5_below_top1_in_history(tmp_path, config):
    config = _small_config(config)
    rows = [
        {"epoch": 1, "val_macro_f1": .2, "val_top1_accuracy": .3, "val_top5_accuracy": .4, "val_loss": 1.5},
        {"epoch": 2, "val_macro_f1": .4, "val_top1_accuracy": .5, "val_top5_accuracy": .4, "val_loss": 1.25},
        {"epoch": 3, "val_macro_f1": .4, "val_top1_accuracy": .5, "val_top5_accuracy": .6, "val_loss": 1.3},
        {"epoch": 4, "val_macro_f1": .4, "val_top1_accuracy": .5, "val_top5_accuracy": .6, "val_loss": 1.3},
    ]
    _write_run(tmp_path, config, 42, history_rows=rows)
    with pytest.raises(ValueError, match="Top-5"):
        e5.read_run(config, 42, tmp_path)


def test_plan_is_inventory_only_and_reports_partial_run(tmp_path):
    paths = e5.artifact_paths("E1", 42, tmp_path)
    paths["history"].parent.mkdir(parents=True)
    paths["history"].write_text("partial", encoding="utf-8")
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    plan = e5.plan_runs([CONFIG_E1, CONFIG_E2], tmp_path)
    statuses = {(r["experiment_id"], r["seed"]): r["status"] for r in plan}
    assert statuses[("E1", 42)] == "existing_invalid_or_incomplete"
    assert statuses[("E1", 123)] == statuses[("E2", 2026)] == "not_started"
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before


def test_no_overwrite_existing_partial_run_summary_or_lock(tmp_path, config, monkeypatch):
    config = _small_config(config)
    paths = e5.artifact_paths("E1", 42, tmp_path)
    paths["history"].parent.mkdir(parents=True)
    paths["history"].write_text("partial original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        e5.run_seed(CONFIG_E1, 42, dry_run=False, drive_upload_confirmed=True, project_root=tmp_path)
    assert paths["history"].read_text(encoding="utf-8") == "partial original"
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    (summary_dir / "seed_stability_summary.csv").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        e5.write_summary({"summary": []}, summary_dir)
    assert (summary_dir / "seed_stability_summary.csv").read_text() == "existing"
    lock = paths["history"].parent / ".seed_stability.lock"
    lock.write_text("busy", encoding="utf-8")
    monkeypatch.setattr(e5, "_published_source", lambda _: "a" * 40)
    # Remove the earlier partial artifact to reach the exclusive lock check.
    paths["history"].unlink()
    with pytest.raises(FileExistsError):
        e5.run_seed(CONFIG_E1, 42, dry_run=False, drive_upload_confirmed=True, project_root=tmp_path)
    assert lock.read_text() == "busy"


def test_runner_requires_drive_confirmation_and_published_source(config, monkeypatch, tmp_path):
    monkeypatch.setattr(e5, "_published_source", lambda _: (_ for _ in ()).throw(ValueError("source not published")))
    with pytest.raises(ValueError, match="shared Drive"):
        e5.run_seed(CONFIG_E1, 42, dry_run=False, project_root=tmp_path)
    with pytest.raises(ValueError, match="source not published"):
        e5.run_seed(CONFIG_E1, 42, dry_run=False, drive_upload_confirmed=True, project_root=tmp_path)


@pytest.mark.parametrize("config_path,experiment_id", [
    (CONFIG_E1, "E1"), (ROOT / "configs/experiments/e4_cutmix.yaml", "E4")])
def test_runner_invokes_fresh_shared_process_with_seed_config_and_cleans_temp(
        monkeypatch, tmp_path, config_path, experiment_id):
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/train.py").write_text("# unchanged runner placeholder", encoding="utf-8")
    captured = {}
    def fake_run(command, *, cwd, check):
        captured.update(command=command, cwd=cwd, check=check)
        temp_config = Path(command[command.index("--config") + 1])
        captured["temp_config"] = temp_config
        captured["config_exists_during_call"] = temp_config.is_file()
        captured["loaded_config"] = load_experiment_config(temp_config)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(e5.subprocess, "run", fake_run)
    result = e5.run_seed(config_path, 123, batch_size=8, num_workers=2, device="cpu",
                         dry_run=True, dataset_root="/dataset", project_root=root)
    cmd = captured["command"]
    assert captured["cwd"] == root and captured["check"] is True
    assert cmd[cmd.index("--config") + 1] == str(captured["temp_config"])
    assert cmd[cmd.index("--batch-size") + 1] == "8"
    assert cmd[cmd.index("--num-workers") + 1] == "2"
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert cmd[cmd.index("--dataset-root") + 1] == "/dataset"
    assert cmd[-1] == "--dry-run" and str(root / "scripts/train.py") in cmd
    assert captured["loaded_config"]["experiment"]["id"] == experiment_id
    assert captured["loaded_config"]["experiment"]["seed"] == captured["loaded_config"]["training"]["seed"] == 123
    assert captured["loaded_config"]["data"]["split_seed"] == 42
    if experiment_id == "E4":
        assert captured["loaded_config"]["augmentation"]["cutmix"] == {"enabled": True, "alpha": 1.0, "probability": 0.5}
    assert not captured["temp_config"].exists()
    assert result["seed"] == 123 and result["dry_run"] is True


def test_runner_failure_cleans_temp_and_lock_but_preserves_partial_outputs(monkeypatch, tmp_path):
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/train.py").write_text("runner", encoding="utf-8")
    monkeypatch.setattr(e5, "_published_source", lambda _: "a" * 40)
    observed = {}
    def fail_after_partial(command, *, cwd, check):
        cfg = Path(command[command.index("--config") + 1])
        observed["cfg"] = cfg
        paths = e5.artifact_paths("E1", 42, root)
        paths["history"].parent.mkdir(parents=True, exist_ok=True)
        paths["history"].write_text("partial output", encoding="utf-8")
        raise subprocess.CalledProcessError(2, command)
    monkeypatch.setattr(e5.subprocess, "run", fail_after_partial)
    with pytest.raises(subprocess.CalledProcessError):
        e5.run_seed(CONFIG_E1, 42, dry_run=False, drive_upload_confirmed=True, project_root=root)
    assert not observed["cfg"].exists()
    assert not (root / "outputs/logs/E1/seed_42/.seed_stability.lock").exists()
    assert (root / "outputs/logs/E1/seed_42/history.csv").read_text() == "partial output"


@pytest.mark.parametrize("metadata_sha,should_succeed", [
    ("f" * 40, True), ("e" * 40, False),
])
def test_official_runner_validates_actual_sha_and_cleans_resources(
        monkeypatch, tmp_path, metadata_sha, should_succeed):
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/train.py").write_text("unchanged shared runner", encoding="utf-8")
    source_sha = "f" * 40
    monkeypatch.setattr(e5, "_published_source", lambda _: source_sha)
    captured = {}

    def fake_train(command, *, cwd, check):
        cfg_path = Path(command[command.index("--config") + 1])
        captured["temp_config"] = cfg_path
        loaded = load_experiment_config(cfg_path)
        captured["loaded"] = loaded
        _write_run(root, loaded, 42, sha=metadata_sha,
                   macro=(.2, .4, .4, .4, .4, .4, .4, .4, .4),
                   top1=(.3, .5, .5, .5, .5, .5, .5, .5, .5))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(e5.subprocess, "run", fake_train)
    paths = e5.artifact_paths("E1", 42, root)
    if should_succeed:
        result = e5.run_seed(CONFIG_E1, 42, dry_run=False, drive_upload_confirmed=True, project_root=root)
        assert result["git_commit"] == source_sha
        assert captured["loaded"]["experiment"]["seed"] == captured["loaded"]["training"]["seed"] == 42
        assert all(p.is_file() and p.stat().st_size for p in paths.values())
    else:
        with pytest.raises(ValueError, match="metadata SHA"):
            e5.run_seed(CONFIG_E1, 42, dry_run=False, drive_upload_confirmed=True, project_root=root)
        assert all(p.is_file() and p.stat().st_size for p in paths.values())
    assert not captured["temp_config"].exists()
    assert not (paths["history"].parent / ".seed_stability.lock").exists()


def test_cli_plan_readonly_and_summary_failure_writes_nothing(tmp_path, monkeypatch, capsys):
    original_cwd = Path.cwd()
    plan = e5.plan_runs([CONFIG_E1, CONFIG_E2], tmp_path)
    assert len(plan) == 6
    assert not tmp_path.exists() or not list(tmp_path.iterdir())
    monkeypatch.setattr(cli, "plan_runs", lambda configs: e5.plan_runs(configs, tmp_path))
    cli.main(["plan", "--configs", str(CONFIG_E1), str(CONFIG_E2)])
    plan_output = json.loads(capsys.readouterr().out)
    assert len(plan_output) == 6
    assert {row["status"] for row in plan_output} == {"not_started"}
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.main(["summarize", "--configs", str(CONFIG_E1), str(CONFIG_E2)])
    assert exc.value.code == 1
    assert "seed_42" in capsys.readouterr().err
    assert not (tmp_path / "outputs/metrics/E5").exists()
    assert Path.cwd() == original_cwd

