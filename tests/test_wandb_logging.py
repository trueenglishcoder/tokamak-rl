from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tokamak_rl.config import load_experiment_config
from tokamak_rl.training.cli import _resolve_wandb_config
from tokamak_rl.training.wandb_logging import WandBConfig, WandBLogger


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeArtifact:
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files: list[str] = []
        self.dirs: list[str] = []

    def add_file(self, path: str) -> None:
        self.files.append(path)

    def add_dir(self, path: str) -> None:
        self.dirs.append(path)


class _FakeRun:
    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, object], int | None]] = []
        self.summary: dict[str, object] = {}
        self.artifacts: list[_FakeArtifact] = []
        self.finished = False

    def log(self, values: dict[str, object], step: int | None = None) -> None:
        self.logs.append((dict(values), step))

    def log_artifact(self, artifact: _FakeArtifact) -> None:
        self.artifacts.append(artifact)

    def finish(self) -> None:
        self.finished = True


def test_wandb_config_loads_from_experiment_yaml(tmp_path: Path) -> None:
    sim_config = REPO_ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml"
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        f"name: wandb_cfg\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n"
        f"wandb:\n"
        f"  enabled: true\n"
        f"  project: tokamak-test\n"
        f"  entity: test-user\n"
        f"  name: explicit-run\n"
        f"  group: debug\n"
        f"  mode: offline\n"
        f"  tags: smoke, local\n"
        f"  log_interval_steps: 5\n"
        f"  log_artifacts: false\n",
        encoding="utf-8",
    )

    cfg = load_experiment_config(experiment)

    assert cfg.wandb.enabled is True
    assert cfg.wandb.project == "tokamak-test"
    assert cfg.wandb.entity == "test-user"
    assert cfg.wandb.name == "explicit-run"
    assert cfg.wandb.group == "debug"
    assert cfg.wandb.mode == "offline"
    assert cfg.wandb.tags == ("smoke", "local")
    assert cfg.wandb.log_interval_steps == 5
    assert cfg.wandb.log_artifacts is False


def test_wandb_cli_overrides_config() -> None:
    args = SimpleNamespace(
        wandb=True,
        wandb_project="override-project",
        wandb_entity=None,
        wandb_name=None,
        wandb_group="group-a",
        wandb_mode="offline",
        wandb_tag=["cli", "server"],
        wandb_log_interval_steps=3,
        wandb_no_artifacts=True,
    )

    cfg = _resolve_wandb_config(args=args, base=WandBConfig(project="base-project", log_artifacts=True), experiment_name="exp-a")

    assert cfg.enabled is True
    assert cfg.project == "override-project"
    assert cfg.name == "exp-a"
    assert cfg.group == "group-a"
    assert cfg.mode == "offline"
    assert cfg.tags == ("cli", "server")
    assert cfg.log_interval_steps == 3
    assert cfg.log_artifacts is False


def test_wandb_logger_uses_lazy_fake_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run = _FakeRun()
    fake_wandb = SimpleNamespace(init=lambda **_kwargs: run, Artifact=_FakeArtifact)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}\n", encoding="utf-8")

    logger = WandBLogger(WandBConfig(enabled=True, project="tokamak-test", mode="offline", log_interval_steps=2), config={"total_steps": 10})
    logger.log({"train": {"reward": 1.0, "ignored": "text"}}, step=1)
    logger.log({"train": {"reward": 2.0}}, step=2)
    logger.log_final({"total_steps": 2, "eval_mean_return": 3.0}, artifact_paths={"metrics": metrics}, step=2)
    logger.close()

    assert run.logs[0] == ({"train/reward": 2.0}, 2)
    assert run.logs[1][0]["final/total_steps"] == 2
    assert run.summary["total_steps"] == 2
    assert run.artifacts and run.artifacts[0].files == [str(metrics)]
    assert run.finished is True


def test_wandb_logger_disabled_does_not_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    logger = WandBLogger(WandBConfig(enabled=False), config={})
    logger.log({"train": {"reward": 1.0}}, step=1)
    logger.close()
    assert logger.enabled is False
