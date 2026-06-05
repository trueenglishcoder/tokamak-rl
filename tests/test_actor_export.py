from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pytest
import torch

from tokamak_rl.export import export_actor, export_actor_from_training_checkpoint, load_numpy_actor
from tokamak_rl.networks import ActorConfig, FeedForwardActor
from tokamak_rl.observations import ObservationSchema


def test_exported_numpy_actor_matches_pytorch_deterministic_actor(tmp_path: Path) -> None:
    torch.manual_seed(12)
    schema = ObservationSchema(n_active_total=3, n_angles=8)
    actor = FeedForwardActor(ActorConfig(obs_dim=schema.obs_dim, action_dim=3))
    obs = torch.randn(7, schema.obs_dim)
    normalization = {
        "ip_scale": 100000.0,
        "radius_scale": 1.0,
        "current_scale": np.array([1.0e5, 1.0e5, 1.0e5]),
        "derivative_scale": np.array([1.0e6, 1.0e6, 1.0e6]),
        "phase": "step_index / max_episode_steps",
    }

    paths = export_actor(
        actor,
        tmp_path / "exported_policy",
        schema=schema,
        normalization=normalization,
        metadata={"training_run_id": "unit-test"},
    )
    np_actor = load_numpy_actor(paths.output_dir, expected_schema={"schema_version": "v1", "obs_dim": schema.obs_dim, "action_dim": 3})

    with torch.no_grad():
        torch_action = actor.deterministic_action(obs).detach().cpu().numpy()
    numpy_action = np_actor.deterministic_action(obs.detach().cpu().numpy())

    assert np.max(np.abs(torch_action - numpy_action)) < 2.0e-6
    assert paths.policy_weights_npz.exists()
    assert paths.schema_json.exists()
    assert paths.normalization_json.exists()
    assert paths.metadata_json.exists()


def test_export_metadata_files_are_runtime_mean_path_only(tmp_path: Path) -> None:
    schema = ObservationSchema(n_active_total=2, n_angles=4)
    actor = FeedForwardActor(ActorConfig(obs_dim=schema.obs_dim, action_dim=2))

    paths = export_actor(
        actor,
        tmp_path / "exported_policy",
        schema=schema,
        normalization={"ip_scale": 1.0, "radius_scale": 1.0, "current_scale": [1.0, 1.0], "derivative_scale": [1.0, 1.0]},
    )

    schema_json = json.loads(paths.schema_json.read_text(encoding="utf-8"))
    normalization_json = json.loads(paths.normalization_json.read_text(encoding="utf-8"))
    metadata_json = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    with np.load(paths.policy_weights_npz, allow_pickle=False) as weights:
        names = set(weights.files)

    assert schema_json["obs_dim"] == schema.obs_dim
    assert schema_json["action_dim"] == 2
    assert schema_json["field_order"] == list(schema.field_order)
    assert normalization_json["current_scale"] == [1.0, 1.0]
    assert metadata_json["actor_architecture_name"] == actor.architecture_name
    assert "mean_head.weight" in names
    assert "std_head.weight" not in names


def test_export_rejects_schema_actor_mismatch(tmp_path: Path) -> None:
    schema = ObservationSchema(n_active_total=2, n_angles=4)
    actor = FeedForwardActor(ActorConfig(obs_dim=schema.obs_dim + 1, action_dim=2))

    with pytest.raises(ValueError, match="schema obs_dim"):
        export_actor(actor, tmp_path / "exported_policy", schema=schema, normalization={})


def test_numpy_actor_rejects_expected_schema_mismatch(tmp_path: Path) -> None:
    schema = ObservationSchema(n_active_total=2, n_angles=4)
    actor = FeedForwardActor(ActorConfig(obs_dim=schema.obs_dim, action_dim=2))
    paths = export_actor(actor, tmp_path / "exported_policy", schema=schema, normalization={})

    with pytest.raises(ValueError, match="exported schema mismatch"):
        load_numpy_actor(paths.output_dir, expected_schema={"obs_dim": schema.obs_dim + 1})


def test_numpy_actor_rejects_bad_observation(tmp_path: Path) -> None:
    schema = ObservationSchema(n_active_total=2, n_angles=4)
    actor = FeedForwardActor(ActorConfig(obs_dim=schema.obs_dim, action_dim=2))
    paths = export_actor(actor, tmp_path / "exported_policy", schema=schema, normalization={})
    np_actor = load_numpy_actor(paths.output_dir)

    with pytest.raises(ValueError, match="observation shape"):
        np_actor.deterministic_action(np.zeros((schema.obs_dim,), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        bad = np.zeros((1, schema.obs_dim), dtype=np.float32)
        bad[0, 0] = np.nan
        np_actor.deterministic_action(bad)


def test_export_package_import_keeps_numpy_loader_available() -> None:
    import tokamak_rl.export as export_pkg

    assert export_pkg.load_numpy_actor is load_numpy_actor


def test_export_actor_from_full_training_checkpoint(tmp_path: Path) -> None:
    schema = ObservationSchema(n_active_total=2, n_angles=4)
    actor = FeedForwardActor(ActorConfig(obs_dim=schema.obs_dim, action_dim=2, hidden_dim=16))
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "trainer": "unit_trainer",
            "algorithm": "unit_algorithm",
            "total_steps": 12,
            "best_eval_score": -1.25,
            "actor_config": asdict(actor.cfg),
            "actor_state_dict": actor.state_dict(),
        },
        checkpoint,
    )

    paths = export_actor_from_training_checkpoint(
        checkpoint,
        tmp_path / "exported_best",
        schema=schema,
        normalization={"ip_scale": 1.0, "radius_scale": 1.0, "current_scale": [1.0, 1.0], "derivative_scale": [2.0, 3.0]},
    )

    loaded = load_numpy_actor(paths.output_dir, expected_schema={"obs_dim": schema.obs_dim, "action_dim": 2})
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert metadata["checkpoint_trainer"] == "unit_trainer"
    assert metadata["checkpoint_algorithm"] == "unit_algorithm"
    assert loaded.deterministic_action(np.zeros((1, schema.obs_dim), dtype=np.float32)).shape == (1, 2)
