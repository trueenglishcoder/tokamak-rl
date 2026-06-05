from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tokamak_rl.env import EnvConfig, TerminationConfig, TokamakRLEnv
from tokamak_rl.randomization import DomainRandomizer
from tokamak_rl.rewards import JointCurrentBoundaryReward
from tokamak_control.realism import SensorRealismSettings


def _write_small_sim_config(path: Path, *, realism=None) -> None:
    from tokamak_control.config.settings import PhysicsSettings
    from tokamak_control.core.coils import Coil, CoilActuator, CoilGroup
    from tokamak_control.core.grid import Grid1D, Grid2D
    from tokamak_control.io.config_io import dump_config

    grid = Grid2D(
        r=Grid1D(start=0.2, step=0.02, size=121, center=1.2),
        z=Grid1D(start=-1.2, step=0.02, size=121, center=0.0),
    )
    pfc = CoilGroup(
        name="pfc",
        coils=[CoilActuator([Coil(0.8, 0.8)]), CoilActuator([Coil(1.6, -0.8)])],
        currents=np.array([1000.0, -1000.0], dtype=float),
    )
    sol = CoilGroup(
        name="sol",
        coils=[CoilActuator([Coil(1.2, 1.0)])],
        currents=np.array([500.0], dtype=float),
    )
    physics = PhysicsSettings(
        Ip0=5.0e4,
        R0=1.2,
        Z0=0.0,
        sigma=2.0e6,
        inductance_L=2.0e-6,
        t_step=1.0e-3,
        pfc_current_limit=1.0e5,
        sol_current_limit=1.0e5,
        pfc_deriv_limit=1.0e6,
        sol_deriv_limit=1.0e6,
    )
    dump_config(
        path,
        grid=grid,
        pfc=pfc,
        sol=sol,
        physics=physics,
        realism=realism,
        limiter_name="T15MD",
        boundary_mode="limited",
    )


def test_tokamak_env_reset_returns_observation_and_info(tmp_path: Path) -> None:
    config_path = tmp_path / "small_sim.toml"
    _write_small_sim_config(config_path)
    env = TokamakRLEnv(
        EnvConfig(
            sim_config_path=config_path,
            scenario_name="nominal",
            angles=8,
            max_episode_steps=3,
            realism_enabled=False,
        )
    )

    obs, info = env.reset(seed=123)

    assert obs.shape == (env.obs_dim,)
    assert env.action_dim == 3
    assert info["machine"].n_active_total == 3
    assert info["snapshot"].measured_active_currents.shape == (3,)
    assert info["snapshot"].measured_radii is not None
    assert info["config_path"] == config_path
    env.close()


def test_tokamak_env_reset_can_start_from_zero_ip_and_zero_coils(tmp_path: Path) -> None:
    config_path = tmp_path / "small_sim_zero.toml"
    _write_small_sim_config(config_path)
    env = TokamakRLEnv(
        EnvConfig(
            sim_config_path=config_path,
            scenario_name="t15_synthetic_follow",
            scenario_args={
                "duration_s": 0.05,
                "t_step": 1.0e-3,
                "ip_start": 0.0,
                "ip_end": 10_000.0,
                "ip_ramp_s": 0.05,
                "boundary_kind": "static_parameters",
                "boundary_parameters": {"R0": 1.2, "Z0": 0.0, "A0": 0.35, "kappa": 1.0, "delta": 0.0},
            },
            angles=8,
            max_episode_steps=3,
            realism_enabled=False,
            initial_ip=0.0,
            initial_coil_currents="zero",
            initial_ip_scale=5.0e5,
            termination=TerminationConfig(boundary_loss_grace_steps=3),
        )
    )

    obs, info = env.reset(seed=123)

    assert obs.shape == (env.obs_dim,)
    assert info["snapshot"].true_ip == 0.0
    assert info["snapshot"].reference.ip_ref == 0.0
    assert np.allclose(info["snapshot"].true_active_currents, np.zeros((env.action_dim,), dtype=float))
    assert info["machine"].ip_scale == pytest.approx(5.0e5)
    assert info["episode_metadata"]["initial_state_override"]["coil_currents"] == "zero"
    assert info["episode_metadata"]["training_contract"]["environment"]["initial_ip"] == 0.0
    assert info["episode_metadata"]["training_contract"]["environment"]["initial_ip_scale"] == pytest.approx(5.0e5)
    env.close()


def _make_env(tmp_path: Path, *, max_episode_steps: int = 3) -> TokamakRLEnv:
    config_path = tmp_path / "small_sim.toml"
    _write_small_sim_config(config_path)
    return TokamakRLEnv(
        EnvConfig(
            sim_config_path=config_path,
            scenario_name="nominal",
            angles=8,
            max_episode_steps=max_episode_steps,
            realism_enabled=False,
        )
    )


def test_tokamak_env_step_zero_action_returns_gym_tuple(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    env.reset(seed=123)

    obs, reward, terminated, truncated, info = env.step(np.zeros((env.action_dim,), dtype=float))

    assert obs.shape == (env.obs_dim,)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert info["snapshot"].step_index == 1
    assert info["physical_derivatives"].shape == (env.action_dim,)
    env.close()


def test_tokamak_env_step_validates_action_shape_and_finiteness(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    env.reset(seed=123)

    with pytest.raises(ValueError, match="action shape"):
        env.step(np.zeros((env.action_dim + 1,), dtype=float))
    with pytest.raises(ValueError, match="finite"):
        bad = np.zeros((env.action_dim,), dtype=float)
        bad[0] = np.nan
        env.step(bad)
    env.close()


def test_tokamak_env_step_writes_previous_action_into_next_observation(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    env.reset(seed=123)
    action = np.array([0.25, -0.5, 1.5], dtype=float)

    obs, _reward, _terminated, _truncated, info = env.step(action)

    previous_slice = env.schema.field_slices["previous_action_norm"]
    assert np.allclose(obs[previous_slice], np.array([0.25, -0.5, 1.0], dtype=np.float32))
    assert np.allclose(info["action_norm"], np.array([0.25, -0.5, 1.0]))
    env.close()


def test_tokamak_env_truncates_at_episode_length(tmp_path: Path) -> None:
    env = _make_env(tmp_path, max_episode_steps=1)
    env.reset(seed=123)

    _obs, _reward, terminated, truncated, _info = env.step(np.zeros((env.action_dim,), dtype=float))

    assert not terminated
    assert truncated
    env.close()


def test_tokamak_env_terminated_missing_boundary_observation_does_not_reuse_stale_boundary(tmp_path: Path) -> None:
    from tokamak_control.bridge import StepResult

    env = _make_env(tmp_path)
    _obs0, info0 = env.reset(seed=123)
    missing_snapshot = replace(
        info0["snapshot"],
        step_index=1,
        true_boundary_poly=None,
        measured_boundary_poly=None,
        true_radii=None,
        measured_radii=None,
        boundary_found=False,
        boundary_reason="test missing boundary",
    )

    class FakeSession:
        def step_derivatives(self, action):
            return StepResult(snapshot=missing_snapshot, terminated=True, truncated=False, termination_reason="test missing boundary")

        def close(self):
            return None

    env.session = FakeSession()
    obs, reward, terminated, truncated, info = env.step(np.zeros((env.action_dim,), dtype=float))

    slices = env.schema.field_slices
    assert terminated
    assert not truncated
    assert reward < 0.0
    assert obs[1] == 0.0
    assert np.allclose(obs[slices["radii_meas_norm"]], np.zeros((8,), dtype=np.float32))
    assert np.allclose(obs[slices["radii_error_norm"]], np.zeros((8,), dtype=np.float32))
    assert info["termination_reason"] == "boundary_not_found"
    assert info["termination_detail"] == "test missing boundary"
    env.close()


def test_tokamak_env_boundary_loss_grace_keeps_startup_episode_alive(tmp_path: Path) -> None:
    from tokamak_control.bridge import StepResult

    env = TokamakRLEnv(
        EnvConfig(
            sim_config_path=tmp_path / "small_sim.toml",
            scenario_name="nominal",
            angles=8,
            max_episode_steps=3,
            realism_enabled=False,
            termination=TerminationConfig(boundary_loss_grace_steps=2),
        )
    )
    _write_small_sim_config(env.cfg.sim_config_path)
    _obs0, info0 = env.reset(seed=123)
    missing_snapshot = replace(
        info0["snapshot"],
        step_index=1,
        true_boundary_poly=None,
        measured_boundary_poly=None,
        true_radii=None,
        measured_radii=None,
        boundary_found=False,
        boundary_reason="startup no boundary",
    )

    class FakeSession:
        def step_derivatives(self, action):
            return StepResult(snapshot=missing_snapshot, terminated=False, truncated=False, termination_reason=None)

        def close(self):
            return None

    env.session = FakeSession()

    _obs, _reward, terminated, truncated, info = env.step(np.zeros((env.action_dim,), dtype=float))

    assert not terminated
    assert not truncated
    assert info["termination_reason"] is None
    env.close()


def test_tokamak_env_terminates_on_configured_measured_boundary_missing(tmp_path: Path) -> None:
    from tokamak_control.bridge import StepResult

    config_path = tmp_path / "small_sim.toml"
    _write_small_sim_config(config_path)
    env = TokamakRLEnv(
        EnvConfig(
            sim_config_path=config_path,
            scenario_name="nominal",
            angles=8,
            max_episode_steps=3,
            realism_enabled=False,
            termination=TerminationConfig(measured_boundary_missing_steps=1),
        )
    )
    _obs0, info0 = env.reset(seed=123)
    missing_measured_snapshot = replace(
        info0["snapshot"],
        step_index=1,
        measured_boundary_poly=None,
        measured_radii=None,
        boundary_found=True,
    )

    class FakeSession:
        def step_derivatives(self, action):
            return StepResult(snapshot=missing_measured_snapshot, terminated=False, truncated=False, termination_reason=None)

        def close(self):
            return None

    env.session = FakeSession()

    _obs, _reward, terminated, truncated, info = env.step(np.zeros((env.action_dim,), dtype=float))

    assert terminated
    assert not truncated
    assert info["termination_reason"] == "measured_boundary_missing"
    env.close()


def test_tokamak_env_terminates_on_configured_current_limit_margin(tmp_path: Path) -> None:
    from tokamak_control.bridge import StepResult

    config_path = tmp_path / "small_sim.toml"
    _write_small_sim_config(config_path)
    env = TokamakRLEnv(
        EnvConfig(
            sim_config_path=config_path,
            scenario_name="nominal",
            angles=8,
            max_episode_steps=3,
            realism_enabled=False,
            termination=TerminationConfig(current_limit_margin_min=0.2),
        )
    )
    _obs0, info0 = env.reset(seed=123)
    limit_snapshot = replace(info0["snapshot"], step_index=1, current_limit_margin=np.array([0.1, 0.3, 0.4], dtype=float))

    class FakeSession:
        def step_derivatives(self, action):
            return StepResult(snapshot=limit_snapshot, terminated=False, truncated=False, termination_reason=None)

        def close(self):
            return None

    env.session = FakeSession()

    _obs, _reward, terminated, _truncated, info = env.step(np.zeros((env.action_dim,), dtype=float))

    assert terminated
    assert info["termination_reason"] == "current_limit_breach"
    assert "0.2" in info["termination_detail"]
    env.close()


def test_tokamak_env_realism_observation_uses_measured_ip_but_reward_uses_true_ip(tmp_path: Path) -> None:
    from tokamak_control.realism import RealismSettings, SensorRealismSettings

    config_path = tmp_path / "small_sim_realism.toml"
    ip_bias = 250.0
    _write_small_sim_config(
        config_path,
        realism=RealismSettings(enabled=True, seed=11, sensors=SensorRealismSettings(ip_bias=ip_bias)),
    )
    env = TokamakRLEnv(
        EnvConfig(
            sim_config_path=config_path,
            scenario_name="nominal",
            angles=8,
            max_episode_steps=3,
            realism_enabled=True,
        )
    )

    obs0, info0 = env.reset(seed=123)

    ip_meas_slice = env.schema.field_slices["ip_meas_norm"]
    measured0 = float(info0["snapshot"].measured_ip)
    true0 = float(info0["snapshot"].true_ip)
    assert np.isclose(measured0 - true0, ip_bias)
    assert np.isclose(float(obs0[ip_meas_slice][0]), measured0 / info0["machine"].ip_scale)

    _obs, _reward, _terminated, _truncated, info = env.step(np.zeros((env.action_dim,), dtype=float))

    snapshot = info["snapshot"]
    true_error_norm = abs(float(snapshot.true_ip) - float(snapshot.reference.ip_ref)) / info0["machine"].ip_scale
    measured_error_norm = abs(float(snapshot.measured_ip) - float(snapshot.reference.ip_ref)) / info0["machine"].ip_scale
    assert np.isclose(info["reward_components"]["ip_error_norm"], true_error_norm)
    assert not np.isclose(info["reward_components"]["ip_error_norm"], measured_error_norm)
    env.close()


def test_tokamak_env_reset_uses_t15_synthetic_reference_source(tmp_path: Path) -> None:
    config_path = tmp_path / "small_sim.toml"
    ip_path = tmp_path / "t15md_4242_ip.csv"
    _write_small_sim_config(config_path)
    ip_path.write_text("0;0\n0.5;100000\n1.0;150000\n", encoding="utf-8")
    env = TokamakRLEnv(
        EnvConfig(
            sim_config_path=config_path,
            scenario_name="t15_synthetic_follow",
            scenario_args={
                "seed": 5,
                "duration_s": 0.05,
                "t_step": 1.0e-3,
                "target_update_s": 0.01,
                "ip_template_csv": str(ip_path),
                "ip_seed": 7,
                "amplitude_jitter": 0.0,
                "duration_jitter": 0.0,
                "shape_jitter": 0.0,
            },
            angles=8,
            max_episode_steps=3,
            realism_enabled=False,
        )
    )

    obs0, info0 = env.reset(seed=123)
    obs1, _reward, _terminated, _truncated, info1 = env.step(np.zeros((env.action_dim,), dtype=float))

    assert obs0.shape == (env.obs_dim,)
    assert info0["snapshot"].reference.radii_ref.shape == (8,)
    assert np.all(np.isfinite(info0["snapshot"].reference.radii_ref))
    assert info0["snapshot"].reference.ip_ref == pytest.approx(0.0)
    assert info1["snapshot"].reference.ip_ref > info0["snapshot"].reference.ip_ref
    assert np.all(np.isfinite(obs1))
    env.close()


def test_tokamak_env_v2_observation_includes_future_target_preview(tmp_path: Path) -> None:
    config_path = tmp_path / "small_sim.toml"
    _write_small_sim_config(config_path)
    env = TokamakRLEnv(
        EnvConfig(
            sim_config_path=config_path,
            scenario_name="t15_synthetic_follow",
            scenario_args={
                "seed": 5,
                "duration_s": 0.05,
                "t_step": 1.0e-3,
                "target_update_s": 0.01,
                "ip_start": 0.0,
                "ip_end": 100000.0,
                "ip_ramp_s": 0.05,
                "boundary_kind": "static_parameters",
                "boundary_parameters": {"R0": 1.2, "Z0": 0.0, "A0": 0.35, "kappa": 1.0, "delta": 0.0},
            },
            angles=8,
            max_episode_steps=10,
            realism_enabled=False,
            observation_version="v2",
            target_preview_steps=3,
            target_preview_stride=2,
        )
    )

    obs, info = env.reset(seed=123)
    slices = env.schema.field_slices

    assert env.schema.version == "v2"
    assert obs.shape == (env.obs_dim,)
    assert env.obs_dim == 5 + 2 * 3 + 3 * 8 + 3 * (2 + 8)
    assert np.allclose(obs[slices["target_preview_time_norm"]], np.array([0.2, 0.4, 0.6], dtype=np.float32))
    assert np.all(obs[slices["ip_ref_preview_norm"]] > obs[slices["ip_ref_norm"]][0])
    preview_radii = obs[slices["radii_ref_preview_norm"]].reshape(3, 8)
    assert np.all(np.isfinite(preview_radii))
    assert info["episode_metadata"]["training_contract"]["observation_schema"]["schema_version"] == "v2"
    assert info["episode_metadata"]["training_contract"]["target_preview"] == {"steps": 3, "stride": 2}
    env.close()


def test_tokamak_env_resamples_synthetic_reference_by_reset_seed(tmp_path: Path) -> None:
    config_path = tmp_path / "small_sim.toml"
    ip_path = tmp_path / "t15md_5151_ip.csv"
    _write_small_sim_config(config_path)
    ip_path.write_text("0;0\n0.5;100000\n1.0;150000\n", encoding="utf-8")
    env = TokamakRLEnv(
        EnvConfig(
            sim_config_path=config_path,
            scenario_name="t15_synthetic_follow",
            scenario_args={
                "seed": 5,
                "duration_s": 0.05,
                "t_step": 1.0e-3,
                "target_update_s": 0.01,
                "ip_template_csv": str(ip_path),
                "ip_seed": 7,
                "amplitude_jitter": 0.0,
                "duration_jitter": 0.0,
                "shape_jitter": 0.0,
            },
            angles=8,
            max_episode_steps=3,
            realism_enabled=False,
            resample_references_on_reset=True,
        )
    )

    _obs_a, info_a = env.reset(seed=100)
    radii_a = info_a["snapshot"].reference.radii_ref.copy()
    meta_a = info_a["episode_metadata"]
    _obs_b, info_b = env.reset(seed=100)
    radii_b = info_b["snapshot"].reference.radii_ref.copy()
    _obs_c, info_c = env.reset(seed=101)
    radii_c = info_c["snapshot"].reference.radii_ref.copy()

    assert meta_a["reference_resampling_enabled"] is True
    assert meta_a["reference_base_seed"] == 5
    assert meta_a["reference_base_ip_seed"] == 7
    assert meta_a["reference_episode_seed"] == 100
    assert meta_a["reference_effective_seed"] == info_b["episode_metadata"]["reference_effective_seed"]
    assert np.allclose(radii_a, radii_b)
    assert not np.allclose(radii_a, radii_c)
    assert meta_a["reference_effective_seed"] != info_c["episode_metadata"]["reference_effective_seed"]
    env.close()


def test_tokamak_env_can_keep_synthetic_reference_fixed_across_reset_seeds(tmp_path: Path) -> None:
    config_path = tmp_path / "small_sim.toml"
    ip_path = tmp_path / "t15md_6161_ip.csv"
    _write_small_sim_config(config_path)
    ip_path.write_text("0;0\n0.5;100000\n1.0;150000\n", encoding="utf-8")
    env = TokamakRLEnv(
        EnvConfig(
            sim_config_path=config_path,
            scenario_name="t15_synthetic_follow",
            scenario_args={
                "seed": 5,
                "duration_s": 0.05,
                "t_step": 1.0e-3,
                "target_update_s": 0.01,
                "ip_template_csv": str(ip_path),
                "ip_seed": 7,
                "amplitude_jitter": 0.0,
                "duration_jitter": 0.0,
                "shape_jitter": 0.0,
            },
            angles=8,
            max_episode_steps=3,
            realism_enabled=False,
            resample_references_on_reset=False,
        )
    )

    _obs_a, info_a = env.reset(seed=100)
    radii_a = info_a["snapshot"].reference.radii_ref.copy()
    _obs_b, info_b = env.reset(seed=101)
    radii_b = info_b["snapshot"].reference.radii_ref.copy()

    assert info_a["episode_metadata"]["reference_resampling_enabled"] is False
    assert np.allclose(radii_a, radii_b)
    env.close()


def test_tokamak_env_uses_supplied_reward_function(tmp_path: Path) -> None:
    config_path = tmp_path / "small_sim.toml"
    _write_small_sim_config(config_path)
    zero_reward = JointCurrentBoundaryReward(
        ip_weight=0.0,
        shape_weight=0.0,
        action_weight=0.0,
        delta_action_weight=0.0,
        current_limit_weight=0.0,
        derivative_limit_weight=0.0,
        termination_penalty=0.0,
    )
    env = TokamakRLEnv(
        EnvConfig(
            sim_config_path=config_path,
            scenario_name="nominal",
            angles=8,
            max_episode_steps=3,
            realism_enabled=False,
        ),
        reward_fn=zero_reward,
    )

    env.reset(seed=123)
    _obs, reward, _terminated, _truncated, info = env.step(np.zeros((env.action_dim,), dtype=float))

    assert reward == pytest.approx(0.0)
    assert info["reward_components"]["r_ip"] >= 0.0
    assert info["reward_components"]["r_shape"] >= 0.0
    env.close()


def test_tokamak_env_reset_records_supplied_randomization_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "small_sim.toml"
    _write_small_sim_config(config_path)
    env = TokamakRLEnv(
        EnvConfig(
            sim_config_path=config_path,
            scenario_name="nominal",
            angles=8,
            max_episode_steps=3,
            realism_enabled=False,
        ),
        randomizer=DomainRandomizer(enabled=True),
    )

    _obs, info = env.reset(seed=123)

    randomization = info["episode_metadata"]["randomization"]
    assert randomization["enabled"] is True
    assert randomization["seed"] == 123
    assert randomization["simulator_realism"]["enabled"] is True
    env.close()


def test_tokamak_env_randomization_applies_simulator_sensor_noise(tmp_path: Path) -> None:
    config_path = tmp_path / "small_sim.toml"
    _write_small_sim_config(config_path)
    ip_bias = 123.0
    env = TokamakRLEnv(
        EnvConfig(
            sim_config_path=config_path,
            scenario_name="nominal",
            angles=8,
            max_episode_steps=3,
            realism_enabled=False,
        ),
        randomizer=DomainRandomizer(enabled=True, sensors=SensorRealismSettings(ip_bias=ip_bias)),
    )

    _obs, info = env.reset(seed=123)
    snapshot = info["snapshot"]

    assert np.isclose(snapshot.measured_ip - snapshot.true_ip, ip_bias)
    assert info["episode_metadata"]["randomization"]["has_nonzero_effect"] is True
    assert info["episode_metadata"]["realism_active"] is True
    env.close()
