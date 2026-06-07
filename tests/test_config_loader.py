from __future__ import annotations

from pathlib import Path

import pytest

from tokamak_rl.config import load_experiment_config
from tokamak_rl.training.cli import _make_eval_env_factory
from tests.test_env_reset import _write_small_sim_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_experiment_config_resolves_paths() -> None:
    cfg = load_experiment_config(REPO_ROOT / "configs/experiments/t15md_joint_current_boundary.yaml")

    assert cfg.name == "t15md_joint_current_boundary"
    assert cfg.env.sim_config_path.name == "T15MD_new_data.toml"
    assert cfg.env.sim_config_path.exists()
    assert cfg.reward_config_path is not None
    assert cfg.reward_config_path.name == "joint_current_boundary.yaml"
    assert cfg.reward.ip_weight == pytest.approx(1.0)
    assert cfg.reward.shape_tolerance_norm == pytest.approx(0.02)
    assert cfg.randomization_config_path is not None
    assert cfg.randomization.enabled is False
    assert cfg.env.resample_references_on_reset is True


def test_load_randomization_config_parses_simulator_noise(tmp_path: Path) -> None:
    sim_config = REPO_ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml"
    randomization_config = tmp_path / "randomization.yaml"
    randomization_config.write_text(
        "enabled: true\n"
        "sensors:\n"
        "  ip_bias: 12.5\n"
        "  radii_noise_sigma: 0.001\n"
        "actuators:\n"
        "  pfc_command_noise_sigma: 3.0\n",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        f"name: randomization_noise\n"
        f"randomization_config: {randomization_config}\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n",
        encoding="utf-8",
    )

    cfg = load_experiment_config(experiment)

    assert cfg.randomization.enabled is True
    assert cfg.randomization.sensors.ip_bias == pytest.approx(12.5)
    assert cfg.randomization.sensors.radii_noise_sigma == pytest.approx(0.001)
    assert cfg.randomization.actuators.pfc_command_noise_sigma == pytest.approx(3.0)


def test_randomization_config_rejects_unknown_fields(tmp_path: Path) -> None:
    sim_config = REPO_ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml"
    randomization_config = tmp_path / "randomization.yaml"
    randomization_config.write_text("enabled: true\nunknown: 1\n", encoding="utf-8")
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        f"name: bad_randomization\n"
        f"randomization_config: {randomization_config}\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown fields in randomization"):
        load_experiment_config(experiment)


def test_stage_m_training_preset_configs_load() -> None:
    real_like = load_experiment_config(REPO_ROOT / "configs/experiments/t15md_training_real_replay_like.yaml")
    circle = load_experiment_config(REPO_ROOT / "configs/experiments/t15md_training_circle_static_boundary.yaml")
    smoke = load_experiment_config(REPO_ROOT / "configs/experiments/t15md_training_real_replay_like_smoke.yaml")
    keep_boundary = load_experiment_config(REPO_ROOT / "configs/experiments/t15md_training_keep_initial_boundary.yaml")

    assert real_like.env.scenario_name == "t15_synthetic_follow"
    assert real_like.env.scenario_args["reference_preset"] == "t15_real_replay_like"
    assert real_like.env.scenario_args["ip_segmented"] is True
    assert real_like.env.scenario_args["boundary_bounds"]["R0"] == {"min": 1.3266, "max": 1.4756}
    assert real_like.env.observation_version == "v2"
    assert real_like.env.target_preview_steps == 8
    assert real_like.env.initial_coil_currents == "sample_ranges"
    assert real_like.env.initial_ip is None
    assert real_like.env.replay_initial_state is None
    assert real_like.env.range_initial_state is not None
    assert real_like.env.range_initial_state.ip == pytest.approx((124750.45168962443, 124967.92554755499))
    assert len(real_like.env.range_initial_state.pfc_currents) == 6
    assert len(real_like.env.range_initial_state.sol_currents) == 3
    assert real_like.env.range_initial_state.boundary_parameters["kappa"][0] == pytest.approx(1.1120020187817363)
    assert real_like.env.scenario_args["boundary_bounds"]["kappa"] == {"min": 1.1108, "max": 1.4966}
    assert real_like.env.scenario_args["boundary_bounds"]["delta"] == {"min": 0.0992, "max": 0.3656}
    assert keep_boundary.env.scenario_name == "t15_synthetic_follow"
    assert keep_boundary.env.initial_coil_currents == "sample_ranges"
    assert keep_boundary.env.scenario_args["boundary_kind"] == "static_parameters"
    assert keep_boundary.env.scenario_args["boundary_static_from_initial_state"] is True
    assert "boundary_parameters" not in keep_boundary.env.scenario_args
    assert circle.env.scenario_name == "t15_synthetic_follow"
    assert circle.env.scenario_args["reference_preset"] == "circle_static_boundary"
    assert circle.env.scenario_args["boundary_parameters"]["kappa"] == pytest.approx(1.0)
    assert circle.env.scenario_args["boundary_parameters"]["delta"] == pytest.approx(0.0)
    assert circle.env.observation_version == "v2"
    assert circle.env.target_preview_stride == 10
    assert smoke.training.enabled is True
    assert smoke.training.trainer == "tcv_style"
    assert smoke.training.total_steps == 8
    assert smoke.training.sequence_length == 2
    assert smoke.training.hidden_dim == 16
    assert smoke.training.mpo_action_samples == 4
    assert smoke.training.mpo_temperature_iterations == 3
    assert smoke.training.mpo_mean_kl_epsilon == pytest.approx(0.01)
    assert smoke.evaluation.validation_seed == 9000
    assert smoke.evaluation.max_steps == 2
    assert smoke.evaluation.randomization_mode == "clean"
    assert smoke.artifacts.checkpoint_interval_steps == 4
    assert smoke.artifacts.max_step_checkpoints == 2
    assert smoke.artifacts.export_best_actor is True


def test_load_experiment_config_requires_sim_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: bad\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sim must be a mapping"):
        load_experiment_config(path)


def test_load_experiment_config_rejects_bad_paths(tmp_path: Path) -> None:
    path = tmp_path / "bad_path.yaml"
    path.write_text(
        "name: bad\n"
        "sim:\n"
        "  config_path: does-not-exist.toml\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="sim.config_path does not exist"):
        load_experiment_config(path)


def test_load_experiment_config_rejects_unknown_top_level_fields(tmp_path: Path) -> None:
    sim_config = REPO_ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml"
    path = tmp_path / "bad_top.yaml"
    path.write_text(
        f"name: bad_top\n"
        f"unknown: 1\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown fields in experiment"):
        load_experiment_config(path)


def test_training_config_rejects_unknown_fields(tmp_path: Path) -> None:
    sim_config = REPO_ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml"
    path = tmp_path / "bad_training.yaml"
    path.write_text(
        f"name: bad_training\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n"
        f"training:\n"
        f"  enabled: true\n"
        f"  mystery: 9\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown fields in training"):
        load_experiment_config(path)


def test_evaluation_config_rejects_unknown_randomization_mode(tmp_path: Path) -> None:
    sim_config = REPO_ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml"
    path = tmp_path / "bad_eval_mode.yaml"
    path.write_text(
        f"name: bad_eval_mode\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n"
        f"evaluation:\n"
        f"  randomization_mode: noisyish\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evaluation.randomization_mode"):
        load_experiment_config(path)


def test_eval_env_factory_supports_clean_and_configured_randomization(tmp_path: Path) -> None:
    sim_config = tmp_path / "small_sim.toml"
    _write_small_sim_config(sim_config)
    randomization_config = tmp_path / "randomization.yaml"
    randomization_config.write_text("enabled: true\nsensors:\n  ip_bias: 10.0\n", encoding="utf-8")
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        f"name: eval_modes\n"
        f"randomization_config: {randomization_config}\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n"
        f"  realism_enabled: true\n"
        f"evaluation:\n"
        f"  randomization_mode: clean\n",
        encoding="utf-8",
    )
    experiment = load_experiment_config(experiment_path)

    configured_env = _make_eval_env_factory(
        experiment=experiment,
        process_envs=False,
        process_start_method="spawn",
        randomization_mode="configured",
    )()
    clean_env = _make_eval_env_factory(
        experiment=experiment,
        process_envs=False,
        process_start_method="spawn",
        randomization_mode="clean",
    )()
    try:
        _configured_obs, configured_info = configured_env.reset(seed=5)
        _clean_obs, clean_info = clean_env.reset(seed=5)
    finally:
        configured_env.close()
        clean_env.close()

    configured_contract = configured_info["episode_metadata"]["training_contract"]
    clean_contract = clean_info["episode_metadata"]["training_contract"]
    assert configured_contract["environment"]["realism_enabled"] is True
    assert configured_contract["randomization"]["enabled"] is True
    assert clean_contract["environment"]["realism_enabled"] is False
    assert clean_contract["randomization"]["enabled"] is False


def test_reward_config_rejects_unknown_fields(tmp_path: Path) -> None:
    sim_config = REPO_ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml"
    reward_config = tmp_path / "reward.yaml"
    reward_config.write_text("ip_weight: 1.0\nunknown: 2.0\n", encoding="utf-8")
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        f"name: bad_reward\n"
        f"reward_config: {reward_config}\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown reward config fields"):
        load_experiment_config(experiment)


def test_reference_source_translates_to_t15_synthetic_follow(tmp_path: Path) -> None:
    sim_config = REPO_ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml"
    template_dir = tmp_path / "ip"
    template_dir.mkdir()
    (template_dir / "t15md_1001_ip.csv").write_text("0;0\n1;100\n", encoding="utf-8")
    experiment = tmp_path / "synthetic.yaml"
    experiment.write_text(
        f"name: synthetic\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n"
        f"  initial_state:\n"
        f"    ip: 0.0\n"
        f"    coil_currents: zero\n"
        f"    ip_scale: 500000.0\n"
        f"  reference_source:\n"
        f"    kind: t15_synthetic_follow\n"
        f"    seed: 12\n"
        f"    duration_s: 1.5\n"
        f"    t_step: 0.001\n"
        f"    target_update_s: 0.05\n"
        f"    theta_count: 256\n"
        f"    ip:\n"
        f"      kind: template_dir\n"
        f"      path: {template_dir}\n"
        f"      seed: 91\n"
        f"      amplitude_jitter: 0.04\n"
        f"      duration_jitter: 0.03\n"
        f"      shape_jitter: 0.02\n"
        f"      start: 0.0\n",
        encoding="utf-8",
    )

    cfg = load_experiment_config(experiment)

    assert cfg.env.scenario_name == "t15_synthetic_follow"
    assert cfg.env.scenario_args["seed"] == 12
    assert cfg.env.scenario_args["duration_s"] == 1.5
    assert cfg.env.scenario_args["target_update_s"] == 0.05
    assert cfg.env.scenario_args["theta_count"] == 256
    assert cfg.env.scenario_args["ip_template_dir"] == str(template_dir.resolve())
    assert cfg.env.scenario_args["ip_seed"] == 91
    assert cfg.env.scenario_args["amplitude_jitter"] == 0.04
    assert cfg.env.scenario_args["ip_start"] == pytest.approx(0.0)
    assert cfg.env.initial_ip == pytest.approx(0.0)
    assert cfg.env.initial_coil_currents == "zero"
    assert cfg.env.initial_ip_scale == pytest.approx(500000.0)


def test_reference_source_translates_segmented_ip_and_static_circle_boundary(tmp_path: Path) -> None:
    sim_config = REPO_ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml"
    experiment = tmp_path / "circle.yaml"
    experiment.write_text(
        f"name: circle\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n"
        f"  reference_source:\n"
        f"    kind: t15_synthetic_follow\n"
        f"    preset: circle_static_boundary\n"
        f"    seed: 3\n"
        f"    duration_s: 0.5\n"
        f"    t_step: 0.001\n"
        f"    ip:\n"
        f"      kind: segmented\n"
        f"      seed: 91\n"
        f"      min: 100000\n"
        f"      max: 420000\n"
        f"      segment_min_steps: 20\n"
        f"      segment_max_steps: 50\n"
        f"      segment_count_min: 3\n"
        f"      segment_count_max: 5\n"
        f"      max_steps: 500\n"
        f"      rate_limit: 8000000\n"
        f"      hold_probability: 0.4\n"
        f"      start: 0.0\n"
        f"    boundary:\n"
        f"      kind: static_parameters\n"
        f"      parameters:\n"
        f"        R0: 1.4\n"
        f"        Z0: 0.0\n"
        f"        A0: 0.55\n"
        f"        kappa: 1.0\n"
        f"        delta: 0.0\n",
        encoding="utf-8",
    )

    cfg = load_experiment_config(experiment)

    assert cfg.env.scenario_name == "t15_synthetic_follow"
    assert cfg.env.scenario_args["reference_preset"] == "circle_static_boundary"
    assert cfg.env.scenario_args["ip_segmented"] is True
    assert cfg.env.scenario_args["ip_seed"] == 91
    assert cfg.env.scenario_args["ip_min"] == pytest.approx(100000.0)
    assert cfg.env.scenario_args["ip_max"] == pytest.approx(420000.0)
    assert cfg.env.scenario_args["ip_segment_min_steps"] == 20
    assert cfg.env.scenario_args["ip_segment_count_max"] == 5
    assert cfg.env.scenario_args["ip_rate_limit"] == pytest.approx(8000000.0)
    assert cfg.env.scenario_args["ip_start"] == pytest.approx(0.0)
    assert cfg.env.scenario_args["boundary_kind"] == "static_parameters"
    assert cfg.env.scenario_args["boundary_parameters"] == {"R0": 1.4, "Z0": 0.0, "A0": 0.55, "kappa": 1.0, "delta": 0.0}


def test_reference_source_translates_generated_boundary_bounds_and_rates(tmp_path: Path) -> None:
    sim_config = REPO_ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml"
    experiment = tmp_path / "real_like.yaml"
    experiment.write_text(
        f"name: real_like\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n"
        f"  reference_source:\n"
        f"    kind: t15_synthetic_follow\n"
        f"    preset: t15_real_replay_like\n"
        f"    boundary:\n"
        f"      kind: generated_parameters\n"
        f"      bounds:\n"
        f"        R0:\n"
        f"          min: 1.3266\n"
        f"          max: 1.4756\n"
        f"        Z0:\n"
        f"          min: -0.0496\n"
        f"          max: 0.0137\n"
        f"        A0:\n"
        f"          min: 0.5200\n"
        f"          max: 0.6641\n"
        f"        kappa:\n"
        f"          min: 1.1212\n"
        f"          max: 1.4966\n"
        f"        delta:\n"
        f"          min: 0.1044\n"
        f"          max: 0.3656\n"
        f"      rate_limits:\n"
        f"        R0: 0.30\n"
        f"        Z0: 0.70\n"
        f"        A0: 0.45\n"
        f"        kappa: 1.20\n"
        f"        delta: 0.80\n",
        encoding="utf-8",
    )

    cfg = load_experiment_config(experiment)

    assert cfg.env.scenario_args["reference_preset"] == "t15_real_replay_like"
    assert cfg.env.scenario_args["boundary_kind"] == "generated_parameters"
    assert cfg.env.scenario_args["boundary_bounds"]["R0"] == {"min": 1.3266, "max": 1.4756}
    assert cfg.env.scenario_args["boundary_rate_limits"]["delta"] == pytest.approx(0.80)


def test_reference_source_rejects_ambiguous_scenario_config(tmp_path: Path) -> None:
    sim_config = REPO_ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml"
    experiment = tmp_path / "ambiguous.yaml"
    experiment.write_text(
        f"name: ambiguous\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n"
        f"  scenario_name: nominal\n"
        f"  scenario_args:\n"
        f"    ip_start: 1.0\n"
        f"  reference_source:\n"
        f"    kind: t15_synthetic_follow\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reference_source cannot be combined"):
        load_experiment_config(experiment)


def test_reference_source_rejects_unknown_ip_kind(tmp_path: Path) -> None:
    sim_config = REPO_ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml"
    experiment = tmp_path / "bad_ip.yaml"
    experiment.write_text(
        f"name: bad_ip\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n"
        f"  reference_source:\n"
        f"    kind: t15_synthetic_follow\n"
        f"    ip:\n"
        f"      kind: made_up\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ip.kind"):
        load_experiment_config(experiment)


def test_reference_source_rejects_unknown_boundary_kind(tmp_path: Path) -> None:
    sim_config = REPO_ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml"
    experiment = tmp_path / "bad_boundary.yaml"
    experiment.write_text(
        f"name: bad_boundary\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n"
        f"  reference_source:\n"
        f"    kind: t15_synthetic_follow\n"
        f"    boundary:\n"
        f"      kind: made_up\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="boundary.kind"):
        load_experiment_config(experiment)


def test_reference_resampling_flag_can_be_disabled(tmp_path: Path) -> None:
    sim_config = REPO_ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml"
    experiment = tmp_path / "fixed_reference.yaml"
    experiment.write_text(
        f"name: fixed_reference\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n"
        f"  resample_references_on_reset: false\n",
        encoding="utf-8",
    )

    cfg = load_experiment_config(experiment)

    assert cfg.env.resample_references_on_reset is False


def test_observation_v2_config_loads_target_preview_settings(tmp_path: Path) -> None:
    sim_config = REPO_ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml"
    experiment = tmp_path / "observation_v2.yaml"
    experiment.write_text(
        f"name: observation_v2\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n"
        f"  observation:\n"
        f"    version: v2\n"
        f"    target_preview_steps: 4\n"
        f"    target_preview_stride: 3\n",
        encoding="utf-8",
    )

    cfg = load_experiment_config(experiment)

    assert cfg.env.observation_version == "v2"
    assert cfg.env.target_preview_steps == 4
    assert cfg.env.target_preview_stride == 3


def test_observation_v1_rejects_preview_steps(tmp_path: Path) -> None:
    sim_config = REPO_ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml"
    experiment = tmp_path / "bad_observation.yaml"
    experiment.write_text(
        f"name: bad_observation\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n"
        f"  observation:\n"
        f"    version: v1\n"
        f"    target_preview_steps: 2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target_preview_steps"):
        load_experiment_config(experiment)


def test_termination_config_loads_explicit_rules(tmp_path: Path) -> None:
    sim_config = REPO_ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml"
    experiment = tmp_path / "termination.yaml"
    experiment.write_text(
        f"name: termination\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n"
        f"  termination:\n"
        f"    terminate_on_boundary_loss: true\n"
        f"    boundary_loss_grace_steps: 50\n"
        f"    terminate_on_nonfinite_observation: true\n"
        f"    terminate_on_nonfinite_reward: true\n"
        f"    current_limit_margin_min: 0.15\n"
        f"    derivative_limit_margin_min: 0.20\n"
        f"    measured_boundary_missing_steps: 2\n",
        encoding="utf-8",
    )

    cfg = load_experiment_config(experiment)

    assert cfg.env.termination.current_limit_margin_min == pytest.approx(0.15)
    assert cfg.env.termination.derivative_limit_margin_min == pytest.approx(0.20)
    assert cfg.env.termination.measured_boundary_missing_steps == 2
    assert cfg.env.termination.boundary_loss_grace_steps == 50


def test_termination_config_rejects_unknown_fields(tmp_path: Path) -> None:
    sim_config = REPO_ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml"
    experiment = tmp_path / "bad_termination.yaml"
    experiment.write_text(
        f"name: bad_termination\n"
        f"sim:\n"
        f"  config_path: {sim_config}\n"
        f"  termination:\n"
        f"    made_up: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sim.termination"):
        load_experiment_config(experiment)
