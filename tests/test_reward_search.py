from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokamak_rl.rewards import JointCurrentBoundaryReward
from tokamak_rl.training.reward_search import (
    RewardSearchScoreWeights,
    generate_candidates,
    parse_float_values,
    score_metrics,
    write_reward_config,
)


def test_generate_candidates_uses_cartesian_grid_and_base_defaults() -> None:
    base = JointCurrentBoundaryReward(action_weight=0.25, shape_tolerance_norm=0.03)
    candidates = generate_candidates(
        base_reward=base,
        search_space={"ip_weight": (0.5, 1.0), "shape_weight": (2.0,), "termination_penalty": (5.0, 10.0)},
        max_candidates=0,
        seed=0,
    )

    assert len(candidates) == 4
    assert candidates[0].reward.ip_weight == pytest.approx(0.5)
    assert candidates[0].reward.shape_weight == pytest.approx(2.0)
    assert candidates[0].reward.action_weight == pytest.approx(0.25)
    assert candidates[0].reward.shape_tolerance_norm == pytest.approx(0.03)
    assert {candidate.reward.termination_penalty for candidate in candidates} == {5.0, 10.0}


def test_generate_candidates_subsamples_reproducibly() -> None:
    base = JointCurrentBoundaryReward()
    space = {"ip_weight": (0.5, 1.0, 2.0), "shape_weight": (0.5, 1.0, 2.0)}

    first = generate_candidates(base_reward=base, search_space=space, max_candidates=4, seed=123)
    second = generate_candidates(base_reward=base, search_space=space, max_candidates=4, seed=123)

    assert [candidate.reward for candidate in first] == [candidate.reward for candidate in second]
    assert len(first) == 4


def test_score_metrics_uses_physical_diagnostics_not_return_scale() -> None:
    metrics = {
        "eval_returns": [1000.0],
        "eval_tracking_diagnostics": {
            "mean_ip_error_norm": 0.02,
            "mean_shape_error_norm": 0.03,
            "boundary_failure_rate": 0.1,
        },
    }

    score = score_metrics(metrics, weights=RewardSearchScoreWeights(ip_error=2.0, shape_error=3.0, boundary_failure=10.0))

    assert score == pytest.approx(2.0 * 0.02 + 3.0 * 0.03 + 10.0 * 0.1)


def test_score_metrics_penalizes_missing_diagnostics() -> None:
    score = score_metrics({}, weights=RewardSearchScoreWeights(missing_metric=7.0))

    assert score == pytest.approx(7.0 + 7.0 + 10.0 * 7.0)


def test_parse_float_values_and_write_reward_config(tmp_path: Path) -> None:
    assert parse_float_values("0.5, 1.0,2", default=9.0) == (0.5, 1.0, 2.0)
    assert parse_float_values(None, default=9.0) == (9.0,)
    with pytest.raises(ValueError):
        parse_float_values(",", default=1.0)

    path = tmp_path / "reward.yaml"
    write_reward_config(path, JointCurrentBoundaryReward(ip_weight=2.0, shape_weight=3.0))
    text = path.read_text(encoding="utf-8")
    assert "ip_weight" in text
    assert "shape_weight" in text
    if text.lstrip().startswith("{"):
        data = json.loads(text)
        assert data["ip_weight"] == pytest.approx(2.0)
