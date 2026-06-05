from __future__ import annotations


TRAINING_READINESS_CONTRACT_VERSION = "training_readiness_v1"


KNOWN_TERMINATION_REASONS = (
    "max_episode_steps",
    "boundary_not_found",
    "measured_boundary_missing",
    "simulator_terminated",
    "invalid_observation",
    "invalid_reward",
    "current_limit_breach",
    "derivative_limit_breach",
    "simulator_exception",
)


REQUIRED_TRAINING_ARTIFACTS = (
    "metrics.json",
    "config_snapshot.json",
    "losses.csv",
    "episodes.csv",
    "eval_history.csv",
    "reward_components.csv",
    "reference_samples.npz",
    "termination_counts.json",
    "artifact_manifest.json",
)

CONDITIONAL_TRAINING_ARTIFACTS = (
    "exports/best_actor/",
    "rollouts/",
)


__all__ = [
    "KNOWN_TERMINATION_REASONS",
    "CONDITIONAL_TRAINING_ARTIFACTS",
    "REQUIRED_TRAINING_ARTIFACTS",
    "TRAINING_READINESS_CONTRACT_VERSION",
]
