from __future__ import annotations

import argparse
from pathlib import Path

from tokamak_rl.config import load_experiment_config
from tokamak_rl.evaluation.rollouts import RolloutConfig, run_rollout_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tokamak-rl-evaluate", description="Run deterministic tokamak-rl rollouts.")
    parser.add_argument("--config", required=True, help="Experiment YAML path.")
    parser.add_argument("--out", default="outputs/evaluation", help="Output directory.")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes. Defaults to config value or 1.")
    parser.add_argument("--policy", choices=("zero", "random"), default="zero", help="Evaluation policy.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    args = parser.parse_args(argv)

    experiment = load_experiment_config(args.config)
    episodes = int(args.episodes) if args.episodes is not None else 1
    result = run_rollout_evaluation(
        RolloutConfig(
            env=experiment.env,
            episodes=episodes,
            policy=args.policy,
            seed=int(args.seed),
            output_dir=Path(args.out),
            reward=experiment.reward,
            randomizer=experiment.randomization,
        )
    )
    outputs = result.get("outputs")
    if outputs is not None:
        print(outputs.output_dir)
        print(outputs.summary_json)
        print(outputs.episode_metrics_csv)
        print(outputs.rollouts_npz)
    return 0
