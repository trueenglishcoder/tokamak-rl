# Repository Layout

This document describes the current source-tree layout for `tokamak-rl`.

## Versioned Source Areas

```text
README.md         Project overview and common commands
AGENTS.md         AI-agent coding rules
pyproject.toml    Package metadata and dependencies
Dockerfile        Dedicated RL training image; installs tokamak-sim as dependency
docker-compose.yml Local CPU/GPU-profile RL training services
.dockerignore     Local Docker build-context exclusions
Dockerfile.dockerignore Parent-context exclusions for sibling tokamak-sim builds
configs/          YAML presets for experiments, rewards, and randomization
docs/             Human-facing project documentation
scripts/          Directly runnable workflow scripts
tests/            Regression and smoke tests
tokamak_rl/       Importable RL package
```

## Configs

```text
configs/
  experiments/       Experiment-level configs
  randomization/     Domain-randomization presets
  rewards/           Reward weight/config presets
```

The first tested experiment config is:

```text
configs/experiments/t15md_joint_current_boundary.yaml
```

## Scripts

```text
scripts/
  train.py           Training entry point
  evaluate.py        Evaluation entry point
  rollout_policy.py  Rollout/evaluation artifact entry point
  export_policy.py   Policy export entry point
```

Exact command examples are recorded in `docs/workflows.md` after script CLIs are reviewed.

## Package

```text
tokamak_rl/
  actions/           Normalized-to-physical action scaling
  config/            YAML experiment config loading
  env/               Tokamak RL environment, process env workers, and environment config
  evaluation/        Rollout and evaluation utilities
  export/            Actor export and NumPy runtime loading
  networks/          Feedforward actor and critic networks
  observations/      Observation schema and builder
  randomization/     Training-side randomization helpers
  rewards/           Reward components
  training/          Replay buffers and training algorithms
```

Current public package exports include:

```python
from tokamak_rl.actions import ActionScaler
from tokamak_rl.config import ExperimentConfig, load_experiment_config
from tokamak_rl.env import EnvConfig, ProcessTokamakEnv, ProcessVectorEnv, TokamakRLEnv
from tokamak_rl.export import ExportedPolicyPaths, NumpyFeedForwardActor, export_actor, load_numpy_actor
from tokamak_rl.networks import ActorConfig, CriticConfig, FeedForwardActor, FeedForwardQCritic, RecurrentCriticConfig, RecurrentQCritic
from tokamak_rl.observations import ObservationSchema
from tokamak_rl.randomization import DomainRandomizer
from tokamak_rl.rewards import JointCurrentBoundaryReward
from tokamak_rl.training import ReplayBatch, ReplayBuffer, Episode, EpisodeReplayBuffer, RecurrentUpdateResult, SequenceBatch, SimpleTrainerConfig, TCVStyleTrainerConfig, TCVStyleTrainingResult, TrainingResult, evaluate_actor, evaluate_tcv_actor, load_actor_from_checkpoint, save_training_checkpoint, recurrent_critic_update_once, train_simple_actor_critic, train_tcv_style_actor_critic
```

## Training Package

```text
tokamak_rl/training/
  cli.py                     Training CLI for simple and TCV-style trainers
  replay_buffer.py           Fixed-size transition replay buffer
  sequence_replay.py         Complete-episode storage and padded sequence sampling
  recurrent_critic.py        Masked recurrent MPO update
  simple_actor_critic.py     Simple feedforward actor-critic trainer
  tcv_style_actor_critic.py  TCV-style feedforward-actor/recurrent-critic trainer
```

## Tests

```text
tests/
  test_actor.py                Feedforward actor shape, sampling, and validation tests
  test_actor_export.py         PyTorch-to-NumPy actor export parity tests
  test_config_loader.py        Experiment config loader tests
  test_env_reset.py            Environment reset, step, termination, and realism tests
  test_recurrent_critic.py     Recurrent critic and episode replay tests
  test_rollouts.py             Rollout evaluation output tests
  test_core_contracts.py       Action, observation, reward, and randomization contract tests
  test_simple_trainer.py       Simple actor-critic trainer tests
  test_tcv_style_trainer.py    TCV-style recurrent trainer tests
```

## Ignored Local Areas

Expected local generated areas:

```text
runs/             Rollout, training, and evaluation artifacts
output/           Diagnostics, plots, summaries, exported reports
outputs/          Default evaluation, training, and export CLI outputs
checkpoints/      Model checkpoints
_local_archive/   Local notes and scratch files
.venv/            Local virtual environment
.pytest_cache/    Pytest cache
```

## Dependency Boundary

`tokamak-rl` may import public APIs from `tokamak-sim`.

`tokamak-sim` must not import `tokamak-rl`.

Training code, reward definitions, replay buffers, neural-network code, checkpoints, trajectory sampling, evaluation scripts, and export logic belong in `tokamak-rl`.

Randomization in `tokamak-rl` is limited to tokamak-sim-backed sensor and actuator perturbations. Plant parameter randomization belongs behind explicit neutral plant perturbation hooks in `tokamak-sim`.
