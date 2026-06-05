from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tokamak_rl.export import export_actor, load_actor_from_training_checkpoint
from tokamak_rl.networks import ActorConfig, FeedForwardActor
from tokamak_rl.observations import ObservationSchema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a deterministic actor to a NumPy-compatible artifact.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for exported_policy files.")
    parser.add_argument("--obs-dim", required=True, type=int, help="Actor observation dimension.")
    parser.add_argument("--action-dim", required=True, type=int, help="Actor action dimension.")
    parser.add_argument("--n-active-total", required=True, type=int, help="Active coil count in actor/action order.")
    parser.add_argument("--n-angles", required=True, type=int, help="Boundary/radii angle count.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional PyTorch actor state_dict checkpoint.")
    args = parser.parse_args(argv)

    schema = ObservationSchema(n_active_total=args.n_active_total, n_angles=args.n_angles)
    if schema.obs_dim != int(args.obs_dim):
        raise SystemExit(f"--obs-dim {args.obs_dim} does not match schema dimension {schema.obs_dim}")
    actor = FeedForwardActor(ActorConfig(obs_dim=args.obs_dim, action_dim=args.action_dim))
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(state, dict) and "actor_config" in state and "actor_state_dict" in state:
            actor, _checkpoint = load_actor_from_training_checkpoint(args.checkpoint, device="cpu")
            if int(actor.obs_dim) != int(args.obs_dim) or int(actor.action_dim) != int(args.action_dim):
                raise SystemExit("--obs-dim/--action-dim do not match checkpoint actor dimensions")
        else:
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            actor.load_state_dict(state)
    normalization = {
        "ip_scale": 1.0,
        "radius_scale": 1.0,
        "current_scale": np.ones((args.n_active_total,), dtype=float),
        "derivative_scale": np.ones((args.action_dim,), dtype=float),
        "phase": "step_index / max_episode_steps",
    }
    paths = export_actor(
        actor,
        args.out,
        schema=schema,
        normalization=normalization,
        metadata={"checkpoint_path": args.checkpoint},
    )
    print(paths.output_dir)
    return 0
