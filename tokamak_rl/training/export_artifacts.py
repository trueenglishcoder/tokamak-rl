from __future__ import annotations

from pathlib import Path
from typing import Mapping

from tokamak_rl.contracts import TRAINING_READINESS_CONTRACT_VERSION
from tokamak_rl.export import export_actor_from_training_checkpoint
from tokamak_rl.observations import ObservationSchema
from tokamak_rl.training.diagnostics import json_safe


def export_best_actor_artifact(
    *,
    checkpoint_path: Path | None,
    output_dir: Path | None,
    training_contract: Mapping[str, object] | None,
    metadata: Mapping[str, object] | None = None,
) -> Path | None:
    """Export `exports/best_actor/` from the selected best checkpoint when metadata is available."""
    if checkpoint_path is None or output_dir is None or training_contract is None:
        return None
    schema = _schema_from_contract(training_contract)
    normalization = _normalization_from_contract(training_contract)
    export_dir = Path(output_dir) / "exports" / "best_actor"
    export_metadata = {
        "contract_version": TRAINING_READINESS_CONTRACT_VERSION,
        "export_kind": "best_actor",
    }
    if metadata is not None:
        export_metadata.update(dict(metadata))
    export_actor_from_training_checkpoint(
        checkpoint_path,
        export_dir,
        schema=schema,
        normalization=normalization,
        metadata=json_safe(export_metadata),
    )
    return export_dir


def _schema_from_contract(contract: Mapping[str, object]) -> ObservationSchema:
    raw = contract.get("observation_schema")
    if not isinstance(raw, Mapping):
        raise ValueError("training contract is missing observation_schema metadata")
    return ObservationSchema(
        n_active_total=int(raw["n_active_total"]),
        n_angles=int(raw["n_angles"]),
        version=str(raw.get("schema_version", "v1")),
        target_preview_steps=int(raw.get("target_preview_steps", 0)),
    )


def _normalization_from_contract(contract: Mapping[str, object]) -> dict[str, object]:
    raw = contract.get("normalization")
    if not isinstance(raw, Mapping):
        raise ValueError("training contract is missing normalization metadata")
    return dict(raw)


__all__ = ["export_best_actor_artifact"]
