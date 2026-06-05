from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    requested: str
    actual: str
    cuda_available: bool
    gpu_name: str | None
    torch_version: str
    cuda_version: str | None

    def to_metadata(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "actual": self.actual,
            "cuda_available": bool(self.cuda_available),
            "gpu_name": self.gpu_name,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
        }


def resolve_training_device(requested: str) -> tuple[torch.device, DeviceSelection]:
    value = str(requested).strip().lower()
    if value not in {"cpu", "cuda", "auto"}:
        raise ValueError("training device must be one of: cpu, cuda, auto")
    cuda_available = bool(torch.cuda.is_available())
    if value == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested for training, but torch.cuda.is_available() is false")
    actual = "cuda" if (value == "cuda" or (value == "auto" and cuda_available)) else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if actual == "cuda" and cuda_available else None
    metadata = DeviceSelection(
        requested=value,
        actual=actual,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        torch_version=str(torch.__version__),
        cuda_version=None if torch.version.cuda is None else str(torch.version.cuda),
    )
    return torch.device(actual), metadata


__all__ = ["DeviceSelection", "resolve_training_device"]
