from __future__ import annotations

import pytest
import torch

from tokamak_rl.training import resolve_training_device


def test_resolve_training_device_cpu_records_metadata() -> None:
    device, metadata = resolve_training_device("cpu")

    assert device.type == "cpu"
    assert metadata.requested == "cpu"
    assert metadata.actual == "cpu"
    assert metadata.to_metadata()["torch_version"] == torch.__version__


def test_resolve_training_device_auto_is_available_device() -> None:
    device, metadata = resolve_training_device("auto")

    expected = "cuda" if torch.cuda.is_available() else "cpu"

    assert device.type == expected
    assert metadata.actual == expected
    assert metadata.requested == "auto"


def test_resolve_training_device_cuda_fails_clearly_without_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_training_device("cuda")


def test_resolve_training_device_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="cpu, cuda, auto"):
        resolve_training_device("gpu")
