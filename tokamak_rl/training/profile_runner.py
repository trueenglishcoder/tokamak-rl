from __future__ import annotations

import argparse
import cProfile
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import pstats
import sys
import time
from typing import Callable

from tokamak_rl.training.diagnostics import json_safe


PROFILE_SORT_CHOICES = ("cumulative", "tottime", "time", "calls")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile tokamak-rl train or reward-search runs.")
    parser.add_argument("--kind", choices=("train", "reward_search"), required=True, help="Entrypoint to profile.")
    parser.add_argument("--profile-dir", type=Path, required=True, help="Directory for profile artifacts.")
    parser.add_argument("--sort", choices=PROFILE_SORT_CHOICES, default="cumulative", help="cProfile sort key.")
    parser.add_argument("--top", type=int, default=80, help="Number of cProfile rows in text/JSON summaries.")
    parser.add_argument("--torch-profiler", action="store_true", help="Also write a PyTorch CPU/CUDA chrome trace.")
    parser.add_argument("--torch-record-shapes", action="store_true", help="Record tensor shapes in PyTorch profiler trace.")
    parser.add_argument("--torch-profile-memory", action="store_true", help="Record memory in PyTorch profiler trace.")
    parser.add_argument("entrypoint_args", nargs=argparse.REMAINDER, help="Arguments passed after -- to the profiled entrypoint.")
    args = parser.parse_args(argv)

    entrypoint_args = list(args.entrypoint_args)
    if entrypoint_args and entrypoint_args[0] == "--":
        entrypoint_args = entrypoint_args[1:]
    profile_dir = Path(args.profile_dir).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)

    _reset_simulator_profilers()
    _reset_cuda_peak_memory()

    profile = cProfile.Profile()
    started_wall = time.perf_counter()
    started_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    exit_code = 0
    error: str | None = None
    try:
        runner = _runner_for_kind(str(args.kind))
        with _optional_torch_profiler(profile_dir, enabled=bool(args.torch_profiler), record_shapes=bool(args.torch_record_shapes), profile_memory=bool(args.torch_profile_memory)):
            profile.enable()
            exit_code = int(runner(entrypoint_args))
            _synchronize_cuda()
            profile.disable()
    except BaseException as exc:  # noqa: BLE001 - profiling must still write artifacts on failure.
        profile.disable()
        _synchronize_cuda()
        exit_code = 1
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.perf_counter() - started_wall
        _write_cprofile_artifacts(profile, profile_dir=profile_dir, sort=str(args.sort), top=int(args.top))
        summary = {
            "kind": str(args.kind),
            "entrypoint_args": entrypoint_args,
            "started_at": started_time,
            "elapsed_s": float(elapsed),
            "exit_code": int(exit_code),
            "error": error,
            "python": _python_metadata(),
            "cuda": _cuda_metadata(),
            "simulator_profiling": _simulator_profiling_snapshot(),
            "artifacts": {
                "cprofile_stats": str(profile_dir / "cprofile.pstats"),
                "cprofile_cumulative": str(profile_dir / "cprofile_cumulative.txt"),
                "cprofile_tottime": str(profile_dir / "cprofile_tottime.txt"),
                "cprofile_top_json": str(profile_dir / "cprofile_top.json"),
                "torch_trace": str(profile_dir / "torch_trace.json") if bool(args.torch_profiler) else None,
            },
        }
        (profile_dir / "profile_summary.json").write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    return int(exit_code)


def _runner_for_kind(kind: str) -> Callable[[list[str]], int]:
    if kind == "train":
        from tokamak_rl.training.cli import main as train_main

        return train_main
    if kind == "reward_search":
        from tokamak_rl.training.reward_search import main as reward_search_main

        return reward_search_main
    raise ValueError(f"unknown profiling kind: {kind}")


def _write_cprofile_artifacts(profile: cProfile.Profile, *, profile_dir: Path, sort: str, top: int) -> None:
    stats_path = profile_dir / "cprofile.pstats"
    profile.dump_stats(str(stats_path))
    for sort_key, filename in (("cumulative", "cprofile_cumulative.txt"), ("tottime", "cprofile_tottime.txt"), (sort, "cprofile_selected.txt")):
        stream = io.StringIO()
        stats = pstats.Stats(profile, stream=stream).strip_dirs().sort_stats(sort_key)
        stats.print_stats(max(int(top), 1))
        (profile_dir / filename).write_text(stream.getvalue(), encoding="utf-8")
    top_rows = _profile_top_rows(profile, sort=sort, top=top)
    (profile_dir / "cprofile_top.json").write_text(json.dumps(top_rows, indent=2, sort_keys=True), encoding="utf-8")


def _profile_top_rows(profile: cProfile.Profile, *, sort: str, top: int) -> list[dict[str, object]]:
    stats = pstats.Stats(profile).strip_dirs()
    rows: list[dict[str, object]] = []
    for func, values in stats.stats.items():
        cc, nc, tt, ct, _callers = values
        file_name, line_no, func_name = func
        rows.append(
            {
                "file": str(file_name),
                "line": int(line_no),
                "function": str(func_name),
                "primitive_calls": int(cc),
                "total_calls": int(nc),
                "tottime_s": float(tt),
                "cumulative_s": float(ct),
            }
        )
    key = "tottime_s" if sort in {"tottime", "time"} else ("total_calls" if sort == "calls" else "cumulative_s")
    rows.sort(key=lambda row: float(row[key]), reverse=True)
    return rows[: max(int(top), 1)]


@contextlib.contextmanager
def _optional_torch_profiler(profile_dir: Path, *, enabled: bool, record_shapes: bool, profile_memory: bool):
    if not enabled:
        yield None
        return
    try:
        import torch
    except Exception:
        yield None
        return
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=False,
    ) as prof:
        yield prof
        _synchronize_cuda()
        prof.export_chrome_trace(str(profile_dir / "torch_trace.json"))
        (profile_dir / "torch_key_averages.txt").write_text(
            prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=80),
            encoding="utf-8",
        )


def _reset_simulator_profilers() -> None:
    try:
        from tokamak_control.core.gpu_plasma_model import configure_gpu_plasma_model_profiling
        from tokamak_control.core.plasma_model import configure_plasma_model_profiling
        from tokamak_control.geometry.boundary import configure_boundary_profiling

        configure_plasma_model_profiling(enabled=True, summary_every=0, reset=True)
        configure_gpu_plasma_model_profiling(enabled=True, summary_every=0, reset=True)
        configure_boundary_profiling(enabled=True, summary_every=0, reset=True)
    except Exception:
        return


def _simulator_profiling_snapshot() -> dict[str, object]:
    try:
        from tokamak_control.core.gpu_plasma_model import gpu_plasma_model_profiling_snapshot
        from tokamak_control.core.plasma_model import plasma_model_profiling_snapshot
        from tokamak_control.geometry.boundary import boundary_profiling_snapshot
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "plasma_model": plasma_model_profiling_snapshot(),
        "gpu_plasma_model": gpu_plasma_model_profiling_snapshot(),
        "boundary": boundary_profiling_snapshot(),
    }


def _python_metadata() -> dict[str, object]:
    return {
        "executable": sys.executable,
        "version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
    }


def _cuda_metadata() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:
        return {"torch_available": False, "error": f"{type(exc).__name__}: {exc}"}
    data: dict[str, object] = {
        "torch_available": True,
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        current = int(torch.cuda.current_device())
        data.update(
            {
                "current_device": current,
                "device_name": str(torch.cuda.get_device_name(current)),
                "memory_allocated_bytes": int(torch.cuda.memory_allocated(current)),
                "memory_reserved_bytes": int(torch.cuda.memory_reserved(current)),
                "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(current)),
                "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(current)),
            }
        )
    return data


def _reset_cuda_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


if __name__ == "__main__":  # pragma: no cover - exercised through scripts/profile_run.py.
    raise SystemExit(main())
