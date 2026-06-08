from __future__ import annotations

import json
from pathlib import Path

from tokamak_rl.training.profile_runner import main as profile_main


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_profile_runner_writes_cprofile_artifacts_for_reward_search_dry_run(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profile"
    output_dir = tmp_path / "reward_search"
    rc = profile_main(
        [
            "--kind",
            "reward_search",
            "--profile-dir",
            str(profile_dir),
            "--top",
            "10",
            "--",
            "--config",
            str(REPO_ROOT / "configs/experiments/t15md_training_real_replay_like_smoke.yaml"),
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--max-candidates",
            "1",
        ]
    )

    assert rc == 0
    summary_path = profile_dir / "profile_summary.json"
    assert summary_path.exists()
    assert (profile_dir / "cprofile.pstats").exists()
    assert (profile_dir / "cprofile_cumulative.txt").exists()
    assert (profile_dir / "cprofile_tottime.txt").exists()
    assert (profile_dir / "cprofile_top.json").exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["kind"] == "reward_search"
    assert summary["exit_code"] == 0
    assert summary["elapsed_s"] >= 0.0
    top_rows = json.loads((profile_dir / "cprofile_top.json").read_text(encoding="utf-8"))
    assert top_rows
    assert "function" in top_rows[0]
