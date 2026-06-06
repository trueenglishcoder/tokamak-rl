from __future__ import annotations

from io import StringIO

from tokamak_rl.training.progress import TrainingProgressBar


def test_training_progress_bar_writes_final_status() -> None:
    stream = StringIO()
    bar = TrainingProgressBar(total_steps=10, label="test", stream=stream, min_interval_s=0.0)

    bar.update(5, status={"updates": 2, "critic": 1.25})
    bar.update(10, status={"updates": 3, "actor": 0.5})
    bar.close(status={"updates": 3, "actor": 0.5})

    text = stream.getvalue()
    assert "test" in text
    assert "10/10" in text
    assert "100.0%" in text
    assert "updates=3" in text
    assert "actor=0.5" in text


def test_training_progress_bar_disabled_is_silent() -> None:
    stream = StringIO()
    bar = TrainingProgressBar(total_steps=10, label="test", enabled=False, stream=stream)

    bar.update(10, status={"updates": 1}, force=True)
    bar.close(status={"updates": 1})

    assert stream.getvalue() == ""
