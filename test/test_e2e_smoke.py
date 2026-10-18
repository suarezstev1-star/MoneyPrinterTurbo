"""End-to-end render smoke test: local clips, no voice, no network.

Slow (real FFmpeg encode), so it only runs when ``RUN_SLOW=1``:

    RUN_SLOW=1 python -m pytest -q test/test_e2e_smoke.py
"""

import io
import json
import os
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1 to run slow tests"
    ),
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required"),
]


def _make_clip(path: Path, color: str, seconds: int = 2) -> None:
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c={color}:s=540x960:r=15:d={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


def _duration(path: Path) -> float:
    completed = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    stamp = [
        line for line in completed.stderr.splitlines() if "time=" in line
    ][-1].split("time=")[1].split()[0]
    hours, minutes, seconds = stamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def test_local_clips_no_voice_renders_short_video(tmp_path):
    import cli
    from app.config import config

    clips = tmp_path / "clips"
    clips.mkdir()
    _make_clip(clips / "a.mp4", "red")
    _make_clip(clips / "b.mp4", "blue")
    storage = tmp_path / "storage"
    storage.mkdir()

    argv = [
        "--video-script", "A tiny offline smoke test.",
        "--video-source", "local",
        "--video-materials", f"{clips / 'a.mp4'},{clips / 'b.mp4'}",
        "--voice-name", "no-voice",
        "--bgm-type", "none",
        "--no-subtitle-enabled",
        "--video-clip-duration", "2",
        "--video-aspect", "9:16",
        "--n-threads", "2",
    ]
    output = io.StringIO()
    with (
        patch("app.utils.utils.storage_dir", side_effect=lambda sub="", create=False: _storage(storage, sub, create)),
        patch.dict(config.ui, {}, clear=True),
        redirect_stdout(output),
    ):
        exit_code = cli.run_cli(argv)

    assert exit_code == 0, output.getvalue()
    summary = json.loads(output.getvalue().strip().splitlines()[-1])
    videos = summary.get("videos") or summary.get("result", {}).get("videos")
    assert videos, summary
    rendered = Path(videos[0])
    assert rendered.is_file() and rendered.stat().st_size > 0
    assert 1.0 <= _duration(rendered) <= 6.0


def _storage(root: Path, sub: str, create: bool) -> str:
    path = root / sub if sub else root
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return str(path)
