"""Batch manifests may only reference files inside the manifest/storage dirs by default."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cli
from app.config import config as app_config


@pytest.fixture(autouse=True)
def _isolated_ui_config():
    with patch.dict(app_config.ui, {}, clear=True):
        yield


def _write_manifest(directory: Path, entry: dict) -> Path:
    manifest = directory / "tasks.json"
    manifest.write_text(json.dumps([entry]), encoding="utf-8")
    return manifest


def _args(manifest: Path, *extra: str):
    return cli.parse_args(
        ["--batch-file", str(manifest), "--stop-at", "audio", *extra]
    )


def test_help_documents_flag_and_default_policy(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.parse_args(["--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--allow-outside-manifest-dir" in help_text
    assert "manifest" in help_text and "storage" in help_text


def test_flag_defaults_to_false_and_can_be_enabled(tmp_path):
    manifest = _write_manifest(tmp_path, {"video_subject": "x"})
    assert _args(manifest).allow_outside_manifest_dir is False
    assert _args(manifest, "--allow-outside-manifest-dir").allow_outside_manifest_dir


def test_audio_inside_manifest_directory_is_allowed(tmp_path):
    (tmp_path / "sub").mkdir()
    audio = tmp_path / "sub" / "voice.mp3"
    audio.write_bytes(b"audio")
    manifest = _write_manifest(
        tmp_path, {"video_subject": "x", "custom_audio_file": "sub/voice.mp3"}
    )
    (task,) = cli._build_batch_tasks(_args(manifest))
    assert task.custom_audio_file == str(audio.resolve())


def test_parent_traversal_is_rejected_by_default(tmp_path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (tmp_path / "secret.mp3").write_bytes(b"audio")
    manifest = _write_manifest(
        manifest_dir, {"video_subject": "x", "custom_audio_file": "../secret.mp3"}
    )
    with pytest.raises(ValueError, match="allow-outside-manifest-dir"):
        cli._build_batch_tasks(_args(manifest))


def test_absolute_path_outside_is_rejected_by_default(tmp_path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    outside = tmp_path / "secret.mp3"
    outside.write_bytes(b"audio")
    manifest = _write_manifest(
        manifest_dir, {"video_subject": "x", "custom_audio_file": str(outside)}
    )
    with pytest.raises(ValueError, match="invalid batch task 1"):
        cli._build_batch_tasks(_args(manifest))


def test_symlink_escape_is_rejected_by_default(tmp_path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    outside = tmp_path / "secret.mp3"
    outside.write_bytes(b"audio")
    (manifest_dir / "link.mp3").symlink_to(outside)
    manifest = _write_manifest(
        manifest_dir, {"video_subject": "x", "custom_audio_file": "link.mp3"}
    )
    with pytest.raises(ValueError, match="allow-outside-manifest-dir"):
        cli._build_batch_tasks(_args(manifest))


def test_flag_permits_outside_paths(tmp_path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    outside = tmp_path / "secret.mp3"
    outside.write_bytes(b"audio")
    manifest = _write_manifest(
        manifest_dir, {"video_subject": "x", "custom_audio_file": "../secret.mp3"}
    )
    (task,) = cli._build_batch_tasks(
        _args(manifest, "--allow-outside-manifest-dir")
    )
    assert task.custom_audio_file == str(outside.resolve())


def test_project_storage_directory_is_allowed(tmp_path):
    manifest_dir = tmp_path / "manifests"
    storage = tmp_path / "storage"
    manifest_dir.mkdir()
    storage.mkdir()
    audio = storage / "voice.mp3"
    audio.write_bytes(b"audio")
    manifest = _write_manifest(
        manifest_dir, {"video_subject": "x", "custom_audio_file": str(audio)}
    )
    with patch("app.utils.utils.storage_dir", return_value=str(storage)):
        (task,) = cli._build_batch_tasks(_args(manifest))
    assert task.custom_audio_file == str(audio)


def test_local_material_outside_manifest_directory_is_rejected(tmp_path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (tmp_path / "clip.mp4").write_bytes(b"video")
    manifest = _write_manifest(
        manifest_dir,
        {
            "video_subject": "x",
            "video_source": "local",
            "video_materials": [
                {"provider": "local", "url": "../clip.mp4", "duration": 0}
            ],
        },
    )
    with pytest.raises(ValueError, match="allow-outside-manifest-dir"):
        cli._build_batch_tasks(_args(manifest))
