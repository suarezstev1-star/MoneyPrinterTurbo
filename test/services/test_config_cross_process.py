import errno
import multiprocessing
import os
import stat
import threading
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

from app.config import config


@pytest.fixture
def isolated_config(tmp_path):
    """Point the config module at a temp file and restore global state after."""
    config_path = tmp_path / "config.toml"
    config_path.write_text('[app]\nexisting = "keep"\n', encoding="utf-8")
    saved_cfg = dict(config._cfg)
    saved_app = dict(config.app)
    saved_baseline = config._baseline
    with (
        patch.object(config, "root_dir", str(tmp_path)),
        patch.object(config, "config_file", str(config_path)),
    ):
        config._baseline = {"app": dict(config.app)}
        yield config_path
    config.app.clear()
    config.app.update(saved_app)
    config._cfg.clear()
    config._cfg.update(saved_cfg)
    config._baseline = saved_baseline


def _child_save(config_path, root, key, value, barrier):
    config.root_dir = root
    config.config_file = config_path
    config.app[key] = value
    barrier.wait()
    for _ in range(5):
        config.app[f"{key}_again"] = value
        config.save_config()


def test_concurrent_processes_do_not_lose_keys(isolated_config, tmp_path):
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(4)
    procs = [
        ctx.Process(
            target=_child_save,
            args=(
                str(isolated_config),
                str(tmp_path),
                f"proc_key_{i}",
                f"v{i}",
                barrier,
            ),
        )
        for i in range(4)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(60)
        assert proc.exitcode == 0

    saved = tomllib.loads(isolated_config.read_text(encoding="utf-8"))["app"]
    for i in range(4):
        assert saved[f"proc_key_{i}"] == f"v{i}"
        assert saved[f"proc_key_{i}_again"] == f"v{i}"
    assert saved["existing"] == "keep"


def test_save_merges_instead_of_overwriting_foreign_changes(isolated_config):
    # Another process rotates a token on disk after this process loaded.
    isolated_config.write_text(
        '[app]\nexisting = "keep"\nkimi_refresh = "rotated-by-peer"\n',
        encoding="utf-8",
    )
    config.app["mine"] = "1"
    config.save_config()

    saved = tomllib.loads(isolated_config.read_text(encoding="utf-8"))["app"]
    assert saved["kimi_refresh"] == "rotated-by-peer"
    assert saved["mine"] == "1"


def test_deletes_are_propagated_but_untouched_peer_keys_survive(isolated_config):
    config.app["temp_key"] = "x"
    config.save_config()
    isolated_config.write_text(
        '[app]\nexisting = "keep"\ntemp_key = "x"\npeer_only = "p"\n',
        encoding="utf-8",
    )
    del config.app["temp_key"]
    config.save_config()

    saved = tomllib.loads(isolated_config.read_text(encoding="utf-8"))["app"]
    assert "temp_key" not in saved
    assert saved["peer_only"] == "p"


def test_force_save_overwrites_disk_value_for_rotated_token(isolated_config):
    config.app["token"] = "old"
    config.save_config()
    isolated_config.write_text('[app]\ntoken = "stale-peer"\n', encoding="utf-8")
    # Unchanged in memory: a normal save keeps the peer's value...
    config.save_config()
    assert tomllib.loads(isolated_config.read_text())["app"]["token"] == "stale-peer"
    # ...a forced durable save writes this process's value.
    assert config.try_save_config(force=True) is True
    assert tomllib.loads(isolated_config.read_text())["app"]["token"] == "old"


def test_threads_writing_distinct_keys_keep_all(isolated_config):
    def work(i):
        config.app[f"thread_{i}"] = i
        config.save_config()

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    saved = tomllib.loads(isolated_config.read_text(encoding="utf-8"))["app"]
    assert all(saved[f"thread_{i}"] == i for i in range(8))


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_saved_config_is_owner_only(isolated_config):
    isolated_config.chmod(0o644)
    config.app["secret"] = "s"
    config.save_config()
    assert stat.S_IMODE(isolated_config.stat().st_mode) == 0o600


def test_chmod_failure_is_ignored(isolated_config):
    config.app["secret"] = "s"
    with patch.object(config.os, "chmod", side_effect=PermissionError):
        config.save_config()
    assert tomllib.loads(isolated_config.read_text())["app"]["secret"] == "s"


def test_in_place_fallback_keeps_other_keys(isolated_config):
    real_replace = os.replace

    def busy(src, dst):
        if str(dst) == str(isolated_config):
            raise OSError(errno.EBUSY, "busy")
        return real_replace(src, dst)

    isolated_config.write_text(
        '[app]\nexisting = "keep"\npeer = "yes"\n' + "# pad\n" * 50,
        encoding="utf-8",
    )
    config.app["mine"] = "1"
    with patch.object(config.os, "replace", side_effect=busy):
        config.save_config()
    saved = tomllib.loads(isolated_config.read_text(encoding="utf-8"))["app"]
    assert saved["peer"] == "yes" and saved["mine"] == "1"
    assert list(Path(isolated_config.parent).glob(".config-*.toml.tmp")) == []


def test_reader_retries_while_writer_holds_partial_content(isolated_config):
    calls = {"n": 0}
    real_load = config.toml.load

    def flaky(path):
        calls["n"] += 1
        if calls["n"] < 3:
            raise config.toml.TomlDecodeError("partial", "x", 0)
        return real_load(path)

    with patch.object(config.toml, "load", side_effect=flaky):
        loaded = config._load_toml_config(str(isolated_config))
    assert loaded["app"]["existing"] == "keep"
    assert calls["n"] == 3


def test_windows_never_locks_config_file_and_readers_take_no_lock(isolated_config):
    locked = []
    real_locked_path = config._locked_path

    def spy(path, **kwargs):
        locked.append((os.path.basename(path), kwargs.get("exclusive")))
        return real_locked_path(path, **kwargs)

    with (
        patch.object(config, "_IS_WINDOWS", True),
        patch.object(config, "_locked_path", side_effect=spy),
    ):
        # Reader: no lock of any kind.
        assert config._load_toml_config(str(isolated_config))["app"]["existing"] == "keep"
        assert locked == []
        # Writer: only the sidecar is locked, never config.toml.
        config.app["win"] = "1"
        config.save_config()
    assert locked == [(".config.toml.lock", True)]
    assert tomllib.loads(isolated_config.read_text())["app"]["win"] == "1"


def test_transient_permission_error_on_load_is_retried(isolated_config):
    calls = {"n": 0}
    real_load = config.toml.load

    def flaky(path):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "Permission denied")
        return real_load(path)

    with patch.object(config.toml, "load", side_effect=flaky):
        loaded = config._load_toml_config(str(isolated_config))
    assert loaded["app"]["existing"] == "keep"
    assert calls["n"] == 3


def test_save_retries_transient_permission_error_on_replace(isolated_config):
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "Permission denied")
        return real_replace(src, dst)

    config.app["retry"] = "1"
    with patch.object(config.os, "replace", side_effect=flaky):
        config.save_config()
    assert tomllib.loads(isolated_config.read_text())["app"]["retry"] == "1"
