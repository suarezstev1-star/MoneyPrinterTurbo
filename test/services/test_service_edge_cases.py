"""Edge-case behaviour tests for small service modules (no network, no FFmpeg)."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from app.config import config
from app.services import bgm, cache_manager, muapi, task_artifacts

# ---------------------------------------------------------------- muapi config


def test_muapi_api_key_prefers_settings_then_environment(monkeypatch):
    monkeypatch.setenv("MUAPI_API_KEY", " env-key ")
    assert muapi.get_api_key({"muapi_api_key": "  cfg-key "}) == "cfg-key"
    assert muapi.get_api_key({"muapi_api_key": ""}) == "env-key"
    monkeypatch.delenv("MUAPI_API_KEY")
    assert muapi.get_api_key({}) == ""
    assert muapi.is_enabled({"muapi_api_key": "x"}) is True
    assert muapi.is_enabled({}) is False


@pytest.mark.parametrize("bad", ["ftp://host", "not a url", "/relative"])
def test_muapi_base_url_must_be_absolute_http(bad):
    with patch.dict(config.app, {"muapi_base_url": bad}):
        with pytest.raises(muapi.MuAPIError, match="HTTP"):
            muapi._base_url()


def test_muapi_base_url_strips_trailing_slash():
    with patch.dict(config.app, {"muapi_base_url": " https://api.example.test/ "}):
        assert muapi._base_url() == "https://api.example.test"


@pytest.mark.parametrize(
    "bad", ["https://evil.test/x", "path?x=1", "path#frag", "//"]
)
def test_muapi_endpoint_rejects_urls_and_query(bad):
    with patch.dict(config.app, {"muapi_video_endpoint": bad}):
        with pytest.raises(muapi.MuAPIError, match="relative endpoint"):
            muapi._endpoint()


def test_muapi_endpoint_strips_slashes():
    with patch.dict(config.app, {"muapi_video_endpoint": "/v1/gen/"}):
        assert muapi._endpoint() == "v1/gen"


def test_muapi_bounded_float_falls_back_and_clamps():
    with patch.dict(config.app, {"k": "nan"}):
        assert muapi._bounded_float("k", 2.0, 1.0, 5.0) == 2.0
    with patch.dict(config.app, {"k": "abc"}):
        assert muapi._bounded_float("k", 2.0, 1.0, 5.0) == 2.0
    with patch.dict(config.app, {"k": "100"}):
        assert muapi._bounded_float("k", 2.0, 1.0, 5.0) == 5.0
    with patch.dict(config.app, {"k": "-3"}):
        assert muapi._bounded_float("k", 2.0, 1.0, 5.0) == 1.0


def test_muapi_duration_bounds_ignore_invalid_and_keep_max_at_least_min():
    with patch.dict(
        config.app, {"muapi_min_duration": "9", "muapi_max_duration": "3"}
    ):
        low, high = muapi._duration_bounds()
        assert (low, high) == (9, 9)
    with patch.dict(
        config.app, {"muapi_min_duration": "oops", "muapi_max_duration": "0"}
    ):
        low, high = muapi._duration_bounds()
        assert 1 <= low <= high


@pytest.mark.parametrize(
    ("value", "expected"),
    [("false", False), ("OFF", False), ("0", False), ("", False), ("yes", True), (0, False), (1, True)],
)
def test_muapi_config_bool(value, expected):
    with patch.dict(config.app, {"flag": value}):
        assert muapi._config_bool("flag", True) is expected


def test_muapi_redacts_key_and_proxy_credentials():
    with patch.dict(config.proxy, {"https": "http://user:pw@proxy.test:3128"}, clear=True):
        text = muapi._redact_secret(
            "bad key s3cr+t and http://user:pw@proxy.test:3128 failed", "s3cr+t"
        )
    assert "s3cr+t" not in text and "pw@proxy" not in text
    assert text.count("***") == 2
    with patch.dict(config.proxy, {}, clear=True):
        assert len(muapi._redact_secret("x" * 5000, "")) == muapi.MAX_ERROR_TEXT_LENGTH


def _response(status, payload=None, raises=False):
    def json_():
        if raises:
            raise ValueError("not json")
        return payload

    return SimpleNamespace(status_code=status, json=json_)


def test_muapi_response_error_shapes_and_redaction():
    assert muapi._response_error(_response(502, raises=True), "k") == "HTTP 502"
    assert muapi._response_error(_response(500, ["list"]), "k") == "HTTP 500"
    assert (
        muapi._response_error(_response(400, {"error": {"message": "bad k-secret"}}), "k-secret")
        == "bad ***"
    )
    assert (
        muapi._response_error(_response(422, {"detail": ["a", "b"]}), "k") == "a; b"
    )
    assert muapi._response_error(_response(418, {}), "k") == "HTTP 418"


def test_muapi_retryable_classification():
    assert muapi._is_retryable_error(requests.exceptions.Timeout())
    assert muapi._is_retryable_error(requests.exceptions.ConnectionError())
    retry_status = next(iter(muapi.RETRYABLE_STATUS_CODES))
    error = requests.exceptions.HTTPError(response=_response(retry_status))
    assert muapi._is_retryable_error(error)
    assert not muapi._is_retryable_error(
        requests.exceptions.HTTPError(response=_response(400))
    )
    assert not muapi._is_retryable_error(ValueError("x"))


def test_muapi_generate_requires_key_and_term():
    with patch.dict(config.app, {"muapi_api_key": ""}), patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MUAPI_API_KEY", None)
        with pytest.raises(muapi.MuAPIError, match="API key"):
            muapi.generate_videos("cats", 3)
    with patch.dict(config.app, {"muapi_api_key": "k"}):
        with pytest.raises(muapi.MuAPIError, match="empty"):
            muapi.generate_videos("   ", 3)
        with pytest.raises(muapi.MuAPIError, match="duration"):
            muapi.generate_videos("cats", "abc")


# ---------------------------------------------------------------- cache_manager


def test_cache_scan_missing_directory_is_empty(tmp_path):
    with patch.object(cache_manager.utils, "storage_dir", return_value=str(tmp_path / "gone")):
        stats = cache_manager.get_video_cache_stats()
    assert stats.file_count == 0 and stats.total_size == 0


def test_cache_scan_os_error_is_reported_as_empty_not_raised(tmp_path):
    with (
        patch.object(cache_manager.utils, "storage_dir", return_value=str(tmp_path)),
        patch.object(cache_manager.os, "scandir", side_effect=PermissionError("denied")),
    ):
        stats = cache_manager.get_video_cache_stats()
    assert stats.file_count == 0


def test_cache_clean_skips_files_that_vanish_between_scan_and_delete(tmp_path):
    victim = tmp_path / f"vid-{'a' * 32}.mp4"
    victim.write_bytes(b"x" * 10)
    real_stat = os.stat
    calls = {"n": 0}

    def flaky_stat(path, *args, **kwargs):
        # Scan uses DirEntry.stat; the deletion guard re-stats by path.
        if str(path) == str(victim):
            calls["n"] += 1
            victim.unlink(missing_ok=True)
            raise FileNotFoundError(path)
        return real_stat(path, *args, **kwargs)

    with (
        patch.object(cache_manager.utils, "storage_dir", return_value=str(tmp_path)),
        patch.object(cache_manager.os, "stat", side_effect=flaky_stat),
    ):
        result = cache_manager.clean_video_cache()
    assert calls["n"] == 1
    assert result.deleted_count == 0 and result.failed_count == 0
    assert not victim.exists()


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "7"])
def test_cache_rejects_invalid_max_age(bad):
    with pytest.raises(ValueError):
        cache_manager.get_video_cache_stats(max_age_days=bad)


# ---------------------------------------------------------------- task_artifacts


def test_patch_script_data_merges_and_preserves_existing_fields(tmp_path):
    with patch.object(task_artifacts, "_script_file", lambda task_id: tmp_path / f"{task_id}.json"):
        task_artifacts.write_script_data("t1", {"script": "hi", "terms": ["a"]})
        assert task_artifacts.patch_script_data("t1", materials=["m"]) is True
        data = (tmp_path / "t1.json").read_text(encoding="utf-8")
    import json

    assert json.loads(data) == {"script": "hi", "terms": ["a"], "materials": ["m"]}


def test_patch_script_data_missing_file_returns_false(tmp_path):
    with patch.object(task_artifacts, "_script_file", lambda task_id: tmp_path / f"{task_id}.json"):
        assert task_artifacts.patch_script_data("nope", x=1) is False
        assert list(tmp_path.iterdir()) == []


def test_patch_script_data_degrades_on_corrupt_or_non_object_json(tmp_path):
    with patch.object(task_artifacts, "_script_file", lambda task_id: tmp_path / f"{task_id}.json"):
        (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
        (tmp_path / "list.json").write_text("[1, 2]", encoding="utf-8")
        assert task_artifacts.patch_script_data("bad", x=1) is False
        assert task_artifacts.patch_script_data("list", x=1) is False
        assert (tmp_path / "list.json").read_text(encoding="utf-8") == "[1, 2]"


# ---------------------------------------------------------------- bgm validation


def test_bgm_validation_rejects_decoder_error_even_when_exit_code_is_zero():
    """FFmpeg 6.1 logs decoder failures but can still exit 0 under -xerror."""
    completed = SimpleNamespace(
        returncode=0,
        stdout=b"\x00\x01" * 100,
        stderr=b"[aist#0:0/mp3 @ 0x1] Decoder thread returned error: Invalid data\n",
    )
    with patch.object(bgm.subprocess, "run", return_value=completed):
        with pytest.raises(bgm.BgmUploadError, match="decodable"):
            bgm._validate_audio("/tmp/x.mp3")


def test_bgm_validation_accepts_short_clip_with_decoded_samples():
    completed = SimpleNamespace(returncode=0, stdout=b"\x00\x00" * 4, stderr=b"")
    with patch.object(bgm.subprocess, "run", return_value=completed):
        bgm._validate_audio("/tmp/x.wav")


def test_bgm_validation_rejects_empty_decode():
    completed = SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
    with patch.object(bgm.subprocess, "run", return_value=completed):
        with pytest.raises(bgm.BgmUploadError, match="nonempty"):
            bgm._validate_audio("/tmp/x.wav")
