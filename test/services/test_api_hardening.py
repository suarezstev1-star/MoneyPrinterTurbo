import os
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import asgi
from app.config import config
from app.controllers import base
from app.models.schema import AudioRequest, SubtitleRequest, TaskVideoRequest, VideoParams
from app.services import state as sm
from app.utils import file_security, utils


def _client(**kwargs):
    return TestClient(asgi.get_application(), raise_server_exceptions=False, **kwargs)


# ---- Host header / DNS rebinding ----


@pytest.mark.parametrize(
    "host",
    ["localhost", "localhost:8080", "127.0.0.1:8080", "[::1]:8080", "192.168.1.9:8080"],
)
def test_expected_hosts_are_accepted(host):
    with patch.dict(config.app, {"api_key": ""}), _client() as client:
        assert client.get("/ping", headers={"host": host}).status_code == 200


def test_rebinding_host_is_rejected_without_key():
    with patch.dict(config.app, {"api_key": ""}), _client() as client:
        response = client.get("/ping", headers={"host": "evil.example:8080"})
    assert response.status_code == 400


def test_allowed_hosts_setting_extends_the_list():
    with patch.dict(config.app, {"api_key": "", "allowed_hosts": ["api.example.com", "*.corp.test"]}):
        assert asgi.is_host_allowed("api.example.com:443")
        assert asgi.is_host_allowed("x.corp.test")
        assert not asgi.is_host_allowed("evil.example")


def test_non_loopback_listener_with_key_allows_any_host_when_unset():
    with (
        patch.dict(config.app, {"api_key": "secret"}),
        patch.object(config, "listen_host", "0.0.0.0"),
    ):
        assert asgi.is_host_allowed("api.anything.example")
    with (
        patch.dict(config.app, {"api_key": ""}),
        patch.object(config, "listen_host", "0.0.0.0"),
    ):
        assert not asgi.is_host_allowed("api.anything.example")


# ---- Cross-post opt-in ----


def test_api_request_defaults_to_no_cross_post_but_webui_params_keep_default():
    assert TaskVideoRequest(video_subject="a").cross_post is False
    assert TaskVideoRequest(video_subject="a", cross_post=True).cross_post is True
    assert VideoParams(video_subject="a").cross_post is True


# ---- Input bounds ----


@pytest.mark.parametrize(
    "field,value",
    [
        ("video_subject", "x" * 501),
        ("video_script", "x" * 20001),
        ("video_terms", ["t"] * 51),
        ("video_terms", ["x" * 201]),
        ("video_terms", "x" * 5001),
        ("font_size", 100000),
        ("stroke_width", 1e6),
        ("voice_rate", 1000),
        ("voice_volume", -1),
        ("bgm_volume", 11),
        ("video_count", 6),
        ("custom_system_prompt", "x" * 8001),
    ],
)
def test_oversized_task_fields_are_rejected(field, value):
    payload = {"video_subject": "ok", field: value}
    with pytest.raises(ValidationError):
        TaskVideoRequest(**payload)


def test_oversized_audio_and_subtitle_requests_are_rejected():
    for model in (AudioRequest, SubtitleRequest):
        with pytest.raises(ValidationError):
            model(video_script="x" * 20001)
        with pytest.raises(ValidationError):
            model(video_script="ok", voice_rate=1000)


def test_valid_defaults_and_boundaries_still_work():
    params = TaskVideoRequest(
        video_subject="s" * 500,
        video_script="x" * 20000,
        video_terms=["t"] * 50,
        font_size=400,
        stroke_width=50,
        voice_rate=10,
        bgm_volume=0,
    )
    assert params.video_count == 1


def test_api_returns_400_for_out_of_range_values():
    with (
        patch.dict(config.app, {"api_key": ""}),
        patch("app.controllers.v1.video.task_manager.add_task") as schedule,
        _client() as client,
    ):
        response = client.post("/api/v1/videos", json={"video_subject": "x", "font_size": 10**9})
    assert response.status_code == 400
    schedule.assert_not_called()


# ---- Upload size cap ----


def test_declared_oversized_upload_is_rejected_with_413_before_parsing():
    with patch.dict(config.app, {"api_key": ""}), _client() as client:
        response = client.post(
            "/api/v1/musics",
            content=b"x",
            headers={"content-length": str(200 * 1024 * 1024), "content-type": "multipart/form-data; boundary=a"},
        )
    assert response.status_code == 413


def test_streamed_oversized_body_is_rejected_with_413():
    def chunks():
        for _ in range(8):
            yield b"x" * (1024 * 1024)

    with (
        patch.dict(config.app, {"api_key": "", "max_request_body_bytes": 2 * 1024 * 1024}),
        _client() as client,
    ):
        response = client.post(
            "/api/v1/scripts", content=chunks(), headers={"content-type": "application/json"}
        )
    assert response.status_code == 413


# ---- Upload retention / quota ----


def test_prune_removes_expired_then_oldest_over_quota(tmp_path):
    now = time.time()
    names = {}
    for index, age_days in enumerate((40, 3, 2, 1)):
        name = f"{index:032x}.mp4"
        path = tmp_path / name
        path.write_bytes(b"x" * 100)
        os.utime(path, (now - age_days * 86400, now - age_days * 86400))
        names[age_days] = name
    (tmp_path / "manual-clip.mp4").write_bytes(b"x" * 100)
    os.utime(tmp_path / "manual-clip.mp4", (now - 99 * 86400,) * 2)

    removed = file_security.prune_upload_directory(
        str(tmp_path),
        max_age_seconds=30 * 86400,
        max_total_bytes=250,
        name_pattern=file_security.UPLOADED_FILE_NAME_PATTERN,
    )

    assert removed == [names[40], names[3]]
    assert (tmp_path / "manual-clip.mp4").exists()
    assert (tmp_path / names[1]).exists()


# ---- Rate limiting ----


def test_post_rate_limit_returns_429_and_gets_are_unaffected():
    with patch.dict(config.app, {"api_key": "", "api_rate_limit_per_minute": 2}), _client() as client:
        statuses = [client.post("/api/v1/scripts", json={"video_subject": ""}).status_code for _ in range(3)]
        assert statuses[-1] == 429
        assert client.get("/api/v1/tasks").status_code == 200


def test_repeated_wrong_keys_are_throttled():
    with patch.dict(config.app, {"api_key": "right-key", "api_auth_failure_limit": 3}), _client() as client:
        codes = [client.get("/api/v1/tasks", headers={"x-api-key": f"bad{i}"}).status_code for i in range(5)]
        assert codes[:3] == [401, 401, 401]
        assert codes[3:] == [429, 429]


def test_llm_concurrency_cap_returns_429():
    import threading

    from app.controllers.v1 import llm as llm_controller

    started = threading.Event()
    release = threading.Event()

    def slow(**_):
        started.set()
        release.wait(5)
        return "script"

    with (
        patch.dict(config.app, {"api_key": "", "max_concurrent_llm_requests": 1}),
        patch("app.controllers.v1.llm.llm.generate_script", side_effect=slow),
        _client() as client,
    ):
        first = {}
        worker = threading.Thread(
            target=lambda: first.setdefault("r", client.post("/api/v1/scripts", json={"video_subject": "a"}))
        )
        worker.start()
        assert started.wait(5)
        second = client.post("/api/v1/scripts", json={"video_subject": "b"})
        release.set()
        worker.join(5)
    assert second.status_code == 429
    assert first["r"].status_code == 200
    assert llm_controller._llm_semaphore() is not None


# ---- Docs / ping ----


def test_docs_require_key_but_ping_stays_open():
    docs_client = TestClient(asgi.app, raise_server_exceptions=False)
    with patch.dict(config.app, {"api_key": "k"}), docs_client as client:
        assert client.get("/ping").status_code == 200
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code == 401
        assert client.get("/openapi.json", headers={"x-api-key": "k"}).status_code == 200


# ---- Task ownership ----


def test_tasks_are_stamped_and_hidden_from_other_keys():
    memory = sm.MemoryState()
    with (
        patch.object(sm, "state", memory),
        patch("app.controllers.v1.video.task_manager.add_task"),
        patch.dict(config.app, {"api_key": "key-a"}),
        _client() as client,
    ):
        created = client.post(
            "/api/v1/videos", json={"video_subject": "x"}, headers={"x-api-key": "key-a"}
        ).json()["data"]
        task_id = created["task_id"]
        stored_owner = memory.get_task(task_id)["owner"]
        assert stored_owner and "key-a" not in stored_owner
        listed = client.get("/api/v1/tasks", headers={"x-api-key": "key-a"}).json()["data"]
        assert listed["total"] == 1 and "owner" not in listed["tasks"][0]

    with (
        patch.object(sm, "state", memory),
        patch.dict(config.app, {"api_key": "key-b"}),
        _client() as client,
    ):
        headers = {"x-api-key": "key-b"}
        assert client.get("/api/v1/tasks", headers=headers).json()["data"]["total"] == 0
        assert client.get(f"/api/v1/tasks/{task_id}", headers=headers).status_code == 404
        assert client.delete(f"/api/v1/tasks/{task_id}", headers=headers).status_code == 404

    with patch.object(sm, "state", memory), patch.dict(config.app, {"api_key": ""}), _client() as client:
        assert client.get(f"/api/v1/tasks/{task_id}").status_code == 200


def test_fingerprint_is_none_without_key_and_not_the_key():
    with patch.dict(config.app, {"api_key": ""}):
        assert base.owner_fingerprint() is None
    with patch.dict(config.app, {"api_key": "abc123"}):
        assert "abc123" not in base.owner_fingerprint()


# ---- Error redaction ----


def test_task_errors_are_redacted_in_api_responses():
    memory = sm.MemoryState()
    memory.update_task(
        "t1",
        state=-1,
        error="HTTPSConnectionPool: https://api.example.com/v1/x?key=SUPERSECRETVALUE failed Authorization: Bearer abcdefghijklmnop",
        cross_post_error="https://user:pw@host.test/path?token=zzz",
        cross_post_results=[{"error": "sk-abcdefghijklmnopqrstuv boom", "url": "https://ok.test/a/b"}],
    )
    with patch.object(sm, "state", memory), patch.dict(config.app, {"api_key": ""}), _client() as client:
        data = client.get("/api/v1/tasks/t1").json()["data"]
    flat = str(data)
    for leaked in ("SUPERSECRETVALUE", "abcdefghijklmnop", "pw@", "zzz", "sk-abc"):
        assert leaked not in flat
    assert "api.example.com" in data["error"]
    assert data["cross_post_results"][0]["url"] == "https://ok.test/a/b"


def test_redact_sensitive_text_hides_configured_secrets():
    with patch.dict(config.app, {"openai_api_key": "my-configured-secret"}):
        assert "my-configured-secret" not in utils.redact_sensitive_text("failed with my-configured-secret")


# ---- Redis state parsing ----


@pytest.mark.parametrize(
    "stored,expected",
    [
        (b"'2026'", "2026"),
        (b"42", 42),
        (b"1.5", 1.5),
        (b"True", True),
        (b"None", None),
        (b"['a', 'b']", ["a", "b"]),
        (b"{'a': 1}", {"a": 1}),
        (b'{"a": [1, 2]}', {"a": [1, 2]}),
        (b'"quoted"', "quoted"),
        (b"plain text", "plain text"),
    ],
)
def test_redis_values_parse_to_plain_types(stored, expected):
    assert sm.RedisState._convert_to_original_type(stored) == expected


@pytest.mark.parametrize(
    "stored",
    [b"{1, 2}", b"b'bytes'", b"1+2j", b"__import__('os').system('true')", b"(lambda: 1)()", b"[" * 5000],
)
def test_redis_values_never_become_unexpected_types(stored):
    result = sm.RedisState._convert_to_original_type(stored)
    assert result == stored.decode()
