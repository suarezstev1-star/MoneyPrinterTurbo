import runpy
from pathlib import Path
from unittest.mock import patch

from app.config import config


ROOT_DIR = Path(__file__).resolve().parent.parent


def test_main_starts_uvicorn_with_runtime_config():
    """
    服务启动入口只负责把运行配置交给 Uvicorn。这里 mock 真正的服务器启动，
    既避免测试占用端口，也确认监听地址、端口和热重载配置不会在入口层丢失。
    """
    with (
        patch.object(config, "listen_host", "127.0.0.1"),
        patch.object(config, "listen_port", 8765),
        patch.object(config, "reload_debug", True),
        patch("uvicorn.run") as run_server,
    ):
        runpy.run_path(str(ROOT_DIR / "main.py"), run_name="__main__")

    run_server.assert_called_once_with(
        app="app.asgi:app",
        host="127.0.0.1",
        port=8765,
        reload=True,
        log_level="warning",
    )


def _load_main_module():
    return runpy.run_path(str(ROOT_DIR / "main.py"), run_name="main_under_test")


def test_loopback_hosts_are_local_only():
    """只有回环地址被视为本机访问，0.0.0.0、:: 和局域网地址都可从网络到达。"""
    is_loopback_host = _load_main_module()["is_loopback_host"]

    for host in ("127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.5"):
        assert is_loopback_host(host), host
    for host in ("0.0.0.0", "::", "192.168.1.10", "example.com", ""):
        assert not is_loopback_host(host), host


def test_remote_listen_without_api_key_is_refused():
    """监听网络地址但未配置 API Key 时，入口必须拒绝启动且不调用 Uvicorn。"""
    with (
        patch.object(config, "listen_host", "0.0.0.0"),
        patch.object(config, "allow_unauthenticated_remote_api", False),
        patch.dict(config.app, {"api_key": ""}),
        patch("uvicorn.run") as run_server,
    ):
        try:
            runpy.run_path(str(ROOT_DIR / "main.py"), run_name="__main__")
        except SystemExit as exc:
            assert exc.code == 1
        else:
            raise AssertionError("main.py should exit when the API is exposed")

    run_server.assert_not_called()


def test_remote_listen_is_allowed_with_api_key_or_explicit_opt_in():
    """配置 API Key 或显式声明可信网络后，远程监听可以正常启动。"""
    error_for = _load_main_module()["unauthenticated_remote_api_error"]

    with (
        patch.object(config, "listen_host", "0.0.0.0"),
        patch.object(config, "allow_unauthenticated_remote_api", False),
        patch.dict(config.app, {"api_key": "secret"}),
    ):
        assert error_for() is None

    with (
        patch.object(config, "listen_host", "0.0.0.0"),
        patch.object(config, "allow_unauthenticated_remote_api", True),
        patch.dict(config.app, {"api_key": ""}),
    ):
        assert error_for() is None

    with (
        patch.object(config, "listen_host", "127.0.0.1"),
        patch.object(config, "allow_unauthenticated_remote_api", False),
        patch.dict(config.app, {"api_key": ""}),
    ):
        assert error_for() is None
