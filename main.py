import ipaddress
import sys

import uvicorn
from loguru import logger

from app.config import config


def is_loopback_host(host: str) -> bool:
    """判断监听地址是否只允许本机访问；0.0.0.0、:: 和局域网地址均视为远程可达。"""
    host = str(host or "").strip().strip("[]")
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def unauthenticated_remote_api_error() -> str | None:
    """
    未配置 API Key 却监听非本机地址时返回拒绝启动的原因。

    免认证 API 可以创建付费任务、删除任务并触发自动发布；暴露到网络后任何
    可达的客户端都能使用。只有显式声明网络可信时才允许这种组合。
    """
    if is_loopback_host(config.listen_host):
        return None
    if config.app.get("api_key", "") not in (None, ""):
        return None
    if config.allow_unauthenticated_remote_api:
        return None
    return (
        f"refusing to start: listen_host {config.listen_host!r} is reachable from "
        "the network but app.api_key is empty. Set app.api_key in config.toml, "
        'use listen_host = "127.0.0.1", or set allow_unauthenticated_remote_api = '
        "true only if the network is trusted."
    )


if __name__ == "__main__":
    startup_error = unauthenticated_remote_api_error()
    if startup_error:
        logger.error(startup_error)
        sys.exit(1)
    logger.info(
        "start server, docs: http://127.0.0.1:" + str(config.listen_port) + "/docs"
    )
    # FFmpeg 探测已经移到 app/services/task.py 的共享任务流水线里，这样
    # API、CLI 和 WebUI 三条路径都能统一覆盖，这里不再单独检查。
    uvicorn.run(
        app="app.asgi:app",
        host=config.listen_host,
        port=config.listen_port,
        reload=config.reload_debug,
        log_level="warning",
    )
