"""Application implementation - ASGI."""

import ipaddress
import math
import os
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import config
from app.controllers import base, rate_limit
from app.models.exception import HttpException
from app.router import root_api_router
from app.services import bgm as bgm_service
from app.services import material_upload as material_upload_service
from app.utils import utils


@asynccontextmanager
async def application_lifespan(_: FastAPI):
    """集中处理 API 进程启动恢复和关闭日志。"""
    logger.info("startup event")

    configured_api_key = config.app.get("api_key", "")
    if configured_api_key in (None, ""):
        logger.warning(
            "API key authentication is disabled; keep the API on a trusted network"
        )
    elif isinstance(configured_api_key, str):
        # 只记录保护范围，不得输出 Key、长度或摘要，避免凭据进入日志系统。
        logger.info("API key authentication is enabled for /api/v1 and /tasks")
    else:
        logger.error(
            "API key authentication is misconfigured: app.api_key must be a string"
        )

    # 跨平台发布由当前进程线程池执行，不会在服务重启后恢复。启动时把 Redis
    # 中确认已失去执行进程的活动状态收敛为失败，避免任务永久无法删除。
    from app.services import task as task_service

    task_service.recover_interrupted_cross_posts()
    # 同理，执行进程已退出的视频任务也不会继续运行，需要在恢复队列前收敛。
    task_service.recover_interrupted_tasks()

    # Redis queue entries persist across API restarts. No task_done callback
    # exists in the new process to dispatch them, so fill its worker slots now.
    from app.controllers.manager.redis_manager import RedisTaskManager
    from app.controllers.v1 import video as video_controller

    if isinstance(video_controller.task_manager, RedisTaskManager):
        video_controller.task_manager.resume_queued_tasks()
    try:
        yield
    finally:
        logger.info("shutdown event")


def exception_handler(request: Request, e: HttpException):
    return JSONResponse(
        status_code=e.status_code,
        content=utils.get_response(e.status_code, e.data, e.message),
    )


def validation_exception_handler(request: Request, e: RequestValidationError):
    # Rejected inputs can contain NaN/Infinity, and custom validators attach
    # exception objects to ctx. Neither can be emitted by JSONResponse directly.
    errors = jsonable_encoder(
        e.errors(),
        custom_encoder={
            float: lambda value: value if math.isfinite(value) else str(value),
            Exception: str,
        },
    )
    return JSONResponse(
        status_code=400,
        content=utils.get_response(
            status=400, data=errors, message="field required"
        ),
    )


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _normalize_allowed_origin(raw_origin: str) -> str | None:
    """把配置项折叠成浏览器实际发送的 Origin 形式。

    Origin 请求头由 RFC 6454 固定为 ``scheme://host[:port]``：没有路径、没有尾
    斜杠，scheme 与 host 已经规范化为小写，并且省略协议的默认端口。用户在浏览器
    地址栏复制得到的 ``https://frontend.example/``，以及显式写出默认端口的
    ``https://frontend.example:443``，都与浏览器实际发送的
    ``https://frontend.example`` 逐字符不相等，白名单因此静默失效：前端只看到
    CORS 报错，服务端只留下一行 blocked 日志，两边都指不到配置本身。

    返回 ``None`` 表示该写法不可能匹配任何 Origin（缺少 scheme、scheme 不是
    http/https、主机名为空或含空白、IPv6 字面量畸形、端口非法或越界），调用方应
    丢弃并告警。解析失败必须收敛成 ``None``：本函数在模块导入期就会执行，让
    ``urlsplit()`` 的 ``ValueError`` 冒出去等于整条 API 无法启动。
    """

    if raw_origin == "*":
        return raw_origin

    try:
        parsed = urlsplit(raw_origin)
        scheme = parsed.scheme.lower()
        # 读取 hostname / port 时才校验 netloc：畸形 IPv6 字面量由 urlsplit()
        # 抛出，非数字端口与超出 0-65535 的端口在读取 port 时抛出。
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None

    # 只有 http/https 具备浏览器 Origin 语义，且两者都有可省略的默认端口。
    if scheme not in _DEFAULT_PORTS or not host:
        return None
    if any(character.isspace() for character in host):
        return None

    # 显式小写，不依赖 urlsplit 在不同 Python 版本上的大小写处理。
    host = host.lower()
    # 浏览器序列化 Origin 时给 IPv6 字面量保留方括号；``:443``、``:0443``、
    # ``:80`` 这类默认端口写法必须折叠成不带端口的形态才能与真实前端匹配。
    if ":" in host:
        host = f"[{host}]"
    if port is None or port == _DEFAULT_PORTS[scheme]:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def parse_cors_allowed_origins(raw_origins: str | None) -> list[str]:
    """解析浏览器跨域来源白名单。

    CORS 只约束浏览器中的跨域 JavaScript，不影响 curl、Postman、n8n
    或服务端 SDK。未配置时返回空列表，表示默认不开放跨域访问；用户确实
    部署了独立网页前端时，再通过 ``CORS_ALLOWED_ORIGINS`` 显式开启。

    每个来源都先折叠成 Origin 请求头的规范形式，因此 ``https://a.example/``、
    ``HTTPS://A.Example`` 与显式默认端口的 ``https://a.example:443`` 得到同一个
    结果，无法匹配任何 Origin 的写法会被丢弃并留下告警。畸形条目只丢弃自身，
    不会中断这里的解析——本函数在模块导入期调用，解析异常会让 API 无法启动。
    """

    if not raw_origins:
        return []

    # 去除逗号分隔项两侧的空白，并忽略空项，避免常见的环境变量格式
    # ``https://a.example, https://b.example,`` 产生永远无法匹配的来源。
    origins: list[str] = []
    for candidate in raw_origins.split(","):
        item = candidate.strip()
        if not item:
            continue
        origin = _normalize_allowed_origin(item)
        if origin is None:
            logger.warning(
                f"ignoring configured CORS origin that cannot match a browser "
                f"Origin header: {item!r}"
            )
            continue
        if origin not in origins:
            origins.append(origin)
    return origins


def configure_cors(instance: FastAPI, allowed_origins: list[str]) -> None:
    """按显式白名单配置 CORS；空白名单保持默认同源策略。"""

    if not allowed_origins:
        logger.info(
            "browser cross-origin API access is disabled; set "
            "CORS_ALLOWED_ORIGINS to enable trusted origins"
        )
        return

    allow_all_origins = "*" in allowed_origins
    configured_api_key = config.app.get("api_key", "")
    if allow_all_origins and configured_api_key in (None, ""):
        # ``*`` 是用户显式选择的兼容模式，因此不强制拒绝启动；但在免认证
        # 状态下它会允许任意网页读取和调用 API，必须留下可定位的安全告警。
        logger.warning(
            "CORS allows every browser origin while API key authentication is "
            "disabled; configure app.api_key or restrict CORS_ALLOWED_ORIGINS"
        )

    instance.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        # Starlette 在 ``*`` 与 credentials 同时启用时会反射任意 Origin。
        # 通配符模式不需要 Cookie 认证，因此主动关闭 credentials；显式来源
        # 仍保留旧行为，避免影响已有独立网页前端的 credentials 请求模式。
        allow_credentials=not allow_all_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        # 远程 HTTPS 前端访问本机或局域网 API 时，现代浏览器会额外发送
        # Private Network Access 预检。只有精确白名单来源可以获得许可；
        # 通配符模式继续拒绝，避免任意网站探测用户的私有网络服务。
        allow_private_network=not allow_all_origins,
    )


def is_browser_origin_allowed(
    request: Request, allowed_origins: list[str]
) -> bool:
    """判断浏览器请求来源是否为同源或显式白名单来源。"""

    origin = request.headers.get("origin")
    if not origin:
        # curl、Postman、n8n 和服务端 SDK 通常不发送 Origin。保留这类请求，
        # 避免安全修复错误地改变现有 API 客户端的调用契约。
        return True
    if "*" in allowed_origins or origin in allowed_origins:
        return True

    # 浏览器对同源 POST 也可能发送 Origin。仅比较 scheme + authority，忽略
    # 路径和查询参数；反向代理部署若未正确转发公网 scheme/host，可通过显式
    # CORS_ALLOWED_ORIGINS 声明外部来源，避免依赖不可信的转发 Header。
    request_url = urlsplit(str(request.url))
    request_origin = f"{request_url.scheme}://{request_url.netloc}"
    return origin == request_origin


def configure_browser_access(instance: FastAPI, allowed_origins: list[str]) -> None:
    """同时配置服务端 Origin 防护与浏览器 CORS 响应策略。"""

    @instance.middleware("http")
    async def reject_untrusted_browser_origin(request: Request, call_next):
        """主动拒绝不可信浏览器来源，覆盖无需 CORS 预检的简单请求。"""

        if not is_browser_origin_allowed(request, allowed_origins):
            origin = request.headers.get("origin", "")
            logger.warning(
                f"blocked untrusted browser origin: method={request.method}, "
                f"path={request.url.path}, origin={origin}"
            )
            return JSONResponse(
                status_code=403,
                content=utils.get_response(
                    status=403,
                    message="cross-origin browser request is not allowed",
                ),
            )

        return await call_next(request)

    # CORS 中间件最后注册后位于 Origin 防护外层：可信预检可直接成功，
    # 非可信预检由 CORS 拒绝；无需预检的实际请求仍会进入上面的 403 防护。
    configure_cors(instance, allowed_origins)


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "testserver"})
_LOOPBACK_LISTEN_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _host_without_port(raw_host: str) -> str:
    """Return the lower-case host part of a Host header (IPv6 aware)."""
    host = raw_host.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        return host[1:end] if end > 0 else host
    if host.count(":") == 1:
        host = host.split(":", 1)[0]
    return host.rstrip(".")


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _configured_allowed_hosts() -> list[str] | None:
    raw = config.app.get("allowed_hosts", None)
    if raw in (None, "", []):
        return None
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, (list, tuple)):
        return None
    hosts = [str(item).strip().lower() for item in raw if str(item).strip()]
    return hosts or None


def is_host_allowed(raw_host: str | None) -> bool:
    """Reject Host headers that are not this server (DNS-rebinding defence).

    A rebinding page makes the browser send an attacker-chosen *name* in Host,
    so IP literals and the explicit allow-list are accepted while unknown names
    are not. With an API key and a non-loopback listen address the key already
    protects the API, so names are allowed unless ``app.allowed_hosts`` is set.
    """
    if not raw_host:
        return False
    host = _host_without_port(raw_host)
    if not host:
        return False
    if host in _LOOPBACK_HOSTS or _is_ip_literal(host):
        return True
    own_hostname = str(getattr(config, "hostname", "") or "").lower()
    if own_hostname and host == own_hostname:
        return True

    configured = _configured_allowed_hosts()
    if configured is not None:
        for pattern in configured:
            if pattern == "*" or host == pattern:
                return True
            if pattern.startswith("*.") and host.endswith(pattern[1:]):
                return True
        return False

    listen_host = str(getattr(config, "listen_host", "") or "").lower()
    api_key = config.app.get("api_key", "")
    return bool(api_key) and listen_host not in _LOOPBACK_LISTEN_HOSTS


class TrustedHostMiddleware:
    """Validate the Host header with IPv6 support and a configurable list."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            raw_host = None
            for name, value in scope.get("headers", []):
                if name == b"host":
                    raw_host = value.decode("latin-1")
                    break
            if not is_host_allowed(raw_host):
                logger.warning(
                    f"blocked untrusted Host header: path={scope.get('path')}, "
                    f"host={raw_host!r}"
                )
                if scope["type"] == "http":
                    response = JSONResponse(
                        status_code=400,
                        content=utils.get_response(
                            status=400, message="invalid host header"
                        ),
                    )
                    await response(scope, receive, send)
                else:
                    await send({"type": "websocket.close", "code": 1008})
                return
        await self.app(scope, receive, send)


_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
_DEFAULT_MAX_BODY_BYTES = 5 * 1024 * 1024


class _BodyTooLarge(Exception):
    pass


def _body_limit_for(scope) -> int | None:
    if scope.get("method") not in ("POST", "PUT", "PATCH"):
        return None
    path = scope.get("path", "").rstrip("/")
    if not path.startswith("/api/v1"):
        return None
    if path == "/api/v1/musics":
        return bgm_service.MAX_BGM_UPLOAD_BYTES + _MULTIPART_OVERHEAD_BYTES
    if path == "/api/v1/video_materials":
        return (
            max(
                material_upload_service.MAX_VIDEO_MATERIAL_UPLOAD_BYTES,
                material_upload_service.MAX_IMAGE_MATERIAL_UPLOAD_BYTES,
            )
            + _MULTIPART_OVERHEAD_BYTES
        )
    configured = config.app.get("max_request_body_bytes", _DEFAULT_MAX_BODY_BYTES)
    if isinstance(configured, bool) or not isinstance(configured, int):
        return _DEFAULT_MAX_BODY_BYTES
    return configured if configured > 0 else None


class RequestBodyLimitMiddleware:
    """Cap request bodies before Starlette spools multipart uploads to disk.

    A declared Content-Length over the cap is rejected immediately with 413;
    chunked or understated bodies are counted while they are read.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        limit = _body_limit_for(scope) if scope["type"] == "http" else None
        if limit is None:
            await self.app(scope, receive, send)
            return

        too_large = JSONResponse(
            status_code=413,
            content=utils.get_response(
                status=413, message="request body is too large"
            ),
        )
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = 0
                if declared > limit:
                    await too_large(scope, receive, send)
                    return
                break

        received = 0
        exceeded = False

        async def limited_receive():
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise _BodyTooLarge()
            return message

        async def guarded_send(message):
            # Framework layers (e.g. FastAPI body parsing) may translate the
            # read failure into their own 400/500. Once the cap is hit only
            # the 413 below may reach the client.
            if not exceeded:
                await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded:
            await too_large(scope, receive, send)


def configure_request_guards(instance: FastAPI) -> None:
    """Rate limiting, body-size cap and Host validation for the API."""

    @instance.middleware("http")
    async def throttle_api_posts(request: Request, call_next):
        if request.method == "POST" and request.url.path.startswith("/api/v1/"):
            if not rate_limit.allow_request(base.rate_limit_client(request)):
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": "60"},
                    content=utils.get_response(
                        status=429, message="too many requests, retry later"
                    ),
                )
        return await call_next(request)

    instance.add_middleware(RequestBodyLimitMiddleware)
    instance.add_middleware(TrustedHostMiddleware)


def get_application() -> FastAPI:
    """Initialize FastAPI application.

    Returns:
       FastAPI: Application object instance.

    """
    instance = FastAPI(
        title=config.project_name,
        description=config.project_description,
        version=config.project_version,
        debug=False,
        lifespan=application_lifespan,
    )
    instance.include_router(root_api_router)
    instance.add_exception_handler(HttpException, exception_handler)
    instance.add_exception_handler(RequestValidationError, validation_exception_handler)
    configure_request_guards(instance)
    return instance


app = get_application()


_PROTECTED_DOC_PATHS = frozenset(
    {"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}
)


@app.middleware("http")
async def protect_generated_task_files(request: Request, call_next):
    """保护任务产物静态路由，防止绕过 API 鉴权直接下载。

    ``/tasks`` 由 StaticFiles 独立挂载，无法复用 APIRouter 的依赖，
    因此在中间件中调用同一个 verify_token。鉴权函数会在未配置
    api_key 时放行；OPTIONS 预检请求也保留给 CORS 中间件处理。
    """

    request_path = request.url.path
    is_task_file = request_path == "/tasks" or request_path.startswith("/tasks/")
    # The interactive docs and schema describe every endpoint; require the key
    # for them too. /ping stays open for health checks.
    is_docs = request_path in _PROTECTED_DOC_PATHS
    if (is_task_file or is_docs) and request.method != "OPTIONS":
        try:
            base.verify_token(request)
        except HttpException as exception:
            return exception_handler(request, exception)

    return await call_next(request)


# 默认遵循浏览器同源策略；仅在用户显式配置可信网页来源时开放跨域。
cors_allowed_origins = parse_cors_allowed_origins(
    os.getenv("CORS_ALLOWED_ORIGINS", "")
)
configure_browser_access(app, cors_allowed_origins)

task_dir = utils.task_dir()
app.mount("/tasks", StaticFiles(directory=task_dir, html=True), name="")

public_dir = utils.public_dir()
app.mount("/", StaticFiles(directory=public_dir, html=True), name="")
