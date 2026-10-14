import copy
import errno
import os
import shutil
import socket
import tempfile
import threading
import time
from contextlib import ExitStack, contextmanager

import toml
from loguru import logger

from app import __version__

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
config_file = f"{root_dir}/config.toml"
_CONTAINER_CGROUP_MARKERS = ("docker", "containerd", "kubepods", "libpod", "podman")
_DOCKER_HOST_GATEWAY_NAME = "host.docker.internal"
_config_save_lock = threading.RLock()
_pending_config_lock = threading.RLock()
_pending_config_updates = {}
_pending_config_save_requested = False
_pending_config_flush_scheduled = False
_MISSING = object()
_DELETE = object()
_UTF8_BOM = "\ufeff"


_CONFIG_LOCK_TIMEOUT_SECONDS = 30.0
_CONFIG_LOCK_POLL_SECONDS = 0.05
_CONFIG_READ_RETRIES = 5
_IS_WINDOWS = os.name == "nt"

if os.name == "nt":  # pragma: no cover - exercised on Windows only
    import msvcrt

    def _try_lock_fd(fd, exclusive):
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock_fd(fd):
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock_fd(fd, exclusive):
        fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)

    def _unlock_fd(fd):
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def _locked_path(path, *, exclusive, create=False, timeout=None):
    """
    Hold an advisory cross-process lock on ``path`` (flock / msvcrt.locking).

    Yields True when locked and False when the path cannot be opened (missing
    file, read-only directory, ...) or the filesystem has no lock support. A
    lock that cannot be acquired in time raises TimeoutError so a wedged peer
    never hangs the process forever.
    """
    timeout = _CONFIG_LOCK_TIMEOUT_SECONDS if timeout is None else timeout
    flags = os.O_RDWR if (exclusive or create) else os.O_RDONLY
    if create:
        flags |= os.O_CREAT
    try:
        fd = os.open(path, flags, 0o600)
    except OSError:
        # A read-only mount refuses O_RDWR; flock still works on a read-only fd.
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            yield False
            return
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                _try_lock_fd(fd, exclusive)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                    yield False
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for config lock: {path}")
                time.sleep(_CONFIG_LOCK_POLL_SECONDS)
        try:
            yield True
        finally:
            try:
                _unlock_fd(fd)
            except OSError:
                pass
    finally:
        os.close(fd)


@contextmanager
def _config_file_locks(*, exclusive):
    """
    Lock a sidecar lock file and, on POSIX, the config file itself.

    The sidecar covers shared directories (atomic replace swaps the config
    inode, so locking only that inode would not exclude peers). Locking the
    config file covers Docker single-file bind mounts, where the sidecar lives
    in each container's own filesystem but the config inode is shared.

    On Windows ``msvcrt.locking`` is a mandatory byte-range lock: it would make
    every other read of the locked file fail with PermissionError. There only
    writers take the sidecar lock (mutual exclusion among writers); config.toml
    is never locked and readers take no lock, relying on retry instead.
    """
    if _IS_WINDOWS and not exclusive:
        yield
        return
    lock_path = os.path.join(
        os.path.dirname(os.path.abspath(config_file)), ".config.toml.lock"
    )
    with ExitStack() as stack:
        stack.enter_context(_locked_path(lock_path, exclusive=exclusive, create=True))
        if not _IS_WINDOWS:
            stack.enter_context(_locked_path(config_file, exclusive=exclusive))
        yield


def _best_effort_chmod_private(path):
    """Restrict the secrets file to its owner where the platform/mount allows."""
    if os.name == "nt":
        return
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Bind mounts and foreign-owned files may refuse chmod; not fatal.
        pass


def _fsync_directory(path):
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class _SynchronizedConfig(dict):
    """保持 dict 使用方式不变，同时让运行期配置写操作服从同一把锁。"""

    def __setitem__(self, key, value):
        # Streamlit 每次整页 rerun 都会把当前控件值重新写回配置。视频任务持有
        # runtime_config_lock 时，如果值没有变化，这次写入没有任何副作用，也
        # 不应让刷新后的页面卡在表单中途。真正改变配置的写入仍进入下方锁，
        # 因而不能在正在生成的视频中途切换 Provider、密钥或其它全局设置。
        current = super().get(key, _MISSING)
        if current is not _MISSING and current == value:
            return
        with _config_save_lock:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        with _config_save_lock:
            super().__delitem__(key)

    def clear(self):
        if not self:
            return
        with _config_save_lock:
            super().clear()

    def pop(self, key, default=_MISSING):
        # ``pop(key, default)`` 在 key 不存在时同样不会改变配置。WebUI 使用
        # 这种写法表达“采用默认策略”，刷新时必须允许它直接完成。
        if key not in self:
            if default is _MISSING:
                raise KeyError(key)
            return default
        with _config_save_lock:
            if default is _MISSING:
                return super().pop(key)
            return super().pop(key, default)

    def setdefault(self, key, default=None):
        # 与 __setitem__ 相同，已存在 key 的 setdefault 是只读操作。提前返回
        # 可以让只读取默认配置的页面刷新不受长任务配置锁影响。
        current = super().get(key, _MISSING)
        if current is not _MISSING:
            return current
        with _config_save_lock:
            return super().setdefault(key, default)

    def update(self, *args, **kwargs):
        changes = dict(*args, **kwargs)
        if all(
            (current := dict.get(self, key, _MISSING)) is not _MISSING
            and current == value
            for key, value in changes.items()
        ):
            return
        with _config_save_lock:
            super().update(changes)


def _pending_update_key(config_section, key):
    """为进程内固定配置分区生成待更新键。"""
    return id(config_section), key


def update_config_nonblocking(config_section, key, value):
    """
    非阻塞更新 WebUI 的运行期配置。

    视频生成会持有 ``runtime_config_lock``，确保同一任务不会在执行中途切换
    Provider、密钥或语音配置。Streamlit 控件发生变化时不能等待这把长任务锁，
    否则浏览器会表现为页面冻结。锁空闲时立即更新；锁繁忙时只保留每个配置项
    的最新值，并在当前任务释放锁时统一应用。

    返回 True 表示值已经生效，False 表示已进入待更新队列。
    """
    # 所有更新都先进入同一队列，再尝试获取配置锁。这样多个页面同时修改同一
    # 配置项时，写入队列的先后顺序就是最终顺序，不会出现较早线程在获取锁后
    # 把较新线程已经排队的值误删掉。
    with _pending_config_lock:
        _pending_config_updates[_pending_update_key(config_section, key)] = (
            config_section,
            key,
            copy.deepcopy(value),
        )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        # 调用方通常会在本次 Streamlit rerun 末尾请求保存，但不能依赖这一步
        # 一定执行。例如页面中途异常或更新恰好发生在任务退出保存阶段时，仍需
        # 有后台刷新线程保证排队值最终生效。
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return config_section.get(key, _MISSING) == value
    finally:
        _config_save_lock.release()


def delete_config_nonblocking(config_section, key):
    """
    非阻塞删除 WebUI 配置项。

    “使用默认值”需要真正移除配置项，而不是写入空字符串。视频任务占用配置
    锁时，删除意图会覆盖同一配置项之前排队的更新，并在任务结束后执行。
    """
    with _pending_config_lock:
        _pending_config_updates[_pending_update_key(config_section, key)] = (
            config_section,
            key,
            _DELETE,
        )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return key not in config_section
    finally:
        _config_save_lock.release()


def _apply_pending_config_updates_locked():
    """在持有配置写锁时应用 WebUI 暂存的最新配置值。"""
    with _pending_config_lock:
        updates = list(_pending_config_updates.values())
        _pending_config_updates.clear()
        # 应用配置时继续持有待更新锁。读取“当前值 + 待更新值”快照的线程由此
        # 只能看到应用前或应用后的完整状态，不会读到只更新了一半的配置集合。
        for config_section, key, value in updates:
            if value is _DELETE:
                config_section.pop(key, None)
            else:
                config_section[key] = value
    return bool(updates)


def snapshot_config_with_pending(config_section):
    """
    返回配置分区的有效快照，并合并尚未应用的 WebUI 更新。

    视频任务持锁期间不能改写全局配置，但用户仍可准备下一条内容。LLM 请求
    使用这个快照后，界面中刚选择的 Provider、模型和密钥会参与新请求，同时
    不会改变正在执行的视频任务。
    """
    with _pending_config_lock:
        snapshot = dict(config_section)
        section_id = id(config_section)
        for (pending_section_id, key), (_, _, value) in _pending_config_updates.items():
            if pending_section_id != section_id:
                continue
            if value is _DELETE:
                snapshot.pop(key, None)
            else:
                snapshot[key] = copy.deepcopy(value)
    return snapshot


def _flush_pending_config_locked(*, suppress_save_errors):
    """在持有配置写锁时应用并保存当前所有待处理配置。"""
    global _pending_config_save_requested

    updates_applied = _apply_pending_config_updates_locked()
    with _pending_config_lock:
        save_requested = _pending_config_save_requested
        _pending_config_save_requested = False

    if not updates_applied and not save_requested:
        return True

    try:
        save_config()
        return True
    except Exception as exc:
        # 内存中的配置已经成功应用，保存失败时只保留待保存标记。视频任务不应
        # 因配置文件暂时不可写而被改判失败；下一次页面交互会再次触发保存。
        with _pending_config_lock:
            _pending_config_save_requested = True
        if not suppress_save_errors:
            raise
        logger.exception(f"failed to save deferred runtime config: {exc}")
        return False


def _run_deferred_config_flush():
    """等待长任务释放配置锁，并可靠清空期间积累的配置更新。"""
    global _pending_config_flush_scheduled

    while True:
        with _config_save_lock:
            flush_succeeded = _flush_pending_config_locked(
                suppress_save_errors=True
            )

        with _pending_config_lock:
            has_pending_work = bool(
                _pending_config_updates or _pending_config_save_requested
            )
            if not flush_succeeded or not has_pending_work:
                _pending_config_flush_scheduled = False
                return


def _schedule_deferred_config_flush():
    """保证同一时间最多只有一个后台线程等待刷新配置。"""
    global _pending_config_flush_scheduled

    with _pending_config_lock:
        if _pending_config_flush_scheduled:
            return
        _pending_config_flush_scheduled = True

    try:
        threading.Thread(
            target=_run_deferred_config_flush,
            name="mpt-config-flush",
            daemon=True,
        ).start()
    except Exception:
        # No worker owns this reservation when construction/start fails.
        # Leave queued updates and the save request intact so a later call
        # can schedule another flush rather than silently waiting forever.
        with _pending_config_lock:
            _pending_config_flush_scheduled = False
        raise


def try_save_config(force=False):
    """
    非阻塞保存 WebUI 配置，锁繁忙时交由当前长任务结束后保存。

    普通 API、CLI 和维护脚本仍可调用 ``save_config`` 获得原来的阻塞写入语义；
    只有 Streamlit rerun 使用本函数，避免页面为等待视频任务而长时间无响应。
    """
    global _pending_config_save_requested

    if force:
        # Durable path for credential rotation: wait for the lock instead of
        # deferring, apply queued updates, then write every in-memory key.
        with _config_save_lock:
            _apply_pending_config_updates_locked()
            with _pending_config_lock:
                _pending_config_save_requested = False
            save_config(force=True)
        return True

    with _pending_config_lock:
        _pending_config_save_requested = True

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        return _flush_pending_config_locked(suppress_save_errors=False)
    finally:
        _config_save_lock.release()


@contextmanager
def runtime_config_lock():
    """
    在一次依赖全局配置的完整操作期间阻止其它 WebUI 会话改写配置。

    当前项目默认绑定本地回环地址，配置仍然是单用户全局配置。这个轻量锁主要
    保护生成、试听等长操作，避免另一个标签页在操作中途切换 Provider 或密钥。
    """
    with _config_save_lock:
        # 如果上一个短操作释放锁时后台刷新线程尚未获得调度，新任务必须在读取
        # Provider、密钥等全局配置前先应用队列，不能继续使用旧配置执行整条流水线。
        _flush_pending_config_locked(suppress_save_errors=True)
        try:
            yield
        finally:
            _flush_pending_config_locked(suppress_save_errors=True)


@contextmanager
def try_runtime_config_lock():
    """
    尝试获取运行期配置锁，并立即返回是否成功。

    WebUI 试听属于用户主动触发的短操作，不应在后台视频任务持锁时等待数分钟。
    调用方可以在未获取锁时就近提示用户稍后重试；成功获取后仍能保证试听期间
    Provider、密钥和模型配置不会被其它会话修改。
    """
    acquired = _config_save_lock.acquire(blocking=False)
    try:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
        yield acquired
    finally:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
            _config_save_lock.release()


def is_running_in_container(
    dockerenv_path: str = "/.dockerenv",
    containerenv_path: str = "/run/.containerenv",
    cgroup_path: str = "/proc/1/cgroup",
) -> bool:
    """
    判断当前进程是否运行在容器内。

    这个判断主要用于 Ollama 默认地址选择：
    - 普通本机运行时，`localhost` 指向用户机器本身；
    - Docker 容器内，`localhost` 指向容器自己，访问宿主机 Ollama
      通常需要使用 `host.docker.internal`。

    不能只判断 `/proc/1/cgroup` 是否存在，因为普通 Linux 也会有这个文件。
    这里只在检测到明确的容器标记时返回 True，避免误伤非 Docker Linux 用户。
    参数保留为可注入路径，便于单元测试覆盖不同运行环境。
    """
    if os.path.isfile(dockerenv_path) or os.path.isfile(containerenv_path):
        return True

    try:
        with open(cgroup_path, mode="r", encoding="utf-8") as fp:
            cgroup_content = fp.read().lower()
    except OSError:
        return False

    return any(marker in cgroup_content for marker in _CONTAINER_CGROUP_MARKERS)


def _can_resolve_hostname(hostname: str) -> bool:
    try:
        socket.gethostbyname(hostname)
    except OSError:
        return False
    return True


def _decode_linux_route_gateway(hex_gateway: str) -> str:
    # /proc/net/route 里的 Gateway 是 16 进制小端序，例如 010011AC 表示
    # 172.17.0.1。这里单独解析，是为了在原生 Linux Docker 没有
    # host.docker.internal DNS 记录时，还能尝试访问容器默认网关上的宿主机。
    if len(hex_gateway) != 8:
        raise ValueError("invalid gateway length")

    octets = [
        str(int(hex_gateway[index : index + 2], 16)) for index in range(6, -1, -2)
    ]
    return ".".join(octets)


def get_container_default_gateway_ip(route_path: str = "/proc/net/route") -> str:
    """
    读取 Linux 容器里的默认网关 IP。

    Docker Desktop 通常提供 `host.docker.internal`，但原生 Linux Docker
    默认不一定提供这个 DNS 名称。默认网关通常可以作为访问宿主机服务的
    兜底地址；如果用户的 Ollama 只监听 127.0.0.1，则仍需要用户让
    Ollama 监听宿主机网卡或手动配置 `ollama_base_url`。
    """
    try:
        with open(route_path, mode="r", encoding="utf-8") as fp:
            route_lines = fp.readlines()
    except OSError:
        return ""

    for line in route_lines[1:]:
        fields = line.strip().split()
        if len(fields) < 8:
            continue

        destination = fields[1]
        gateway = fields[2]
        if (
            destination != "00000000"
            or fields[7] != "00000000"
            or gateway == "00000000"
        ):
            continue

        try:
            return _decode_linux_route_gateway(gateway)
        except ValueError:
            logger.warning(f"invalid container gateway route entry: {line.strip()}")
            return ""

    return ""


def get_default_ollama_base_url() -> str:
    """
    返回 Ollama 的默认 OpenAI-compatible base_url。

    用户显式配置 `ollama_base_url` 时不会走这里；这里只处理“未配置时的
    最佳默认值”。容器内默认指向宿主机，普通本机运行默认指向 localhost。
    """
    if not is_running_in_container():
        return "http://localhost:11434/v1"

    if _can_resolve_hostname(_DOCKER_HOST_GATEWAY_NAME):
        return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"

    gateway_ip = get_container_default_gateway_ip()
    if gateway_ip:
        logger.info(
            "host.docker.internal is not resolvable, fallback to container "
            f"default gateway for Ollama: {gateway_ip}"
        )
        return f"http://{gateway_ip}:11434/v1"

    logger.warning(
        "failed to resolve host.docker.internal and container default gateway; "
        "fallback to host.docker.internal for Ollama"
    )
    return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"


def _load_toml_with_retry(config_path):
    """Read under a shared lock, retrying briefly if a writer is mid-update."""
    for attempt in range(_CONFIG_READ_RETRIES):
        try:
            if os.path.abspath(config_path) == os.path.abspath(config_file):
                with _config_file_locks(exclusive=False):
                    return toml.load(config_path)
            return toml.load(config_path)
        except (toml.TomlDecodeError, UnicodeDecodeError, PermissionError):
            # PermissionError: Windows denies reads while a writer replaces or
            # holds the file; it is transient, so back off briefly and retry.
            if attempt == _CONFIG_READ_RETRIES - 1:
                raise
            time.sleep(0.05 * (attempt + 1))
        except TimeoutError:
            return toml.load(config_path)


def _load_toml_config(config_path: str):
    """
    加载 TOML，并兼容 Windows 编辑器可能写入的重复 UTF-8 BOM。

    ``utf-8-sig`` 只会移除文件开头的一个 BOM。部分 Windows 编辑器或
    解压、保存流程可能再次写入 BOM，导致第二个不可见字符进入 TOML
    解析器并在第一行报错。这里仅在标准解析失败后做一次只读归一化，
    不回写原文件，避免意外覆盖用户已经填写的 API Key。
    """
    try:
        return _load_toml_with_retry(config_path)
    except (toml.TomlDecodeError, UnicodeDecodeError) as exc:
        logger.warning(
            "load config failed, retry with UTF-8 BOM compatibility: "
            f"path={config_path}, error={type(exc).__name__}: {exc}"
        )

    try:
        with open(config_path, mode="r", encoding="utf-8-sig") as fp:
            config_content = fp.read()

        normalized_content = config_content.lstrip(_UTF8_BOM)
        removed_bom_count = len(config_content) - len(normalized_content)
        if removed_bom_count:
            logger.warning(
                "removed repeated UTF-8 BOM characters while loading config: "
                f"path={config_path}, count={removed_bom_count}"
            )
        return toml.loads(normalized_content)
    except (toml.TomlDecodeError, UnicodeDecodeError) as exc:
        logger.error(
            "config file is not valid TOML after UTF-8 BOM normalization: "
            f"path={config_path}, error={type(exc).__name__}: {exc}"
        )
        raise



def _initialize_config_from_example(example_file):
    """Publish a complete first-boot config without replacing another writer."""
    fd, temp_path = tempfile.mkstemp(
        prefix=".config-init-", suffix=".toml.tmp",
        dir=os.path.dirname(os.path.abspath(config_file)),
    )
    try:
        with os.fdopen(fd, "wb") as destination:
            with open(example_file, "rb") as source:
                shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        _best_effort_chmod_private(temp_path)
        try:
            if os.name == "nt":
                # Windows rename refuses an existing destination. POSIX rename
                # overwrites it, so use an exclusive hard-link publication there.
                os.rename(temp_path, config_file)
            else:
                os.link(temp_path, config_file)
        except FileExistsError:
            return False
        except OSError as exc:
            raise OSError(
                exc.errno,
                "cannot safely initialize config.toml; copy config.example.toml "
                "to config.toml before starting the application",
            ) from exc
        return True
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            # A successful Windows rename consumed the temporary path.
            pass


def load_config():
    # Docker may create an empty directory when a missing config.toml is bind-mounted.
    # Only remove that empty placeholder; a nonempty directory may contain user data.
    if os.path.isdir(config_file):
        try:
            os.rmdir(config_file)
        except FileNotFoundError:
            # Another initializer already removed this empty bind-mount stub.
            pass
        except OSError as exc:
            # Another initializer may already have replaced the empty Docker
            # stub with its complete regular file. Accept only that race, not
            # permission failures or a still-present/nonempty directory.
            if not (
                isinstance(exc, NotADirectoryError) and os.path.isfile(config_file)
            ):
                raise IsADirectoryError(
                    f"{config_file} is a directory and cannot be used as the config file; "
                    "move or rename it, then restart"
                ) from exc

    if not os.path.isfile(config_file):
        example_file = f"{root_dir}/config.example.toml"
        if os.path.isfile(example_file):
            if _initialize_config_from_example(example_file):
                logger.info("copy config.example.toml to config.toml")

    logger.info(f"load config from file: {config_file}")

    return _load_toml_config(config_file)


def _build_config_snapshot():
    config_to_save = dict(_cfg)
    config_to_save["app"] = dict(app)
    config_to_save["azure"] = dict(azure)
    config_to_save["siliconflow"] = dict(siliconflow)
    config_to_save["minimax_tts"] = dict(minimax_tts)
    config_to_save["elevenlabs"] = dict(elevenlabs)
    config_to_save["chatterbox"] = dict(chatterbox)
    config_to_save["kokoro"] = dict(kokoro)
    config_to_save["fish_audio"] = dict(fish_audio)
    config_to_save["voxcpm"] = dict(voxcpm)
    config_to_save["ui"] = dict(ui)
    return config_to_save


# What this process last loaded from / wrote to disk. Diffing the live config
# against it yields the keys this process changed, so a save can be merged onto
# the file instead of overwriting changes made by another process.
_baseline = {}


def _config_changes(baseline, current, *, force=False):
    """Return (sets, deletes) of ``(path, value)`` / ``path`` between two configs."""
    sets, deletes = [], []
    for key, value in current.items():
        if isinstance(value, dict):
            old = baseline.get(key)
            old = old if isinstance(old, dict) else {}
            for sub, sub_value in value.items():
                if force or sub not in old or old[sub] != sub_value:
                    sets.append(((key, sub), sub_value))
            for sub in old:
                if sub not in value:
                    deletes.append((key, sub))
            if not value and key not in baseline:
                sets.append(((key,), {}))
        elif force or key not in baseline or baseline[key] != value:
            sets.append(((key,), value))
    for key in baseline:
        if key not in current:
            deletes.append((key,))
    return sets, deletes


def _apply_config_changes(disk_config, sets, deletes):
    merged = copy.deepcopy(disk_config)
    for path, value in sets:
        if len(path) == 1:
            merged[path[0]] = copy.deepcopy(value)
        else:
            section = merged.get(path[0])
            if not isinstance(section, dict):
                section = merged[path[0]] = {}
            section[path[1]] = copy.deepcopy(value)
    for path in deletes:
        if len(path) == 1:
            merged.pop(path[0], None)
        else:
            section = merged.get(path[0])
            if isinstance(section, dict):
                section.pop(path[1], None)
    return merged


def save_config(force=False):
    """
    保存运行时配置（跨进程安全）。

    线程之间由进程内可重入锁串行化；多个进程（例如 docker-compose 中共享同一
    挂载的 webui 与 api 容器）则通过 flock / msvcrt 文件锁串行化。持锁期间会
    重新读取磁盘上的 config.toml，只把本进程相对上次加载/保存发生变化的键合并
    进去，再写回，避免整份覆盖导致另一个进程刚写入的内容（例如轮换后的刷新
    令牌）丢失。

    写入先落到同目录临时文件再 os.replace 原子替换；Docker 单文件 bind mount
    无法 rename（EBUSY）时，在锁内原地覆盖并 fsync，读取方持共享锁读取。

    ``force=True`` 把内存中的每个键都视为本进程已修改并覆盖磁盘上的值（磁盘上
    仅存在于其它进程的键仍保留）；适用于必须落盘的凭据轮换。
    """
    global _baseline
    with _config_save_lock:
        config_to_save = _build_config_snapshot()
        config_dir = os.path.dirname(os.path.abspath(config_file))

        with _config_file_locks(exclusive=True):
            disk_config = None
            disk_text = None
            for attempt in range(_CONFIG_READ_RETRIES):
                try:
                    with open(config_file, mode="r", encoding="utf-8") as f:
                        disk_text = f.read()
                    disk_config = toml.loads(disk_text.lstrip(_UTF8_BOM))
                    break
                except PermissionError:
                    # Transient on Windows; never fall back to a wholesale
                    # overwrite just because a reader briefly held the file.
                    if attempt == _CONFIG_READ_RETRIES - 1:
                        raise
                    time.sleep(0.05 * (attempt + 1))
                except (OSError, UnicodeError, toml.TomlDecodeError):
                    disk_config = None
                    break

            if disk_config is None:
                merged_config = config_to_save
            else:
                sets, deletes = _config_changes(_baseline, config_to_save, force=force)
                merged_config = _apply_config_changes(disk_config, sets, deletes)
            serialized_config = toml.dumps(merged_config)

            # WebUI 完整 rerun 结束时会调用保存。内容没有变化时直接返回，避免每次
            # 点击普通控件都产生一次磁盘写入和 fsync。
            if disk_text is not None and disk_text == serialized_config:
                _baseline = copy.deepcopy(config_to_save)
                _cfg.clear()
                _cfg.update(config_to_save)
                return

            temp_path = ""
            try:
                fd, temp_path = tempfile.mkstemp(
                    prefix=".config-",
                    suffix=".toml.tmp",
                    dir=root_dir,
                )
                with os.fdopen(fd, mode="w", encoding="utf-8") as f:
                    f.write(serialized_config)
                    f.flush()
                    os.fsync(f.fileno())
                _best_effort_chmod_private(temp_path)
                try:
                    for attempt in range(_CONFIG_READ_RETRIES):
                        try:
                            os.replace(temp_path, config_file)
                            break
                        except PermissionError:
                            # Windows: a concurrent reader has the target open.
                            if attempt == _CONFIG_READ_RETRIES - 1:
                                raise
                            time.sleep(0.05 * (attempt + 1))
                    _fsync_directory(config_dir)
                except OSError as exc:
                    if exc.errno != errno.EBUSY:
                        raise

                    logger.warning(
                        "atomic config replacement is unavailable for the mounted "
                        f"file, fallback to in-place write: {config_file}"
                    )
                    # r+/seek/truncate keeps the file non-empty throughout; the
                    # exclusive lock plus shared-lock readers hide the rest.
                    with open(config_file, mode="r+", encoding="utf-8") as f:
                        f.seek(0)
                        f.write(serialized_config)
                        f.truncate()
                        f.flush()
                        os.fsync(f.fileno())
                    _best_effort_chmod_private(config_file)
                _baseline = copy.deepcopy(config_to_save)
                _cfg.clear()
                _cfg.update(config_to_save)
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)


_cfg = load_config()
app = _SynchronizedConfig(_cfg.get("app", {}))
whisper = _cfg.get("whisper", {})
proxy = _cfg.get("proxy", {})
azure = _SynchronizedConfig(_cfg.get("azure", {}))
siliconflow = _SynchronizedConfig(_cfg.get("siliconflow", {}))
minimax_tts = _SynchronizedConfig(_cfg.get("minimax_tts", {}))
elevenlabs = _SynchronizedConfig(_cfg.get("elevenlabs", {}))
chatterbox = _SynchronizedConfig(_cfg.get("chatterbox", {}))
kokoro = _SynchronizedConfig(_cfg.get("kokoro", {}))
fish_audio = _SynchronizedConfig(_cfg.get("fish_audio", {}))
voxcpm = _SynchronizedConfig(_cfg.get("voxcpm", {}))
ui = _SynchronizedConfig(
    _cfg.get(
        "ui",
        {
            "hide_log": False,
        },
    )
)

hostname = socket.gethostname()

log_level = _cfg.get("log_level", "DEBUG")
listen_host = _cfg.get("listen_host", "0.0.0.0")
listen_port = _cfg.get("listen_port", 8080)
project_name = _cfg.get("project_name", "MoneyPrinterTurbo")
project_description = _cfg.get(
    "project_description",
    "<a href='https://github.com/harry0703/MoneyPrinterTurbo'>https://github.com/harry0703/MoneyPrinterTurbo</a>",
)
project_version = _cfg.get("project_version", __version__)
reload_debug = False

app["redis_host"] = os.getenv(
    "MPT_APP_REDIS_HOST",
    os.getenv("REDIS_HOST", app.get("redis_host", "localhost")),
)

_baseline = copy.deepcopy(_build_config_snapshot())

ffmpeg_path = app.get("ffmpeg_path", "")
if ffmpeg_path and os.path.isfile(ffmpeg_path):
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

logger.info(f"{project_name} v{project_version}")
