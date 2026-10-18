import os
import re


def resolve_path_within_directory(
    base_dir: str,
    unsafe_path: str,
    *,
    require_file: bool = True,
) -> str:
    # 用户传入的路径可能是文件名、相对路径、绝对路径，也可能夹带 `../`。
    # 这里统一解析成真实路径，并用 commonpath 判断它是否仍在允许目录内。
    # 这样比简单判断字符串前缀可靠，可以覆盖符号链接、重复分隔符、相对路径
    # 等场景，适用于上传目录、素材目录、任务产物目录这类白名单目录。
    if not unsafe_path:
        raise ValueError("empty path is not allowed")

    base_dir_real = os.path.realpath(base_dir)
    candidate_path = unsafe_path
    if not os.path.isabs(candidate_path):
        candidate_path = os.path.join(base_dir_real, candidate_path)

    resolved_path = os.path.realpath(candidate_path)
    try:
        common_path = os.path.commonpath([base_dir_real, resolved_path])
    except ValueError as exc:
        # Windows 下不同盘符会触发 ValueError，这类路径一定不属于允许目录。
        raise ValueError("path is outside the allowed directory") from exc

    if common_path != base_dir_real:
        raise ValueError("path is outside the allowed directory")

    if require_file and not os.path.isfile(resolved_path):
        raise ValueError("file does not exist")

    return resolved_path


def prune_upload_directory(
    directory: str,
    *,
    max_age_seconds: float = 0,
    max_total_bytes: int = 0,
    incoming_bytes: int = 0,
    suffixes: tuple[str, ...] = (),
    name_pattern: "re.Pattern[str] | None" = None,
    protected_prefix: str = ".",
    now: float | None = None,
) -> list[str]:
    """Delete expired and excess user uploads, oldest first.

    Uploads are stored under immutable UUID names, so age (mtime) and total
    size are the only meaningful retention criteria. ``0`` disables a limit.
    ``name_pattern`` limits pruning to server-generated names so files a user
    placed manually in the directory are never removed. Hidden files (staging files use ``protected_prefix``) and symlinks are never
    touched. Returns the names removed; failures to delete are ignored.
    """
    import time

    if max_age_seconds <= 0 and max_total_bytes <= 0:
        return []
    current_time = time.time() if now is None else now
    entries: list[tuple[float, int, str, str]] = []
    try:
        with os.scandir(directory) as scanner:
            for entry in scanner:
                if entry.name.startswith(protected_prefix):
                    continue
                if name_pattern is not None and not name_pattern.match(entry.name):
                    continue
                if suffixes and os.path.splitext(entry.name)[1].lower() not in suffixes:
                    continue
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    info = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                entries.append((info.st_mtime, info.st_size, entry.name, entry.path))
    except OSError:
        return []

    entries.sort()
    removed: list[str] = []
    remaining = []
    for mtime, size, name, path in entries:
        if max_age_seconds > 0 and current_time - mtime > max_age_seconds:
            try:
                os.remove(path)
                removed.append(name)
                continue
            except OSError:
                pass
        remaining.append((mtime, size, name, path))

    if max_total_bytes > 0:
        total = sum(size for _, size, _, _ in remaining) + max(incoming_bytes, 0)
        for mtime, size, name, path in remaining:
            if total <= max_total_bytes:
                break
            try:
                os.remove(path)
            except OSError:
                continue
            removed.append(name)
            total -= size
    return removed


UPLOADED_FILE_NAME_PATTERN = re.compile(r"^[0-9a-f]{32}\.[A-Za-z0-9]+$")
