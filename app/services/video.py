import bisect
import itertools
import hashlib
from concurrent.futures import ThreadPoolExecutor
import io
import math
import os
import random
import re
import gc
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from contextlib import ExitStack, contextmanager, redirect_stdout
from functools import lru_cache
from typing import Callable, List
from loguru import logger
import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    Effect,
    ImageClip,
    TextClip,
    VideoFileClip,
    afx,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

from app.config import config
from app.models import const
from app.models.schema import (
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoFitMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services import bgm as bgm_service
from app.services.utils import video_effects
from app.utils import file_security, logging_utils, utils
from app.utils.subtitle_reader import moviepy_subtitles

class SubClippedVideoClip:
    def __init__(
        self,
        file_path,
        start_time=None,
        end_time=None,
        width=None,
        height=None,
        duration=None,
        source_file_path=None,
    ):
        self.file_path = file_path
        self.start_time = start_time
        self.end_time = end_time
        self.width = width
        self.height = height
        self.source_file_path = source_file_path or file_path
        if duration is None:
            self.duration = end_time - start_time
        else:
            self.duration = duration

    def __str__(self):
        return f"SubClippedVideoClip(file_path={self.file_path}, start_time={self.start_time}, end_time={self.end_time}, duration={self.duration}, width={self.width}, height={self.height})"


audio_codec = "aac"
# Docker 里的 ffmpeg/AAC 组合在默认配置下更容易出现音频质量波动，
# 这里显式抬高音频码率，避免成片阶段因为默认值过低而引入明显失真。
audio_bitrate = "192k"
fps = 30
# FFmpeg 按帧率拼接/转码时，最终时长可能比 MoviePy 读到的理论时长短几十毫秒。
# 这里给视频素材多留一个很小的安全余量，避免音频末尾因为帧舍入出现黑屏、
# 卡顿或最后一小段旁白没有画面的情况。
_VIDEO_DURATION_SAFETY_MARGIN = 0.1
_MIN_MATERIAL_DIMENSION = 480
# 消息类应用和部分编码器会把画面尺寸向下取整，例如 WhatsApp 会把 9:16 的
# 素材压成 478x850，比 480 少两个像素。直接按 480 硬卡会让这类素材全部被
# 丢弃，最终以 "no valid materials found" 整体失败。这里留一个很小的容差，
# 既能放行仅仅因为取整而略低于阈值的素材，也仍然能挡住真正的低清素材。
_MIN_DIMENSION_TOLERANCE = 10
# 默认串行处理片段，避免旧用户升级后 CPU 和内存占用突然增加。
# 需要加速时可自行提高并发数，但每路都会启动独立的编码任务。
_CLIP_PROCESSING_CONCURRENCY = 1


def _get_clip_processing_concurrency() -> int:
    try:
        concurrency = int(config.app.get("video_clip_concurrency", _CLIP_PROCESSING_CONCURRENCY))
    except (TypeError, ValueError):
        concurrency = _CLIP_PROCESSING_CONCURRENCY
    return max(1, min(8, concurrency))
_DEFAULT_VIDEO_CODEC = "libx264"
# ffmpeg 串联片段期间没有阶段日志，`subprocess.run` 又阻塞到进程退出，耗时拼接在
# 日志上表现为“无输出”。这里按间隔记录存活信息，便于区分编码中与已经卡死。
_FFMPEG_CONCAT_HEARTBEAT_SECONDS = 30.0
# MoviePy 写出最终成片时同样没有任何日志，一分钟的竖屏视频在 CPU 上要编码数
# 分钟。沿用拼接阶段的做法，按间隔记录存活信息。
_STAGE_HEARTBEAT_SECONDS = 30.0
_DEFAULT_FFMPEG_CONCAT_TIMEOUT_SECONDS = 3600
_SUBTITLE_SPRING_DURATION_SECONDS = 0.18
_MIN_SUBTITLE_SPRING_SCALE = 0.05
_MAX_SUBTITLE_SPRING_SCALE = 1.35
_SUPPORTED_VIDEO_CODECS = (
    "libx264",
    "h264_nvenc",
    "h264_amf",
    "h264_qsv",
    "h264_mf",
    "h264_videotoolbox",
)
_runtime_disabled_video_codecs = set()
# MoviePy pipes sRGB frames, and ffmpeg's default RGB→YUV matrix is BT.601 with no color tags,
# while players and YouTube decode untagged HD as BT.709 and shift the colors. Convert with the
# BT.709 matrix and tag the stream; setparams writes the tags the -color_* flags alone do not.
# The pixel format stays the encoder's choice: MoviePy leaves odd-sized frames to libx264.
_BT709_VIDEO_FILTER = (
    "scale=out_color_matrix=bt709:out_range=tv,"
    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
)
_BT709_FFMPEG_PARAMS = ["-vf", _BT709_VIDEO_FILTER]
# Force 4:2:0 chroma so every player (browsers, phones, social uploaders) can decode
# the result, whatever pixel format the encoder would otherwise pick. 4:2:0 needs
# even dimensions, so an odd-sized frame (e.g. a 580x751 photo) loses one pixel row.
_YUV420_VIDEO_FILTER = (
    "crop=trunc(iw/2)*2:trunc(ih/2)*2," + _BT709_VIDEO_FILTER + ",format=yuv420p"
)
_BT709_FFMPEG_PARAMS = ["-vf", _YUV420_VIDEO_FILTER]
_DEFAULT_VIDEO_CRF = 20
_MIN_VIDEO_CRF = 0
_MAX_VIDEO_CRF = 51
# Temporary clips are encoded again by the concat and final passes. Keep them
# visibly cleaner than the delivered file so quality loss does not stack up.
_INTERMEDIATE_CRF_OFFSET = 3
_FINAL_PRESET = "medium"
_INTERMEDIATE_PRESET = "fast"
_DEFAULT_RENDER_TIMEOUT_FLOOR_SECONDS = 900.0
_DEFAULT_RENDER_TIMEOUT_PER_SECOND = 30.0


def _get_subtitle_spring_scale(time_seconds: float, duration_seconds: float) -> float:
    """返回字幕弹跳动画在指定时间点使用的缩放比例。"""
    if duration_seconds <= 0 or time_seconds >= duration_seconds:
        return 1.0

    progress = max(0.0, min(time_seconds / duration_seconds, 1.0))
    scale = 1.0 - math.exp(-6.0 * progress) * math.cos(2.5 * math.pi * progress)
    return max(
        _MIN_SUBTITLE_SPRING_SCALE,
        min(scale, _MAX_SUBTITLE_SPRING_SCALE),
    )


def _scale_subtitle_frame_on_canvas(frame: np.ndarray, scale: float) -> np.ndarray:
    """
    在保持画布尺寸不变的前提下，围绕中心缩放字幕画面或透明蒙版。

    MoviePy 将字幕颜色帧和透明蒙版分开保存。弹跳动画必须对二者使用完全
    相同的缩放与裁剪，否则动画首帧会把透明区域当成黑色文字轮廓合成到视频
    上。二维数组表示取值为 0～1 的蒙版，三维数组表示 RGB/RGBA 颜色帧。
    """
    if frame.ndim not in (2, 3):
        raise ValueError("subtitle frame must be a 2D mask or 3D color frame")

    height, width = frame.shape[:2]
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    offset = ((width - scaled_width) // 2, (height - scaled_height) // 2)

    if frame.ndim == 2:
        # MoviePy 蒙版使用 0～1 浮点数，Pillow 的 L 模式使用 0～255；转换后
        # 再恢复原始类型和范围，确保 CompositeVideoClip 的透明度语义不变。
        mask_image = Image.fromarray(
            np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        )
        resized_mask = mask_image.resize(
            (scaled_width, scaled_height),
            Image.Resampling.BILINEAR,
        )
        mask_canvas = Image.new("L", (width, height), 0)
        mask_canvas.paste(resized_mask, offset)
        return (np.asarray(mask_canvas) / 255.0).astype(frame.dtype, copy=False)

    if frame.shape[2] not in (3, 4):
        raise ValueError("subtitle color frame must use RGB or RGBA channels")
    color_image = Image.fromarray(frame)
    resized_color = color_image.resize(
        (scaled_width, scaled_height),
        Image.Resampling.BILINEAR,
    )
    background = (0, 0, 0, 0) if frame.shape[2] == 4 else (0, 0, 0)
    color_canvas = Image.new(color_image.mode, (width, height), background)
    color_canvas.paste(resized_color, offset)
    return np.asarray(color_canvas).astype(frame.dtype, copy=False)


def _apply_subtitle_spring_animation(clip, subtitle_duration: float):
    """同时缩放字幕颜色帧与蒙版，避免弹跳动画出现黑色首帧。"""
    animation_duration = min(
        _SUBTITLE_SPRING_DURATION_SECONDS,
        max(0.0, subtitle_duration),
    )
    if animation_duration <= 0:
        return clip

    def transform_frame(get_frame, time_seconds):
        frame = get_frame(time_seconds)
        scale = _get_subtitle_spring_scale(time_seconds, animation_duration)
        if scale == 1.0:
            return frame
        return _scale_subtitle_frame_on_canvas(frame, scale)

    # apply_to=["mask"] 是修复的关键：MoviePy 默认只处理颜色帧，旧实现因此
    # 在每条字幕出现时短暂保留原尺寸蒙版，并显示黑色文字轮廓。
    return clip.transform(transform_frame, apply_to=["mask"])


def _get_required_video_duration(audio_duration: float) -> float:
    """
    返回视频素材拼接的目标时长。

    使用场景：合成视频时需要素材时长覆盖旁白音频。只做到“刚好等于”
    音频时长时，FFmpeg 可能因为帧率舍入让最终视频略短，因此统一加一个
    轻量余量。函数独立出来，便于测试和后续按实际反馈调整余量大小。
    """
    return max(0.0, float(audio_duration) + _VIDEO_DURATION_SAFETY_MARGIN)


def is_material_resolution_acceptable(width: int, height: int) -> bool:
    """
    判断素材分辨率是否足够用于合成。

    标称最小值是 480x480，但允许比它低 `_MIN_DIMENSION_TOLERANCE` 个像素，
    以兼容编码器/消息应用向下取整导致的尺寸（例如 WhatsApp 的 478x850）。
    """
    min_dimension = _MIN_MATERIAL_DIMENSION - _MIN_DIMENSION_TOLERANCE
    return width >= min_dimension and height >= min_dimension


def _prioritize_unique_source_clips(
    subclipped_items: List[SubClippedVideoClip],
    concat_mode: VideoConcatMode,
    source_usage: dict[str, int] | None = None,
    source_groups: dict[str, str] | None = None,
) -> List[SubClippedVideoClip]:
    """
    优先让每个源素材只出现一次，降低成片里同一素材反复出现的概率。

    线上素材经常会遇到“一个长视频被切成多个短片段”的情况。旧逻辑在
    random 模式下直接打乱所有短片段，导致同一个源视频的多个切片可能
    分布在开头和中间，用户会感知为素材重复。本函数只调整片段顺序：
    先放每个源文件里最长的一个片段，剩余片段作为兜底；当素材总时长不足时，
    仍然允许后续片段补齐音频长度，避免破坏视频生成成功率。优先选择最长
    片段是为了避免随机选中视频尾部的零碎短片段，导致明明有足够素材却过早复用。
    """
    if not subclipped_items:
        return []

    concat_mode_value = getattr(concat_mode, "value", concat_mode)
    if concat_mode_value != VideoConcatMode.random.value:
        if source_usage is None:
            return subclipped_items
        if not source_groups:
            return sorted(
                subclipped_items,
                key=lambda item: source_usage.get(item.source_file_path, 0),
            )
        # Keep keyword rounds in order while rotating candidates within each keyword.
        groups = {}
        for item in subclipped_items:
            key = source_groups.get(item.source_file_path, item.source_file_path)
            groups.setdefault(key, []).append(item)
        for items in groups.values():
            items.sort(key=lambda item: source_usage.get(item.source_file_path, 0))
        return [
            item
            for row in itertools.zip_longest(*groups.values())
            for item in row
            if item is not None
        ]

    grouped_items: dict[str, list[SubClippedVideoClip]] = {}
    for item in subclipped_items:
        grouped_items.setdefault(item.source_file_path, []).append(item)

    primary_items = []
    overflow_items = []
    for items in grouped_items.values():
        primary_item = max(items, key=lambda item: item.duration)
        primary_items.append(primary_item)
        overflow_items.extend(item for item in items if item is not primary_item)

    random.shuffle(primary_items)
    random.shuffle(overflow_items)
    if source_usage is not None:
        # Stable sorting retains randomness among equally used sources.
        primary_items.sort(key=lambda item: source_usage.get(item.source_file_path, 0))
        overflow_items.sort(key=lambda item: source_usage.get(item.source_file_path, 0))
    logger.info(
        "prioritized unique video materials, "
        f"sources: {len(grouped_items)}, "
        f"primary clips: {len(primary_items)}, "
        f"fallback clips: {len(overflow_items)}"
    )
    return primary_items + overflow_items


def get_ffmpeg_binary():
    """
    兼容历史上直接从 video 服务读取 FFmpeg 路径的调用方。

    真正的解析逻辑已经抽到 `app.utils.utils.get_ffmpeg_binary()`，视频、语音
    和后续新增链路都应复用同一套优先级；这里保留薄包装，避免外部脚本或
    旧测试直接导入 `app.services.video.get_ffmpeg_binary` 时出现 AttributeError。
    """
    return utils.get_ffmpeg_binary()


def _get_configured_video_codec() -> str:
    """
    读取用户配置的视频编码器。

    该配置面向高级用户，用于尝试启用 NVENC/AMF/QSV/VideoToolbox 等硬件
    编码。这里刻意只允许固定白名单，避免开放任意 FFmpeg 参数后，用户填错
    参数导致输出格式不可控，甚至让生成任务在后续阶段才失败。
    """
    configured_codec = str(
        config.app.get("video_codec", _DEFAULT_VIDEO_CODEC) or _DEFAULT_VIDEO_CODEC
    ).strip()
    if configured_codec not in _SUPPORTED_VIDEO_CODECS:
        logger.warning(
            f"unsupported video codec configured: {configured_codec}, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC
    return configured_codec


@lru_cache(maxsize=16)
def _ffmpeg_encoder_exists(ffmpeg_binary: str, codec: str) -> bool:
    """
    检查当前 FFmpeg 是否声明支持指定编码器。

    这只能证明 FFmpeg 编译时包含该 encoder，不能证明当前机器硬件和驱动
    一定可用。因此实际编码失败时仍会再回退到 libx264。
    """
    try:
        result = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "failed to inspect ffmpeg encoders, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {str(exc)}"
        )
        return False

    if result.returncode != 0:
        logger.warning(
            "failed to inspect ffmpeg encoders, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {(result.stderr or result.stdout or '').strip()}"
        )
        return False
    return codec in result.stdout


def _get_effective_video_codec(preferred_codec: str | None = None) -> str:
    """
    返回本次实际使用的视频编码器。

    用户选择硬件编码器时，先做 FFmpeg encoder 列表检测；如果本进程里已经
    实际编码失败过，也直接回退，避免一个任务里每个片段都重复失败。
    """
    selected_codec = preferred_codec or _get_configured_video_codec()
    if selected_codec == _DEFAULT_VIDEO_CODEC:
        return _DEFAULT_VIDEO_CODEC

    if selected_codec in _runtime_disabled_video_codecs:
        logger.warning(
            f"video codec {selected_codec} was disabled after a runtime failure, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    ffmpeg_binary = utils.get_ffmpeg_binary()
    if not _ffmpeg_encoder_exists(ffmpeg_binary, selected_codec):
        logger.warning(
            f"ffmpeg encoder {selected_codec} is not available, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    return selected_codec


def _disable_runtime_video_codec(codec: str, reason: str):
    if codec == _DEFAULT_VIDEO_CODEC:
        return
    _runtime_disabled_video_codecs.add(codec)
    logger.warning(
        f"video codec {codec} failed, fallback to {_DEFAULT_VIDEO_CODEC}. "
        f"reason: {reason}"
    )


def _get_temp_audio_dir(output_dir: str) -> str:
    """
    Return the directory to use for MoviePy's temporary audio file.

    On Windows, Windows Defender can lock files written to the task output
    directory while scanning them, causing MoviePy to fail with a
    PermissionError (WinError 32) on the TEMP_MPY_wvf_snd temp file and
    leaving the final MP4 at 0 bytes.  Using the system temp directory
    sidesteps the scan without changing behaviour on other platforms.

    On Linux/macOS/Docker the output directory is returned unchanged so
    existing behaviour is preserved.
    """
    if sys.platform == "win32":
        return tempfile.gettempdir()
    return output_dir


class RenderTimeoutError(TimeoutError):
    """Raised when a MoviePy/FFmpeg render is killed by the render watchdog."""


_writer_tracking = threading.local()
_writer_tracking_install_lock = threading.Lock()


def _install_writer_tracking() -> None:
    """
    Make MoviePy's FFmpeg writer processes visible to the render watchdog.

    MoviePy keeps the encoder Popen handle private inside ``write_videofile``,
    so a hung encoder cannot be killed from outside. Wrapping the writer
    constructors once lets whichever thread is rendering register its own
    processes without touching any other thread's work.
    """
    with _writer_tracking_install_lock:
        try:
            from moviepy.audio.io import ffmpeg_audiowriter
            from moviepy.video.io import ffmpeg_writer
        except ImportError:  # pragma: no cover - MoviePy layout changed
            return
        for owner, name in (
            (ffmpeg_writer, "FFMPEG_VideoWriter"),
            (ffmpeg_audiowriter, "FFMPEG_AudioWriter"),
        ):
            cls = getattr(owner, name, None)
            if cls is None or getattr(cls.__init__, "_mpt_tracked", False):
                continue
            cls.__init__ = _track_writer_init(cls.__init__)


def _track_writer_init(original_init):
    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        tracker = getattr(_writer_tracking, "tracker", None)
        if tracker is not None:
            tracker.register(getattr(self, "proc", None))

    tracked_init._mpt_tracked = True
    return tracked_init


class _RenderWatchdog:
    """Kill the FFmpeg writers a render spawns once ``timeout`` seconds pass."""

    def __init__(self, timeout: float):
        self.timeout = timeout
        self.expired = threading.Event()
        self._processes = []
        self._lock = threading.Lock()
        self._timer = None

    def register(self, process) -> None:
        if process is None:
            return
        with self._lock:
            self._processes.append(process)
            expired = self.expired.is_set()
        if expired:
            self._kill(process)

    @staticmethod
    def _kill(process) -> None:
        try:
            process.kill()
        except Exception:  # the process may already be gone
            pass

    def _fire(self) -> None:
        self.expired.set()
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            self._kill(process)

    def __enter__(self):
        _install_writer_tracking()
        _writer_tracking.tracker = self
        self._timer = threading.Timer(self.timeout, self._fire)
        self._timer.daemon = True
        self._timer.start()
        return self

    def __exit__(self, *_exc):
        if self._timer is not None:
            self._timer.cancel()
        _writer_tracking.tracker = None
        return False


def _get_render_timeout_seconds(duration: float | None) -> float:
    """
    Watchdog limit for one MoviePy render.

    ``render_timeout_seconds`` in ``[app]`` is an absolute limit. Without it the
    limit scales with the clip length, because encode time is proportional to
    duration, and never drops below a floor that tolerates slow machines.
    """
    configured = config.app.get("render_timeout_seconds")
    if configured not in (None, "", 0):
        try:
            value = float(configured)
        except (TypeError, ValueError):
            value = 0.0
        if math.isfinite(value) and value > 0 and not isinstance(configured, bool):
            return value
        logger.warning(
            f"invalid render_timeout_seconds={configured!r}, using the scaled default"
        )
    try:
        seconds = float(duration or 0)
    except (TypeError, ValueError):
        seconds = 0.0
    if not math.isfinite(seconds) or seconds < 0:
        seconds = 0.0
    return max(
        _DEFAULT_RENDER_TIMEOUT_FLOOR_SECONDS,
        seconds * _DEFAULT_RENDER_TIMEOUT_PER_SECOND + 300.0,
    )


def _get_video_crf() -> int:
    configured = config.app.get("video_crf", _DEFAULT_VIDEO_CRF)
    if isinstance(configured, bool):
        return _DEFAULT_VIDEO_CRF
    try:
        crf = int(configured)
    except (TypeError, ValueError):
        return _DEFAULT_VIDEO_CRF
    return max(_MIN_VIDEO_CRF, min(_MAX_VIDEO_CRF, crf))


def _quality_ffmpeg_params(codec: str, intermediate: bool = False) -> list[str]:
    """
    Constant-quality encoder flags equivalent to the libx264 CRF setting.

    Hardware encoders have no CRF; each family gets its closest constant
    quality control. They fall back to libx264 on failure, which recomputes
    its own flags, so none of these ever reach another encoder.
    """
    crf = _get_video_crf()
    if intermediate:
        crf = max(_MIN_VIDEO_CRF, crf - _INTERMEDIATE_CRF_OFFSET)
    if codec == "h264_nvenc":
        params = ["-rc", "vbr", "-cq", str(min(51, crf + 3)), "-b:v", "0"]
    elif codec == "h264_amf":
        params = [
            "-rc", "cqp", "-qp_i", str(min(51, crf + 2)), "-qp_p", str(min(51, crf + 4)),
        ]
    elif codec == "h264_qsv":
        params = ["-global_quality", str(min(51, crf + 3))]
    elif codec == "h264_videotoolbox":
        params = ["-q:v", str(max(1, min(100, round(100 - crf * 2.2))))]
    elif codec == _DEFAULT_VIDEO_CODEC:
        params = ["-crf", str(crf)]
    else:
        params = []
    if not intermediate:
        params += ["-movflags", "+faststart"]
    return params


def _prepare_render_kwargs(codec: str, kwargs: dict, intermediate: bool) -> dict:
    prepared = dict(kwargs)
    prepared["ffmpeg_params"] = list(prepared.get("ffmpeg_params") or []) + (
        _quality_ffmpeg_params(codec, intermediate)
    )
    if codec == _DEFAULT_VIDEO_CODEC:
        prepared.setdefault(
            "preset", _INTERMEDIATE_PRESET if intermediate else _FINAL_PRESET
        )
    return prepared


def _cleanup_failed_render(output_file: str, kwargs: dict) -> None:
    """Remove the partial video and MoviePy's temporary audio after a kill."""
    delete_files(output_file)
    temp_dir = kwargs.get("temp_audiofile_path") or os.path.dirname(
        os.path.abspath(output_file)
    )
    stem = os.path.splitext(os.path.basename(output_file))[0]
    try:
        leftovers = [
            os.path.join(temp_dir, name)
            for name in os.listdir(temp_dir)
            if name.startswith(f"{stem}TEMP_MPY_wvf_snd")
        ]
    except OSError:
        leftovers = []
    delete_files(leftovers)


def _write_videofile_once(
    clip, output_file: str, codec: str, intermediate: bool = False, **kwargs
):
    """One MoviePy render with quality flags and a kill-on-timeout watchdog."""
    timeout = _get_render_timeout_seconds(getattr(clip, "duration", None))
    prepared = _prepare_render_kwargs(codec, kwargs, intermediate)

    def timeout_error() -> RenderTimeoutError:
        _cleanup_failed_render(output_file, kwargs)
        return RenderTimeoutError(
            f"video render exceeded {timeout:g} seconds and was stopped; "
            "raise [app].render_timeout_seconds for very long or slow renders"
        )

    with _RenderWatchdog(timeout) as watchdog:
        try:
            clip.write_videofile(output_file, codec=codec, **prepared)
        except Exception as exc:
            if watchdog.expired.is_set():
                raise timeout_error() from exc
            raise
    if watchdog.expired.is_set():
        raise timeout_error()


def _fallback_write_videofile(
    clip, output_file: str, failed_codec: str, reason: str, **kwargs
):
    """
    硬件编码失败后用 libx264 重试，只有重试成功才禁用该硬件编码器。

    Windows 上 FFmpeg 失败原因比较复杂：可能是显卡/驱动不支持，也可能是输出
    文件被占用、目录权限、杀软拦截等通用 IO 问题。只有 libx264 能成功写出时，
    才能判断原始失败大概率来自硬件编码器本身，避免误伤后续任务。
    """
    _write_videofile_once(clip, output_file, _DEFAULT_VIDEO_CODEC, **kwargs)
    _disable_runtime_video_codec(failed_codec, reason)
    return _DEFAULT_VIDEO_CODEC


def _write_videofile_with_codec_fallback(
    clip, output_file: str, codec: str, atomic_output: bool = False, **kwargs
):
    """
    使用指定编码器写出视频，失败时自动用 libx264 重试一次。

    硬件编码器是否可用不仅取决于 FFmpeg，还取决于显卡、驱动和当前运行环境。
    生成任务不能因为高级编码器不可用而整体失败，所以这里把回退集中处理。
    渲染超时（RenderTimeoutError）不会触发回退：卡死的编码器不能证明另一个
    编码器可用，重试只会让用户再等一个超时周期。
    """
    kwargs.setdefault("ffmpeg_params", _BT709_FFMPEG_PARAMS)
    if atomic_output:
        # Final videos can be downloaded by path while they are being rendered.
        # Keep both failed encodes and in-progress writes away from that path.
        output_dir = os.path.dirname(os.path.abspath(output_file))
        descriptor, temp_output = tempfile.mkstemp(
            prefix=f".{os.path.basename(output_file)}.",
            suffix=os.path.splitext(output_file)[1] or ".mp4",
            dir=output_dir,
        )
        os.close(descriptor)
        os.unlink(temp_output)
        try:
            used_codec = _write_videofile_with_codec_fallback(
                clip, temp_output, codec, **kwargs
            )
            os.replace(temp_output, output_file)
            return used_codec
        finally:
            try:
                os.unlink(temp_output)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(
                    f"failed to remove temporary final video: {temp_output}, "
                    f"error: {exc}"
                )

    effective_codec = _get_effective_video_codec(codec)
    try:
        _write_videofile_once(clip, output_file, effective_codec, **kwargs)
        return effective_codec
    except RenderTimeoutError:
        raise
    except Exception as exc:
        if effective_codec == _DEFAULT_VIDEO_CODEC:
            raise
        return _fallback_write_videofile(
            clip,
            output_file,
            failed_codec=effective_codec,
            reason=str(exc),
            **kwargs,
        )


def _escape_ffmpeg_concat_path(file_path: str) -> str:
    # concat demuxer 使用单引号包裹路径，路径中的单引号需要先转义。
    return file_path.replace("'", "'\\''")


def _format_ffmpeg_concat_path(file_path: str) -> str:
    """
    生成 concat demuxer 文件列表中的路径。

    FFmpeg 官方文档要求 concat list 中的特殊字符和空格需要转义；Windows
    绝对路径里的反斜杠也容易被解析成转义字符。这里统一转成正斜杠形式，
    让 `C:\\Users\\...` 变成 `C:/Users/...`，再处理单引号，兼容 macOS/Linux。
    """
    absolute_path = os.path.abspath(file_path)
    # Normalize the native separator only. A backslash can be a literal
    # component of a valid POSIX path and must still identify the same file.
    return _escape_ffmpeg_concat_path(absolute_path.replace(os.sep, "/"))


def _describe_concat_output_progress(output_file: str) -> str:
    """返回输出文件当前大小的可读描述，用于拼接心跳日志。"""
    try:
        size = os.path.getsize(output_file)
    except OSError:
        # 输出文件尚未创建时同样要安全降级，不能影响拼接本身。
        return "output size not available"
    return f"output size: {size / (1024 * 1024):.2f} MB"


@contextmanager
def _stage_heartbeat(description: str):
    """
    在一个没有自身日志的耗时阶段期间，按间隔记录存活日志。

    心跳线程绑定到进入该阶段的线程，日志才会出现在所属任务的 WebUI 面板。
    阶段结束或抛出异常时停止并等待心跳线程退出，避免阶段已经结束后仍然
    写出“still running”。
    """
    started_at = time.monotonic()
    stop_event = threading.Event()

    def log_heartbeat() -> None:
        while not stop_event.wait(_STAGE_HEARTBEAT_SECONDS):
            logger.info(
                f"{description} still running: "
                f"elapsed={time.monotonic() - started_at:.0f}s"
            )

    reporter = threading.Thread(
        target=logging_utils.bind_log_scope(log_heartbeat), daemon=True
    )
    reporter.start()
    try:
        yield
    finally:
        stop_event.set()
        reporter.join(timeout=1)


def _report_clip_progress(
    progress_callback: Callable[[float], None] | None,
    covered_duration: float,
    required_duration: float,
) -> None:
    """把已覆盖的成片时长换算成 0~1 的比例并通知调用方。"""
    if progress_callback is None:
        return
    fraction = 1.0
    if required_duration > 0:
        fraction = min(1.0, covered_duration / required_duration)
    try:
        progress_callback(fraction)
    except Exception as exc:
        # 进度只是展示信息。回调失败（例如状态后端暂时不可用）不能让已经
        # 处理好的片段作废。
        logger.warning(
            "failed to report clip processing progress: "
            f"error={type(exc).__name__}, detail={exc}"
        )


def _run_concat_with_heartbeat(command: list[str], output_file: str):
    """
    阻塞等待 ffmpeg 完成，期间按间隔记录存活日志。

    ffmpeg 串联片段时没有阶段日志，`subprocess.run` 又把输出缓冲到进程退出，耗时拼接
    在日志上表现为“无输出”。记录已等待时长与输出文件大小，便于区分仍在编码与已经卡死。
    """
    started_at = time.monotonic()
    stop_event = threading.Event()

    def log_heartbeat() -> None:
        while not stop_event.wait(_FFMPEG_CONCAT_HEARTBEAT_SECONDS):
            logger.info(
                "ffmpeg concat still running: "
                f"elapsed={time.monotonic() - started_at:.0f}s, "
                f"{_describe_concat_output_progress(output_file)}"
            )

    reporter = threading.Thread(
        target=logging_utils.bind_log_scope(log_heartbeat), daemon=True
    )
    reporter.start()
    try:
        configured_timeout = config.app.get(
            "ffmpeg_concat_timeout_seconds", _DEFAULT_FFMPEG_CONCAT_TIMEOUT_SECONDS
        )
        try:
            timeout_seconds = float(configured_timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("ffmpeg_concat_timeout_seconds must be positive") from exc
        if (
            isinstance(configured_timeout, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("ffmpeg_concat_timeout_seconds must be positive")

        try:
            # subprocess.run kills and waits for FFmpeg on timeout, so a stalled
            # encoder cannot leave an orphaned child or a permanently active task.
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"ffmpeg concat exceeded {timeout_seconds:g} seconds"
            ) from exc
    finally:
        stop_event.set()


_MAX_CROSSFADE_CLIPS = 60
_MIN_CROSSFADE_SECONDS = 0.05


def _probe_media_duration(file_path: str) -> float | None:
    """Read a clip's real duration from FFmpeg's banner (no ffprobe needed)."""
    try:
        result = subprocess.run(
            [utils.get_ffmpeg_binary(), "-hide_banner", "-i", file_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr or "")
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _plan_crossfade(clip_files: List[str], requested_seconds: float):
    """
    Return ``(overlap, durations)`` for an xfade chain, or None for hard cuts.

    Offsets must follow each file's real length: frame rounding makes the
    encoded clips differ slightly from their nominal durations, and an offset
    past the end of its input would freeze or drop frames.
    """
    try:
        overlap = float(requested_seconds or 0)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(overlap)
        or overlap < _MIN_CROSSFADE_SECONDS
        or not 2 <= len(clip_files) <= _MAX_CROSSFADE_CLIPS
    ):
        return None
    durations = []
    for clip_file in clip_files:
        duration = _probe_media_duration(clip_file)
        if duration is None or duration <= 0:
            return None
        durations.append(duration)
    overlap = min(overlap, min(durations) / 3.0)
    if overlap < _MIN_CROSSFADE_SECONDS:
        return None
    return overlap, durations


def _crossfade_input_arguments(clip_files: List[str], plan) -> list[str]:
    overlap, durations = plan
    command = [utils.get_ffmpeg_binary(), "-y"]
    for clip_file in clip_files:
        command.extend(["-i", clip_file])
    steps = []
    previous = "[0:v]"
    elapsed = durations[0]
    for index in range(1, len(clip_files)):
        offset = max(0.0, elapsed - index * overlap)
        label = f"[x{index}]"
        steps.append(
            f"{previous}[{index}:v]xfade=transition=fade:"
            f"duration={overlap:.3f}:offset={offset:.3f}{label}"
        )
        previous = label
        elapsed += durations[index]
    steps.append(f"{previous}{_BT709_VIDEO_FILTER}[vout]")
    command.extend(["-filter_complex", ";".join(steps), "-map", "[vout]"])
    return command


def concat_video_clips_with_ffmpeg(
    clip_files: List[str],
    output_file: str,
    threads: int,
    output_dir: str,
    max_duration: float | None = None,
    crossfade_seconds: float = 0.0,
):
    # Separate renders may share a directory. Each FFmpeg process must keep its
    # own manifest until all codec attempts finish, without overwriting or
    # deleting another render's list.
    concat_list_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="ffmpeg-concat-", suffix=".txt",
            dir=output_dir, delete=False,
        ) as fp:
            concat_list_file = fp.name
            for clip_file in clip_files:
                fp.write(f"file '{_format_ffmpeg_concat_path(clip_file)}'\n")
    except Exception:
        if concat_list_file:
            delete_files(concat_list_file)
        raise

    staged_output = None
    crossfade = _plan_crossfade(clip_files, crossfade_seconds)

    def build_command(codec: str) -> list[str]:
        if crossfade is not None:
            command = _crossfade_input_arguments(clip_files, crossfade) + [
                "-c:v",
                codec,
                "-threads",
                str(threads or 2),
                "-pix_fmt",
                "yuv420p",
            ]
        else:
            command = [
                utils.get_ffmpeg_binary(),
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_list_file,
                "-c:v",
                codec,
                "-threads",
                str(threads or 2),
                "-vf",
                _BT709_VIDEO_FILTER,
                "-pix_fmt",
                "yuv420p",
            ]
        if codec == _DEFAULT_VIDEO_CODEC:
            command.extend(["-preset", _INTERMEDIATE_PRESET])
        command.extend(_quality_ffmpeg_params(codec, intermediate=True))
        if max_duration is not None and max_duration > 0:
            command.extend(["-t", f"{max_duration:.3f}"])
        command.append(staged_output)
        return command

    def run_concat(codec: str):
        command = build_command(codec)
        # 使用 ffmpeg 只做一次串联与编码，避免 MoviePy 逐段合并时反复重编码，
        # 从而降低画质劣化与颜色偏移风险。阻塞等待期间由心跳日志体现任务仍在运行。
        result = _run_concat_with_heartbeat(command, staged_output)
        if result.returncode != 0:
            error_message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(error_message or "ffmpeg concat failed")
        return codec

    try:
        descriptor, staged_output = tempfile.mkstemp(
            prefix=".ffmpeg-concat-",
            suffix=os.path.splitext(output_file)[1] or ".mp4",
            dir=os.path.dirname(os.path.abspath(output_file)),
        )
        os.close(descriptor)
        effective_codec = _get_effective_video_codec()
        try:
            result_codec = run_concat(effective_codec)
        except TimeoutError:
            # A hung encoder is not evidence that another codec will work. Do
            # not spend a second timeout period retrying the same input.
            raise
        except Exception as exc:
            if effective_codec == _DEFAULT_VIDEO_CODEC:
                raise
            result_codec = run_concat(_DEFAULT_VIDEO_CODEC)
            _disable_runtime_video_codec(effective_codec, str(exc))
        # Failed attempts and in-progress output stay private until FFmpeg has
        # finished. Publication errors must not trigger another codec attempt.
        if os.path.getsize(staged_output) == 0:
            raise RuntimeError("ffmpeg concat produced no output")
        os.replace(staged_output, output_file)
        return result_codec
    finally:
        delete_files([concat_list_file, staged_output])


def _sanitize_image_file(image_path: str) -> str:
    # 某些本地图片虽然能被 Pillow 打开，但会因为损坏的 EXIF/eXIf 元数据导致
    # ImageClip 在解析阶段直接抛异常。这里重新导出一份“干净图片”，把坏元数据剥离掉。
    # Bound the intermediate basename while retaining complete source identity.
    # Different extensions must not overwrite pixels, and valid long filenames
    # must not exceed the filesystem's component limit after adding a suffix.
    source_digest = hashlib.sha256(os.fsencode(os.path.abspath(image_path)))

    temp_path = ""
    try:
        with Image.open(image_path) as image:
            with ImageOps.exif_transpose(image) as upright:
                upright.load()
                # Strip metadata after applying the camera's orientation.
                # Palette transparency belongs to pixels, not removable metadata.
                mode = "RGBA" if "A" in upright.getbands() or "transparency" in upright.info else "RGB"
                cleaned_image = upright.convert(mode)
                cleaned_image.info.clear()
                source_digest.update(str(cleaned_image.size).encode())
                source_digest.update(cleaned_image.mode.encode())
                source_digest.update(cleaned_image.tobytes())
                sanitized_path = os.path.join(
                    os.path.dirname(image_path),
                    f"{source_digest.hexdigest()}.sanitized.png",
                )
                descriptor, temp_path = tempfile.mkstemp(
                    prefix=".image-sanitize-", suffix=".png",
                    dir=os.path.dirname(os.path.abspath(sanitized_path)),
                )
                os.close(descriptor)
                cleaned_image.save(temp_path)
                cleaned_image.close()
        os.replace(temp_path, sanitized_path)
        temp_path = ""
    finally:
        if temp_path:
            delete_files(temp_path)

    return sanitized_path


def _open_image_clip_with_fallback(image_path: str):
    # MoviePy does not apply camera EXIF orientation while decoding an image.
    # CMYK JPEG channels must become RGB; LA must become RGBA so MoviePy
    # creates a proper opacity mask rather than treating alpha as a color band.
    # Ordinary RGB inputs retain the direct path.
    try:
        with Image.open(image_path) as image:
            orientation = image.getexif().get(274, 1)
            image_mode = image.mode
    except Exception:
        orientation = 1
        image_mode = None
    if orientation in range(2, 9) or image_mode in {"CMYK", "LA"}:
        sanitized_path = _sanitize_image_file(image_path)
        return ImageClip(sanitized_path), sanitized_path

    # 优先直接打开原始图片；如果因为损坏元数据失败，再尝试生成无元数据副本。
    try:
        return ImageClip(image_path), image_path
    except Exception as exc:
        logger.warning(
            f"failed to open image directly, trying sanitized copy: {image_path}, error: {str(exc)}"
        )
        sanitized_path = _sanitize_image_file(image_path)
        return ImageClip(sanitized_path), sanitized_path


_moviepy_reader_open_lock = threading.Lock()


def _open_video_clip_quietly(video_path: str, audio: bool = False) -> VideoFileClip:
    """
    安静地打开视频文件，避免 MoviePy 2.1.x 把 ffmpeg 探测信息直接打印到 stdout。

    背景：
    当前依赖版本的 `FFMPEG_VideoReader` 内部存在 `print(self.infos)` 和
    `print(ffmpeg command)`，读取无音轨的中间视频时会输出
    `audio_found: False`。这只是输入素材 metadata，不代表最终成片没有音频，
    但会误导 WebUI/终端用户以为生成失败。

    实现：
    1. 只在打开 VideoFileClip 的短窗口内重定向 stdout；
    2. 默认 `audio=False`，因为项目视频素材阶段不需要保留素材原声，
       最终音频会在 `generate_video()` 阶段统一挂载；
    3. 如果依赖库确实输出了内容，降级为 debug 日志，便于必要时排查。
    """
    captured_stdout = io.StringIO()
    # redirect_stdout changes process-wide state. Overlapping reader opens can
    # restore each other's capture buffers instead of the original stdout.
    # Serialize this short construction window; clip processing stays parallel.
    with _moviepy_reader_open_lock:
        with redirect_stdout(captured_stdout):
            clip = VideoFileClip(video_path, audio=audio)

    moviepy_stdout = captured_stdout.getvalue().strip()
    if moviepy_stdout:
        logger.debug(
            "suppressed MoviePy video reader stdout for "
            f"{video_path}, chars: {len(moviepy_stdout)}"
        )

    return clip


def close_clip(clip):
    if clip is None:
        return
        
    try:
        # close main resources
        if hasattr(clip, 'reader') and clip.reader is not None:
            clip.reader.close()
            
        # close audio resources
        if hasattr(clip, 'audio') and clip.audio is not None:
            if hasattr(clip.audio, 'reader') and clip.audio.reader is not None:
                clip.audio.reader.close()
            del clip.audio
            
        # close mask resources
        if hasattr(clip, 'mask') and clip.mask is not None:
            if hasattr(clip.mask, 'reader') and clip.mask.reader is not None:
                clip.mask.reader.close()
            del clip.mask
            
        # handle child clips in composite clips
        if hasattr(clip, 'clips') and clip.clips:
            for child_clip in clip.clips:
                if child_clip is not clip:  # avoid possible circular references
                    close_clip(child_clip)
            
        # clear clip list
        if hasattr(clip, 'clips'):
            clip.clips = []
            
    except Exception as e:
        logger.error(f"failed to close clip: {str(e)}")
    
    del clip
    gc.collect()

def delete_files(files: List[str] | str):
    if isinstance(files, str):
        files = [files]

    # 循环补足视频时，同一个临时片段路径会在 FFmpeg 拼接列表中出现多次。
    # 拼接必须保留重复项，但清理只能删除一次；这里按原顺序统一去重，让所有
    # 调用方都获得幂等行为，也避免首次删除成功后连续输出 FileNotFoundError。
    unique_files = dict.fromkeys(file for file in files if file)
    for file in unique_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            # 清理动作允许文件已经不存在，例如 FFmpeg 失败路径或并发清理已经
            # 回收文件；这不是需要用户处理的问题，不应污染生成日志。
            continue
        except OSError as e:
            # 权限、只读文件系统或磁盘异常会留下真实临时文件，保留 warning
            # 便于根据具体路径和系统错误定位环境问题。
            logger.warning(f"failed to delete temporary file {file}: {str(e)}")


def get_bgm_file(bgm_type: str = "random", bgm_file: str = ""):
    if not bgm_type:
        return ""

    if bgm_file:
        try:
            resolved_bgm_file = bgm_service.resolve_bgm_file(bgm_file)
        except ValueError as exc:
            # API 请求里的 bgm_file 来自用户输入，只允许解析到用户 BGM 或内置
            # 歌曲目录，阻止 MoviePy 读取配置、密钥等任意服务器文件。
            logger.warning(
                f"reject unsafe bgm file: {bgm_file}, error: {str(exc)}"
            )
            return ""
        return resolved_bgm_file

    if bgm_type == "random":
        files = bgm_service.list_bgm_files()
        # 当背景音乐目录为空时，直接回退为“不使用 BGM”，避免 random.choice([]) 抛异常。
        if not files:
            logger.warning("no background music files found")
            return ""
        return random.choice(files)

    return ""


def _fit_clip_to_canvas(
    clip,
    *,
    target_width: int,
    target_height: int,
    fit_mode: VideoFitMode | str = VideoFitMode.cover,
):
    """Resize a clip to an exact canvas using cover/crop or contain/letterbox."""
    source_width, source_height = (int(value) for value in clip.size)
    target_width = int(target_width)
    target_height = int(target_height)
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError(
            "video dimensions must be positive: "
            f"source={source_width}x{source_height}, "
            f"target={target_width}x{target_height}"
        )

    mode = VideoFitMode(fit_mode)
    if (source_width, source_height) == (target_width, target_height):
        return clip

    # Exact aspect-ratio matches do not need either a crop or a background.
    if source_width * target_height == source_height * target_width:
        return clip.resized(new_size=(target_width, target_height))

    width_scale = target_width / source_width
    height_scale = target_height / source_height

    if mode == VideoFitMode.cover:
        # ceil guarantees the resized clip covers the complete canvas despite
        # floating-point rounding. Any excess is removed symmetrically.
        scale_factor = max(width_scale, height_scale)
        resized_width = max(target_width, math.ceil(source_width * scale_factor))
        resized_height = max(target_height, math.ceil(source_height * scale_factor))
        resized_clip = clip.resized(new_size=(resized_width, resized_height))
        crop_x = max(0, (resized_width - target_width) // 2)
        crop_y = max(0, (resized_height - target_height) // 2)
        return resized_clip.cropped(
            x1=crop_x,
            y1=crop_y,
            width=target_width,
            height=target_height,
        )

    # contain preserves the legacy behavior: show the complete source frame,
    # centered over a black canvas when the aspect ratios differ.
    scale_factor = min(width_scale, height_scale)
    resized_width = max(1, min(target_width, int(source_width * scale_factor)))
    resized_height = max(1, min(target_height, int(source_height * scale_factor)))
    background = ColorClip(
        size=(target_width, target_height), color=(0, 0, 0)
    ).with_duration(clip.duration)
    resized_clip = clip.resized(
        new_size=(resized_width, resized_height)
    ).with_position("center")
    return CompositeVideoClip(
        [background, resized_clip], size=(target_width, target_height)
    ).with_duration(clip.duration)


_DEFAULT_CROSSFADE_SECONDS = 0.3

# Filters applied to repeated clips so a loop does not replay identical frames.
# Each round walks to the next entry: mirrored, zoomed to one corner, etc.
_REPEAT_VARIANT_FILTERS = (
    "hflip",
    "crop=iw*0.86:ih*0.86:0:0,scale={w}:{h}",
    "crop=iw*0.86:ih*0.86:iw*0.14:ih*0.14,scale={w}:{h}",
    "hflip,crop=iw*0.9:ih*0.9:iw*0.05:ih*0.05,scale={w}:{h}",
)


def _get_default_crossfade_seconds() -> float:
    configured = config.app.get("video_crossfade_seconds", _DEFAULT_CROSSFADE_SECONDS)
    if isinstance(configured, bool):
        return 0.0
    try:
        seconds = float(configured)
    except (TypeError, ValueError):
        return _DEFAULT_CROSSFADE_SECONDS
    if not math.isfinite(seconds) or seconds <= 0:
        return 0.0
    return min(seconds, 2.0)


def _make_repeat_variant(
    clip: SubClippedVideoClip,
    repeat_round: int,
    output_dir: str,
    width: int,
    height: int,
) -> SubClippedVideoClip | None:
    """
    Re-encode ``clip`` as a mirrored/re-cropped variant for a repeated slot.

    Returns None when the variant cannot be produced; the caller then repeats
    the original file, which is always a safe fallback.
    """
    video_filter = _REPEAT_VARIANT_FILTERS[
        repeat_round % len(_REPEAT_VARIANT_FILTERS)
    ].format(w=int(width), h=int(height))
    variant_file = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_dir or ".", prefix="temp-clip-", suffix=".mp4", delete=False
        ) as handle:
            variant_file = handle.name
        command = [
            utils.get_ffmpeg_binary(), "-y", "-hide_banner", "-loglevel", "error",
            "-i", clip.file_path,
            "-vf", f"{video_filter},setsar=1",
            "-an", "-c:v", _DEFAULT_VIDEO_CODEC, "-preset", _INTERMEDIATE_PRESET,
            "-pix_fmt", "yuv420p",
            *_quality_ffmpeg_params(_DEFAULT_VIDEO_CODEC, intermediate=True),
            variant_file,
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False, timeout=300,
        )
        if result.returncode != 0 or os.path.getsize(variant_file) == 0:
            raise RuntimeError((result.stderr or "").strip() or "empty variant")
        return SubClippedVideoClip(
            file_path=variant_file,
            duration=clip.duration,
            width=clip.width,
            height=clip.height,
            source_file_path=clip.source_file_path,
        )
    except Exception as exc:
        logger.warning(f"could not create a repeat variant, reusing clip: {exc}")
        if variant_file:
            delete_files(variant_file)
        return None


def combine_videos(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
    clip_speed: float = 1.0,
    video_fit_mode: VideoFitMode = VideoFitMode.cover,
    source_usage: dict[str, int] | None = None,
    source_groups: dict[str, str] | None = None,
    used_video_paths: List[str] | None = None,
    progress_callback: Callable[[float], None] | None = None,
    warnings: list | None = None,
) -> str:
    # Pydantic leaves this optional default as the string "random" unless
    # explicitly provided. Normalize it once for all downstream enum access.
    video_concat_mode = VideoConcatMode(video_concat_mode or VideoConcatMode.random)
    audio_clip = AudioFileClip(audio_file)
    try:
        # 这里只需要读取旁白音频时长来决定素材视频拼接长度；后续不会再使用
        # audio_clip。读取完成后立即关闭，避免早退或异常路径泄漏文件句柄。
        audio_duration = audio_clip.duration
    finally:
        close_clip(audio_clip)
    logger.info(f"audio duration: {audio_duration} seconds")
    logger.info(f"maximum clip duration: {max_clip_duration} seconds")
    required_video_duration = _get_required_video_duration(audio_duration)
    logger.info(
        f"required video duration: {required_video_duration:.2f} seconds "
        f"(audio duration + {_VIDEO_DURATION_SAFETY_MARGIN:.2f}s safety margin)"
    )

    # 兼容 API 直接调用时未传转场模式的情况，避免后续访问 .value 时崩溃。
    transition_value = getattr(video_transition_mode, "value", video_transition_mode)
    # With no explicit transition, neighbouring clips dissolve into each other
    # instead of hard-cutting. Explicit transitions keep their own behaviour.
    crossfade_seconds = (
        _get_default_crossfade_seconds() if not transition_value else 0.0
    )
    normalized_clip_speed = utils.normalize_clip_speed(clip_speed)
    if normalized_clip_speed != 1.0:
        # 只记录一次最终生效值，既方便定位 API 越界参数被归一化的问题，
        # 也避免在逐片段热路径中重复输出相同日志。
        logger.info(f"clip playback speed: {normalized_clip_speed:.2f}x")
    # max_clip_duration 约束的是成片里的最终播放时长，而不是源视频读取时长。
    # MoviePy 以 0.5 倍速播放 1.5 秒源画面会得到 3 秒片段，以 2 倍速播放
    # 6 秒源画面同样会得到 3 秒片段。因此切片前必须按速度反推源时长；如果
    # 仍固定读取 3 秒再慢放、裁剪，下一段却从源视频第 3 秒开始，会跳过中间
    # 1.5 秒画面。该计算同时保证不同速度下的源时间线连续且无重叠。
    source_clip_duration = max_clip_duration * normalized_clip_speed
    output_dir = os.path.dirname(combined_video_path)

    aspect = VideoAspect(video_aspect)
    fit_mode = VideoFitMode(video_fit_mode)
    video_width, video_height = aspect.to_resolution()

    processed_clips = []
    subclipped_items = []
    source_infos = []
    video_duration = 0
    for video_path in video_paths:
        clip = None
        try:
            clip = _open_video_clip_quietly(video_path)
            clip_duration = float(clip.duration)
            clip_w, clip_h = clip.size
            if (
                not math.isfinite(clip_duration)
                or clip_duration <= 0
                or not all(
                    math.isfinite(float(dimension)) and float(dimension) > 0
                    for dimension in (clip_w, clip_h)
                )
            ):
                raise ValueError("invalid video duration or dimensions")
        except Exception as exc:
            logger.warning(
                f"skipping unreadable video source: path={video_path}, error={exc}"
            )
            continue
        finally:
            close_clip(clip)
        
        start_time = 0
        source_infos.append((video_path, clip_w, clip_h, clip_duration))

        while start_time < clip_duration:
            end_time = min(start_time + source_clip_duration, clip_duration)

            # 保留所有有效分段。
            # 这样既不会丢掉“整段视频本身就短于 max_clip_duration”的素材，
            # 也不会吞掉长视频最后剩下的一小段尾部内容。
            if end_time > start_time:
                subclipped_items.append(
                    SubClippedVideoClip(
                        file_path=video_path,
                        start_time=start_time,
                        end_time=end_time,
                        width=clip_w,
                        height=clip_h,
                        source_file_path=video_path,
                    )
                )

            start_time = end_time
            if video_concat_mode.value == VideoConcatMode.sequential.value:
                break

    subclipped_items = _prioritize_unique_source_clips(
        subclipped_items=subclipped_items,
        concat_mode=video_concat_mode,
        **({"source_usage": source_usage, "source_groups": source_groups}
           if source_usage is not None else {}),
    )
        
    logger.debug(f"total subclipped items: {len(subclipped_items)}")
    
    # Add downloaded clips over and over until the duration of the audio (max_duration) has been reached
    def process_one_clip(indexed_item):
        """把一个源片段裁剪、变速、转场后写出，失败时返回 None。"""
        index, subclipped_item = indexed_item
        source_clip = None
        clip = None
        clip_file = None
        try:
            logger.debug(
                f"processing clip {index + 1}: {subclipped_item.width}x{subclipped_item.height}, "
                f"source: {os.path.basename(subclipped_item.source_file_path)}"
            )
            source_clip = _open_video_clip_quietly(subclipped_item.file_path)
            clip = source_clip.subclipped(
                subclipped_item.start_time, subclipped_item.end_time
            )
            # 播放速度属于素材本身属性，应在转场前应用。这样 Fade/Slide 等一秒转场
            # 不会跟随素材速度变成 0.5 秒或 2 秒；后续最大时长裁剪继续作为
            # 浮点误差或异常素材时长的安全兜底，保证最终片段不突破配置上限。
            if normalized_clip_speed != 1.0:
                clip = clip.with_speed_scaled(normalized_clip_speed)
            # Normalize every source clip before transitions are applied. In cover mode
            # the clip fills the canvas and the excess edges are cropped; contain keeps
            # the complete source frame and uses black bars for the unused area.
            clip_w, clip_h = clip.size
            if clip_w != video_width or clip_h != video_height:
                clip_ratio = clip.w / clip.h
                video_ratio = video_width / video_height
                logger.debug(
                    "resizing clip, "
                    f"source: {clip_w}x{clip_h}, ratio: {clip_ratio:.2f}, "
                    f"target: {video_width}x{video_height}, ratio: {video_ratio:.2f}, "
                    f"fit_mode: {fit_mode.value}"
                )
                clip = _fit_clip_to_canvas(
                    clip,
                    target_width=video_width,
                    target_height=video_height,
                    fit_mode=fit_mode,
                )

            shuffle_side = random.choice(["left", "right", "top", "bottom"])
            # Short footage and faster playback can leave less than one second.
            # Keep the chosen transition inside the rendered clip's timeline.
            transition_duration = min(1.0, clip.duration)
            if transition_value == VideoTransitionMode.fade_in.value:
                clip = video_effects.fadein_transition(clip, transition_duration)
            elif transition_value == VideoTransitionMode.fade_out.value:
                clip = video_effects.fadeout_transition(clip, transition_duration)
            elif transition_value == VideoTransitionMode.slide_in.value:
                clip = video_effects.slidein_transition(clip, transition_duration, shuffle_side)
            elif transition_value == VideoTransitionMode.slide_out.value:
                clip = video_effects.slideout_transition(clip, transition_duration, shuffle_side)
            elif transition_value == VideoTransitionMode.zoom_in.value:
                clip = video_effects.zoomin_transition(clip, 1)
            elif transition_value == VideoTransitionMode.zoom_out.value:
                clip = video_effects.zoomout_transition(clip, 1)
            elif transition_value == VideoTransitionMode.shuffle.value:
                transition_funcs = [
                    lambda c: video_effects.fadein_transition(c, transition_duration),
                    lambda c: video_effects.fadeout_transition(c, transition_duration),
                    lambda c: video_effects.slidein_transition(c, transition_duration, shuffle_side),
                    lambda c: video_effects.slideout_transition(c, transition_duration, shuffle_side),
                    lambda c: video_effects.zoomin_transition(c, 1),
                    lambda c: video_effects.zoomout_transition(c, 1),
                ]
                shuffle_transition = random.choice(transition_funcs)
                clip = shuffle_transition(clip)

            if clip.duration > max_clip_duration:
                clip = clip.subclipped(0, max_clip_duration)

            # Write each candidate clip to a unique temporary file. Threads must not
            # share the same output path.
            # Distinct combinations can share a task/output directory. Reserve
            # an owned path so their encoders and cleanup never share a clip.
            with tempfile.NamedTemporaryFile(
                dir=output_dir or ".", prefix="temp-clip-", suffix=".mp4", delete=False
            ) as temporary_clip:
                clip_file = temporary_clip.name
            _write_videofile_with_codec_fallback(
                clip,
                clip_file,
                codec=_get_configured_video_codec(),
                logger=None,
                fps=fps,
            )

            clip_duration_saved = clip.duration
            processed_clip = SubClippedVideoClip(
                file_path=clip_file,
                duration=clip_duration_saved,
                width=clip_w,
                height=clip_h,
                source_file_path=subclipped_item.source_file_path,
            )
            clip_file = None
            return processed_clip
        except Exception as exc:
            logger.error(f"failed to process clip: {str(exc)}")
            return None
        finally:
            # The derived clip shares its FFmpeg reader with the source. If
            # subclipping itself failed, close the original source instead.
            close_clip(clip if clip is not None else source_clip)
            # MoviePy may leave a truncated MP4 even when encoding raises. It
            # was never returned, so concat cleanup cannot see it.
            # Close the reader first so Windows can remove the partial file.
            if clip_file:
                delete_files(clip_file)

    # 片段始终在线程池里处理。逐片段日志是这一阶段唯一的进度信息，绑定到
    # 发起合成的线程后，WebUI 的任务日志才能收集到它们。
    process_clip_in_task_scope = logging_utils.bind_log_scope(process_one_clip)

    def covered_by(clip_seconds: float) -> float:
        # Crossfaded neighbours overlap, so every clip after the first only
        # adds its length minus the overlap to the final timeline.
        return clip_seconds - (crossfade_seconds if processed_clips else 0.0)

    def process_items(executor, items) -> None:
        nonlocal video_duration
        next_candidate_index = 0
        while (
            next_candidate_index < len(items)
            and video_duration < required_video_duration
        ):
            remaining_duration = required_video_duration - video_duration
            batch = []
            batch_duration = 0.0
            candidate_index = next_candidate_index
            while candidate_index < len(items) and batch_duration < remaining_duration:
                subclipped_item = items[candidate_index]
                source_duration = subclipped_item.end_time - subclipped_item.start_time
                output_duration = min(
                    max_clip_duration,
                    source_duration / normalized_clip_speed,
                )
                batch.append((candidate_index, subclipped_item))
                batch_duration += output_duration
                candidate_index += 1
            if not batch:
                break
            for processed_clip in executor.map(process_clip_in_task_scope, batch):
                if processed_clip is None:
                    continue
                video_duration += covered_by(processed_clip.duration)
                processed_clips.append(processed_clip)
                # 每段 4K 素材要处理十几秒，逐段报告覆盖时长，日志和进度条
                # 才能反映这一阶段仍在推进。
                logger.info(
                    f"processed clip {len(processed_clips)}: "
                    f"{video_duration:.1f} of {required_video_duration:.1f}s covered"
                )
                _report_clip_progress(
                    progress_callback, video_duration, required_video_duration
                )
            next_candidate_index = candidate_index

    clip_processing_workers = 1
    if len(subclipped_items) >= 2:
        clip_processing_workers = min(_get_clip_processing_concurrency(), len(subclipped_items))
    with ThreadPoolExecutor(
        max_workers=clip_processing_workers,
        thread_name_prefix="clip-process",
    ) as executor:
        process_items(executor, subclipped_items)
        if (
            video_duration < required_video_duration
            and video_concat_mode.value == VideoConcatMode.sequential.value
        ):
            # Sequential mode only schedules the first slice of each source.
            # Before repeating anything, use the unseen remainder of the footage.
            extra_items = []
            for source_path, source_w, source_h, source_length in source_infos:
                extra_start = source_clip_duration
                while extra_start < source_length:
                    extra_end = min(extra_start + source_clip_duration, source_length)
                    if extra_end > extra_start:
                        extra_items.append(
                            SubClippedVideoClip(
                                file_path=source_path,
                                start_time=extra_start,
                                end_time=extra_end,
                                width=source_w,
                                height=source_h,
                                source_file_path=source_path,
                            )
                        )
                    extra_start = extra_end
            if extra_items:
                logger.info(
                    f"footage is short; using {len(extra_items)} unused source "
                    "segments before repeating clips"
                )
                process_items(executor, extra_items)

    # Footage still shorter than the narration: repeat clips, but as visibly
    # different variants (mirrored / re-cropped) rather than identical frames.
    if video_duration < required_video_duration and processed_clips:
        logger.warning(
            f"video duration ({video_duration:.2f}s) is shorter than required duration "
            f"({required_video_duration:.2f}s), repeating clips to match audio length."
        )
        base_clips = processed_clips.copy()
        repeated_clips = 0
        variant_clips = 0
        for clip in itertools.cycle(base_clips):
            if video_duration >= required_video_duration:
                break
            repeat_round = repeated_clips // len(base_clips)
            variant = _make_repeat_variant(
                clip, repeat_round, output_dir, video_width, video_height
            )
            if variant is not None:
                variant_clips += 1
            repeated = variant or clip
            video_duration += covered_by(repeated.duration)
            processed_clips.append(repeated)
            repeated_clips += 1
        logger.info(
            f"video duration: {video_duration:.2f}s, audio duration: {audio_duration:.2f}s, "
            f"required duration: {required_video_duration:.2f}s, "
            f"repeated {repeated_clips} clips ({variant_clips} as variants)"
        )
        if warnings is not None:
            warnings.append(
                {
                    "code": "footage_repeated",
                    "repeated_clips": repeated_clips,
                    "variant_clips": variant_clips,
                    "message": (
                        "The footage was shorter than the narration, so "
                        f"{repeated_clips} clips were repeated. Add more or longer "
                        "footage to avoid repetition."
                    ),
                }
            )

    # merge video clips progressively, avoid loading all videos at once to avoid memory overflow
    logger.info("starting clip merging process")
    if not processed_clips:
        if video_paths:
            raise RuntimeError("no readable video clips available for merging")
        logger.warning("no clips available for merging")
        return combined_video_path
    
    clip_files = [clip.file_path for clip in processed_clips]
    logger.info(f"concatenating {len(clip_files)} clips with ffmpeg")
    try:
        concat_video_clips_with_ffmpeg(
            clip_files=clip_files,
            output_file=combined_video_path,
            threads=threads,
            output_dir=output_dir,
            max_duration=audio_duration,
            crossfade_seconds=crossfade_seconds,
        )
        if used_video_paths is not None:
            # Exclude safety-margin clips that FFmpeg trims entirely from the output.
            elapsed = 0.0
            for clip in processed_clips:
                if elapsed >= audio_duration:
                    break
                used_video_paths.append(clip.source_file_path)
                elapsed += clip.duration
    finally:
        # FFmpeg failures and timeouts must not strand one encoded MP4 per clip.
        # Repeated clips share a path; delete_files already deduplicates them.
        delete_files(clip_files)
            
    logger.info("video combining completed")
    return combined_video_path


def wrap_text(text, max_width, font="Arial", fontsize=60):
    # 字幕换行必须在真正创建 TextClip 前完成，否则 MoviePy 只会按原始文本
    # 计算渲染区域。这里用 PIL 按当前字体和字号测量宽度，确保每一行都尽量
    # 控制在视频可用宽度内，避免大字号或中文长句直接溢出画面。
    if "\n" in text:
        # Hard breaks in SRT text are separate layout lines. Measuring a token
        # across a newline makes Pillow count both lines as one wide string,
        # then character wrapping can split an otherwise fitting word.
        wrapped_lines = [
            wrap_text(line, max_width, font=font, fontsize=fontsize)
            for line in text.split("\n")
        ]
        return "\n".join(line for line, _ in wrapped_lines), sum(
            height for _, height in wrapped_lines
        )

    font = ImageFont.truetype(font, fontsize)
    max_width = int(max_width)

    # getbbox() 返回的是“当前字形的可见墨迹高度”，并不是字体行高。例如只含
    # A、m、n 等无下伸部字符的英文会缺少 descent，多行时这个误差会逐行累积，
    # 最终让 TextClip 的最后一行被画布裁掉。ascent + descent 来自字体自身，
    # 不受具体语种和字符组合影响，也与 MoviePy 的 baseline 绘制模型一致。
    ascent, descent = font.getmetrics()
    line_height = int(ascent + descent)
    if line_height <= 0:
        # 正常 TrueType/OpenType 字体不会进入这里；保留可诊断日志和字号兜底，
        # 避免损坏或非常规字体返回异常 metrics 后生成零高度字幕。
        logger.warning(
            "invalid subtitle font metrics, fallback to font size: "
            f"ascent={ascent}, descent={descent}, fontsize={fontsize}"
        )
        line_height = max(1, int(fontsize))

    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        if not inner_text:
            return 0, line_height
        left, top, right, bottom = font.getbbox(inner_text)
        # bbox 仍适合测量换行所需的实际宽度；高度必须始终使用稳定字体行高。
        return right - left, line_height

    width, height = get_text_size(text)
    if width <= max_width:
        # SRT 条目允许作者手工换行。即使整段文本在宽度上不需要再次折行，
        # 画布高度仍必须按现有行数计算，否则第二行及后续行会被裁掉。
        return text, (text.count("\n") + 1) * line_height

    def attached_text_units(token):
        units = []
        for char in token:
            if units and unicodedata.category(char).startswith("M"):
                units[-1] += char
            else:
                units.append(char)
        return units

    def split_long_token(token):
        # 当一个 token 本身就超宽时（常见于中文无空格长句，或英文超长单词），
        # 退化为字符级拆分。关键点是：检测到 candidate 超宽时，先提交上一个
        # 仍然合法的 current，再把当前字符放入下一行，不能把超宽字符塞回上一行。
        lines = []
        current = ""
        for char in attached_text_units(token):
            candidate = f"{current}{char}"
            candidate_width, _ = get_text_size(candidate)
            if candidate_width <= max_width or not current:
                current = candidate
                continue
            lines.append(current)
            current = char
        if current:
            lines.append(current)
        return lines

    lines = []
    current = ""
    words = text.split(" ")
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        word_width, _ = get_text_size(word)
        if word_width <= max_width:
            current = word
        else:
            lines.extend(split_long_token(word))
            current = ""

    if current:
        lines.append(current)

    line_start_punctuation = "，。！？；：、,.!?;:)]}）】》」』”’"
    for index in range(1, len(lines)):
        # 中文长句按字符拆分时，最后一个句号、逗号等闭合标点可能被单独
        # 放到下一行，导致字幕背景被异常撑高，视觉上像一个小点掉在正文
        # 下方。这里在不重新设计换行算法的前提下，把上一行最后一个字
        # 移到标点行前面，让标点跟随文字显示，兼容中英文常见闭合标点。
        if not lines[index] or lines[index][0] not in line_start_punctuation:
            continue
        if len(lines[index - 1]) <= 1:
            continue

        previous_units = attached_text_units(lines[index - 1])
        if len(previous_units) <= 1:
            continue
        last_unit = previous_units[-1]
        candidate = f"{last_unit}{lines[index]}"
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            lines[index] = candidate
            lines[index - 1] = lines[index - 1][:-len(last_unit)]

    result = "\n".join(line.strip() for line in lines if line.strip()).strip()
    # 高度以最终结果为准。原文本中的显式换行可能保留在某个 token 内，
    # 此时临时 lines 列表的长度不等于 MoviePy 实际渲染的行数。
    height = (result.count("\n") + 1) * line_height
    return result, height


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    # Match the Pillow color syntax accepted by subtitle preflight.
    # The background clip sets opacity separately from the color value.
    if isinstance(color, str):
        try:
            return ImageColor.getcolor(color, "RGB")
        except (ValueError, TypeError):
            pass
    return (0, 0, 0)


def _rounded_subtitle_background_clip(
    width: int,
    height: int,
    color: str,
    alpha: int = 140,
    radius: int = 16,
) -> ImageClip:
    # 新字幕背景仅在用户显式开启时使用：通过 RGBA 图片绘制圆角半透明底板，
    # 再交给 MoviePy 作为透明 ImageClip 参与合成。这样默认路径完全不变，
    # 同时可以低成本试验更柔和的字幕视觉效果。
    rgb = _hex_to_rgb(color)
    safe_alpha = max(0, min(255, int(alpha)))
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [0, 0, max(0, width - 1), max(0, height - 1)],
        radius=max(0, int(radius)),
        fill=(rgb[0], rgb[1], rgb[2], safe_alpha),
    )
    return ImageClip(np.array(img), transparent=True)


def _get_visible_center_position(
    text_clip: TextClip,
    container_width: int,
    container_height: int,
) -> tuple[int, int]:
    """
    按文字真实可见像素把 TextClip 放到背景容器中心。

    MoviePy 的 TextClip 会按字体行高和 baseline 创建透明画布。很多字体的
    可见字形并不在这个画布的几何中心，直接 `with_position("center")`
    会把整块透明画布居中，导致字幕看起来偏上或偏下。这里读取 TextClip
    的透明 mask，只根据实际有像素的 bbox 计算偏移，让用户看到的文字
    在字幕背景里视觉居中。
    """
    x = int(round((container_width - text_clip.w) / 2))
    y = int(round((container_height - text_clip.h) / 2))

    try:
        if text_clip.mask is None:
            return x, y

        mask_frame = text_clip.mask.get_frame(0)
        ys, _ = np.where(mask_frame > 0.01)
        if len(ys) == 0:
            return x, y

        visible_top = int(ys.min())
        visible_bottom = int(ys.max())
        visible_height = visible_bottom - visible_top + 1
        y = int(round((container_height - visible_height) / 2 - visible_top))
    except Exception as exc:
        logger.debug(f"failed to center subtitle text by visible mask: {str(exc)}")

    return x, y


def validate_subtitle_colors(params: VideoParams) -> None:
    """Reject colors Pillow cannot render, before generation services run."""
    if not params.subtitle_enabled:
        return
    colors = [("text_fore_color", params.text_fore_color)]
    # Rendering truncates the width to an integer; a zero-width stroke is unused.
    if int(params.stroke_width) > 0:
        colors.append(("stroke_color", params.stroke_color))
    if isinstance(params.text_background_color, str):
        colors.append(("text_background_color", params.text_background_color))
    for field, value in colors:
        if value is None:
            continue
        try:
            ImageColor.getcolor(value, "RGBA")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid subtitle color: {field}") from exc


def subtitle_colors_are_indistinguishable(params: VideoParams) -> bool:
    """判断字幕文字和背景是否同色，提醒用户可能无法看清字幕。"""
    if not params.subtitle_enabled or not params.text_background_color:
        return False

    def normalize_color(value):
        if isinstance(value, bool):
            return "#000000" if value else ""
        return str(value or "").strip().lower()

    text_color = normalize_color(params.text_fore_color)
    background_color = normalize_color(params.text_background_color)
    return bool(text_color and text_color == background_color)


@lru_cache(maxsize=64)
def _subtitle_font_supports_sample(font_path: str, sample: str) -> bool:
    """检查字体是否包含样本文字需要的字形，并缓存重复检查结果。"""
    try:
        font = ImageFont.truetype(font_path, 30)
        missing_mask = font.getmask("\U0010ffff")
        missing_signature = (
            missing_mask.size,
            missing_mask.getbbox(),
            bytes(missing_mask),
        )
        for char in sample:
            char_mask = font.getmask(char)
            char_signature = (
                char_mask.size,
                char_mask.getbbox(),
                bytes(char_mask),
            )
            if char_mask.getbbox() is None or char_signature == missing_signature:
                return False
        return True
    except Exception as e:
        # 字体探测失败不应阻止用户生成；保留日志供环境兼容问题排查。
        logger.warning(f"failed to inspect subtitle font glyphs: {font_path}, {e}")
        return True


def subtitle_font_supports_text(font_path: str, text: str) -> bool:
    """检查字体能否绘制文本中的字母和数字，忽略空白及标点符号。"""
    sample = "".join(
        dict.fromkeys(
            char
            for char in str(text or "")
            if unicodedata.category(char)[0] in {"L", "N"}
        )
    )[:64]
    if not sample:
        return True
    return _subtitle_font_supports_sample(font_path, sample)


_DEFAULT_SUBTITLE_FONT = "STHeitiMedium.ttc"
# Fonts in resource/fonts, best first. The CJK-first default only suits CJK
# text: it renders Latin in a thin CJK face and has no Vietnamese glyphs, so
# non-CJK text prefers the bold Latin/Vietnamese face and falls back from there.
_LATIN_FONT_PREFERENCE = (
    "BeVietnamPro-Bold.ttf",
    "BeVietnamPro-Medium.ttf",
    "STHeitiMedium.ttc",
    "MicrosoftYaHeiBold.ttc",
    "STHeitiLight.ttc",
    "MicrosoftYaHeiNormal.ttc",
)
_CJK_FONT_PREFERENCE = (
    "STHeitiMedium.ttc",
    "MicrosoftYaHeiBold.ttc",
    "STHeitiLight.ttc",
    "MicrosoftYaHeiNormal.ttc",
)
_MAX_COVERAGE_SAMPLE_CHARS = 400

# Unicode blocks whose text needs a shaping/bidi engine (libraqm) to look right:
# Hebrew, Arabic and its presentation forms, then the Indic and Thai scripts.
_COMPLEX_SCRIPT_RANGES = (
    (0x0590, 0x08FF),
    (0x0900, 0x0DFF),
    (0x0E00, 0x0E7F),
    (0xFB1D, 0xFDFF),
    (0xFE70, 0xFEFF),
)
_CJK_RANGES = (
    (0x3040, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),
    (0xF900, 0xFAFF),
)


def _text_has_chars_in(text: str, ranges) -> bool:
    return any(
        low <= ord(char) <= high for char in text for low, high in ranges
    )


def _raqm_available() -> bool:
    try:
        from PIL import features

        return bool(features.check("raqm"))
    except Exception:  # pragma: no cover - Pillow without feature probing
        return False


def _coverage_sample(text: str) -> str:
    return "".join(
        dict.fromkeys(
            char
            for char in str(text or "")
            if unicodedata.category(char)[0] in {"L", "N"}
        )
    )[:_MAX_COVERAGE_SAMPLE_CHARS]


def _font_covers_text(font_path: str, text: str) -> bool:
    sample = _coverage_sample(text)
    return not sample or _subtitle_font_supports_sample(font_path, sample)


def _list_bundled_fonts() -> list[str]:
    try:
        return sorted(
            name
            for name in os.listdir(utils.font_dir())
            if name.lower().endswith((".ttf", ".ttc", ".otf"))
        )
    except OSError:
        return []


def _resolve_subtitle_font(
    font_name: str | None, text: str
) -> tuple[str, list[dict]]:
    """
    Pick the subtitle font for ``text`` and report what could not be honoured.

    The shipped default (a CJK face) is replaced by a script-appropriate bundled
    font. An explicitly chosen font that lacks the needed glyphs is swapped for
    a bundled font that has them. When nothing covers the text, or complex
    scripts would be drawn unshaped, the render still proceeds and the problem
    is returned as a warning instead of being left as silent tofu.
    """
    warnings: list[dict] = []
    requested = font_name or _DEFAULT_SUBTITLE_FONT
    sample = _coverage_sample(text)
    if not sample:
        return requested, warnings

    fonts_dir = utils.font_dir()
    bundled = _list_bundled_fonts()
    is_default = requested == _DEFAULT_SUBTITLE_FONT
    chosen = requested

    def usable(name: str) -> bool:
        return name in bundled and _font_covers_text(
            os.path.join(fonts_dir, name), text
        )

    if is_default:
        preference = (
            _CJK_FONT_PREFERENCE
            if _text_has_chars_in(sample, _CJK_RANGES)
            else _LATIN_FONT_PREFERENCE
        )
        for candidate in preference:
            if usable(candidate):
                chosen = candidate
                break
        if chosen != requested:
            logger.info(f"default subtitle font replaced for this text: {chosen}")
    else:
        try:
            requested_path = file_security.resolve_path_within_directory(
                fonts_dir, requested
            )
        except ValueError:
            # Path validation reports its own, clearer error later.
            return requested, warnings
        if not _font_covers_text(requested_path, text):
            replacement = next(
                (
                    name
                    for name in (*_LATIN_FONT_PREFERENCE, *_CJK_FONT_PREFERENCE, *bundled)
                    if usable(name)
                ),
                None,
            )
            if replacement:
                chosen = replacement
                warnings.append(
                    {
                        "code": "subtitle_font_replaced",
                        "font": requested,
                        "replacement": replacement,
                        "message": (
                            f"Font {requested} lacks glyphs used by the subtitles; "
                            f"{replacement} was used instead."
                        ),
                    }
                )

    try:
        chosen_path = file_security.resolve_path_within_directory(fonts_dir, chosen)
    except ValueError:
        return chosen, warnings
    if not _font_covers_text(chosen_path, text):
        warnings.append(
            {
                "code": "subtitle_glyphs_missing",
                "font": chosen,
                "message": (
                    f"No bundled font covers every character of the subtitles; "
                    f"some will render as empty boxes with {chosen}."
                ),
            }
        )
    if _text_has_chars_in(sample, _COMPLEX_SCRIPT_RANGES) and not _raqm_available():
        warnings.append(
            {
                "code": "subtitle_complex_script_unshaped",
                "message": (
                    "The subtitles use a right-to-left or complex script, but Pillow "
                    "was built without libraqm, so letters are not joined or ordered "
                    "correctly. Install libraqm (libraqm0) for correct rendering."
                ),
            }
        )
    for warning in warnings:
        logger.warning(warning["message"])
    return chosen, warnings


class _RenderedSubtitle:
    """A rendered cue plus the on-screen origin of its top-left corner."""

    def __init__(self, clip, x: int, y: int, start: float, end: float, static_after: float):
        self.clip = clip
        self.x = x
        self.y = y
        self.start = start
        self.end = end
        self.static_after = static_after
        self._static = None

    def frame_at(self, seconds: float):
        relative = min(max(seconds - self.start, 0.0), self.clip.duration or 0.0)
        if relative >= self.static_after and self._static is not None:
            return self._static
        rgb = self.clip.get_frame(relative)
        mask = self.clip.mask.get_frame(relative) if self.clip.mask is not None else None
        pair = (rgb, mask)
        if relative >= self.static_after:
            self._static = pair
        return pair

    def close(self) -> None:
        # Text clips hold only in-memory images (no reader or process), so
        # dropping the references is enough; close_clip's full collection per
        # cue would dominate the encode on long word-by-word subtitles.
        self.clip = None
        self._static = None


class _LazySubtitleOverlay:
    """
    Burn subtitles into frames as they are encoded.

    Rendering every cue up front keeps one text image (plus background and
    mask) per cue alive for the whole encode, which reaches gigabytes for long
    word-by-word subtitles. Here a cue is rendered when playback reaches it and
    only a few recent cues stay cached.
    """

    def __init__(self, cues, build_cue: Callable, cache_size: int = 3):
        ordered = sorted(cues, key=lambda cue: cue[0][0])
        self._cues = ordered
        self._starts = [cue[0][0] for cue in ordered]
        self._ends = [cue[0][1] for cue in ordered]
        self._build_cue = build_cue
        self._cache_size = max(1, cache_size)
        self._cache: dict[int, _RenderedSubtitle] = {}
        self._order: list[int] = []
        self._lock = threading.Lock()

    def _entry(self, index: int) -> _RenderedSubtitle:
        with self._lock:
            entry = self._cache.get(index)
            if entry is not None:
                self._order.remove(index)
                self._order.append(index)
                return entry
            entry = self._build_cue(self._cues[index])
            self._cache[index] = entry
            self._order.append(index)
            while len(self._order) > self._cache_size:
                evicted = self._cache.pop(self._order.pop(0))
                evicted.close()
            return entry

    def _active_indices(self, seconds: float) -> list[int]:
        upper = bisect.bisect_right(self._starts, seconds)
        active = [
            index
            for index in range(upper - 1, max(-1, upper - 9), -1)
            if seconds < self._ends[index]
        ]
        active.reverse()
        return active

    def apply(self, get_frame, seconds):
        frame = get_frame(seconds)
        active = self._active_indices(float(seconds))
        if not active:
            return frame
        output = frame.copy()
        frame_height, frame_width = output.shape[:2]
        for index in active:
            entry = self._entry(index)
            rgb, mask = entry.frame_at(float(seconds))
            cue_height, cue_width = rgb.shape[:2]
            x0, y0 = max(0, entry.x), max(0, entry.y)
            x1 = min(frame_width, entry.x + cue_width)
            y1 = min(frame_height, entry.y + cue_height)
            if x1 <= x0 or y1 <= y0:
                continue
            source = rgb[y0 - entry.y : y1 - entry.y, x0 - entry.x : x1 - entry.x, :3]
            if mask is None:
                output[y0:y1, x0:x1] = source
                continue
            alpha = mask[y0 - entry.y : y1 - entry.y, x0 - entry.x : x1 - entry.x]
            alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)[..., None]
            region = output[y0:y1, x0:x1].astype(np.float32)
            blended = region * (1.0 - alpha) + source.astype(np.float32) * alpha
            output[y0:y1, x0:x1] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
        return output

    def close(self) -> None:
        with self._lock:
            for entry in self._cache.values():
                entry.close()
            self._cache.clear()
            self._order.clear()


_DUCKING_WINDOW_SECONDS = 0.05
_DUCKING_CHUNK_SECONDS = 2.0
_DUCKING_GAIN = 0.35
_DUCKING_ATTACK_SECONDS = 0.12
_DUCKING_RELEASE_SECONDS = 0.5
_DUCKING_HOLD_SECONDS = 0.3


def _voice_ducking_gain_curve(voice_clip) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Per-window music gain that dips while the narration is speaking.

    Window RMS decides when the voice is active; a short hold keeps the music
    down across the gaps between words, and separate attack/release smoothing
    avoids pumping. Returns ``(times, gains)`` or None for a silent voice.
    """
    # Analyse at the clip's own sample rate: MoviePy returns silence when an
    # audio file is resampled on read.
    rate = int(getattr(voice_clip, "fps", 0) or 44100)
    window = max(1, int(rate * _DUCKING_WINDOW_SECONDS))
    window_seconds = window / rate
    levels: list[np.ndarray] = []
    leftover = np.zeros(0, dtype=np.float32)
    # Read explicit in-range timestamps: MoviePy's iter_chunks overshoots the end
    # and an out-of-range read of a short clip comes back as pure silence.
    total_samples = int(float(voice_clip.duration) * rate)
    chunk_samples = int(_DUCKING_CHUNK_SECONDS * rate)
    for first in range(0, total_samples, chunk_samples):
        timings = np.arange(first, min(first + chunk_samples, total_samples)) / rate
        chunk = voice_clip.to_soundarray(
            tt=timings, fps=rate, quantize=False, buffersize=chunk_samples
        )
        mono = np.asarray(chunk, dtype=np.float32)
        if mono.ndim == 2:
            mono = mono.mean(axis=1)
        mono = np.concatenate([leftover, mono])
        usable = (len(mono) // window) * window
        if usable:
            blocks = mono[:usable].reshape(-1, window)
            levels.append(np.sqrt((blocks * blocks).mean(axis=1)))
        leftover = mono[usable:]
    if len(leftover):
        padded = np.zeros(window, dtype=np.float32)
        padded[: len(leftover)] = leftover
        levels.append(np.array([np.sqrt((padded * padded).mean())], dtype=np.float32))
    if not levels:
        return None
    rms = np.concatenate(levels)
    speech = rms[rms > 1e-4]
    if speech.size == 0:
        return None
    threshold = max(0.004, 0.08 * float(np.percentile(speech, 90)))
    active = (rms > threshold).astype(np.float32)
    hold = max(1, int(round(_DUCKING_HOLD_SECONDS / window_seconds)))
    active = (np.convolve(active, np.ones(hold, dtype=np.float32), mode="full")[
        hold // 2 : hold // 2 + len(active)
    ] > 0).astype(np.float32)
    target = 1.0 - active * (1.0 - _DUCKING_GAIN)
    attack = 1.0 - math.exp(-window_seconds / _DUCKING_ATTACK_SECONDS)
    release = 1.0 - math.exp(-window_seconds / _DUCKING_RELEASE_SECONDS)
    gains = np.empty_like(target)
    current = 1.0
    for index, goal in enumerate(target):
        current += (goal - current) * (attack if goal < current else release)
        gains[index] = current
    times = (np.arange(len(gains)) + 0.5) * window_seconds
    return times, gains


class _VoiceDucking(Effect):
    """Multiply music by a gain curve computed from the narration."""

    def __init__(self, times: np.ndarray, gains: np.ndarray):
        self.times = times
        self.gains = gains

    def apply(self, clip):
        times, gains = self.times, self.gains

        def duck(get_frame, seconds):
            frame = get_frame(seconds)
            gain = np.interp(np.atleast_1d(seconds), times, gains)
            if np.ndim(seconds) == 0:
                return frame * float(gain[0])
            return frame * gain[:, None]

        return clip.transform(duck)


def generate_video(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    params: VideoParams,
    bgm_file_override: str | None = None,
    warnings: list | None = None,
) -> bool:
    """
    合成最终视频，并返回本次背景音乐处理是否成功。

    ``warnings`` 是可选的输出列表：字体替换、缺字形、复杂文字无法正确排版等
    不会中断渲染的问题会以 ``{"code", "message", ...}`` 字典追加到其中，由任务
    编排层转交给用户，而不是静默输出方框字。

    返回值只描述 BGM 处理状态：没有请求 BGM 或成功混合时返回 True；请求了
    BGM 但加载、特效或混合失败时返回 False。即使 BGM 失败仍会继续输出只有
    旁白的视频，让任务编排层决定是否向用户展示降级警告。
    """
    aspect = VideoAspect(params.video_aspect)
    video_width, video_height = aspect.to_resolution()

    logger.info(f"generating video: {video_width} x {video_height}")
    logger.info(f"  ① video: {video_path}")
    logger.info(f"  ② audio: {audio_path}")
    logger.info(f"  ③ subtitle: {subtitle_path}")
    logger.info(f"  ④ output: {output_file}")

    # https://github.com/harry0703/MoneyPrinterTurbo/issues/217
    # PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'final-1.mp4.tempTEMP_MPY_wvf_snd.mp3'
    # write into the same directory as the output file
    output_dir = os.path.dirname(output_file)

    font_path = ""
    subtitle_cues = []
    if params.subtitle_enabled:
        if not params.font_name:
            params.font_name = _DEFAULT_SUBTITLE_FONT
        font_name = params.font_name
        if subtitle_path and os.path.exists(subtitle_path):
            try:
                subtitle_cues = moviepy_subtitles(subtitle_path)
            except Exception as exc:
                # 解析错误沿用原有路径：稍后由字幕加载阶段给出同样的异常。
                logger.debug(f"subtitle pre-read failed, deferring error: {exc}")
        if subtitle_cues:
            font_name, font_warnings = _resolve_subtitle_font(
                font_name, " ".join(text for _, text in subtitle_cues)
            )
            if warnings is not None:
                warnings.extend(font_warnings)
        # API 入口虽已预检，WebUI、CLI 和内部调用仍可直接进入渲染层；
        # 始终以真实路径校验字体必须留在 resource/fonts，阻断绝对路径、
        # ../ 穿越及指向目录外的符号链接，再交给 PIL/MoviePy 打开。
        font_path = file_security.resolve_path_within_directory(
            utils.font_dir(), font_name
        )
        if os.name == "nt":
            font_path = font_path.replace("\\", "/")

        logger.info(f"  ⑤ font: {font_path}")

    def resolve_subtitle_background_color():
        # 兼容历史参数：API 里 `text_background_color` 既可能是布尔值，
        # 也可能是实际颜色字符串。统一在这里归一化，避免把 True/False
        # 直接传给 TextClip 后出现不可预期的渲染结果。
        if isinstance(params.text_background_color, bool):
            return "#000000" if params.text_background_color else None
        return params.text_background_color

    def create_text_clip(subtitle_item):
        params.font_size = int(params.font_size)
        params.stroke_width = int(params.stroke_width)
        phrase = subtitle_item[1]
        max_width = video_width * 0.9
        bg_color = resolve_subtitle_background_color()
        rounded_bg_enabled = bool(
            getattr(params, "rounded_subtitle_background", False) and bg_color
        )
        has_subtitle_background = bool(bg_color)
        # 圆角背景按文字真实宽度生成，左右留白应更克制；旧矩形背景仍保留
        # 较大的安全边距，避免历史配置中的长字幕贴边或被裁切。
        padding_ratio = 0.4 if rounded_bg_enabled else 0.6
        pad_x = int(params.font_size * padding_ratio) if has_subtitle_background else 0
        # 字幕背景需要给文字左右留出明确内边距。先从可用宽度中扣除
        # padding 再换行，避免长英文或大字号刚好撑满 90% 视频宽度后，
        # 文字贴到背景框边缘，看起来像被裁切。普通矩形背景和圆角背景
        # 都走这条逻辑；无背景字幕则保持原有最大宽度。
        text_max_width = max(1, int(max_width) - 2 * pad_x)
        wrapped_txt, txt_height = wrap_text(
            phrase,
            max_width=text_max_width,
            font=font_path,
            fontsize=params.font_size,
        )
        interline = int(params.font_size * 0.25)
        line_count = wrapped_txt.count("\n") + 1
        vertical_padding = int(params.font_size * 0.35)
        # Pillow/MoviePy 会把描边向字形上下两侧扩张，并把这部分计入每一行
        # 的行进高度。若只在整个字幕块外增加一次描边留白，粗描边多行文本
        # 仍会逐行累积误差。这里按实际行数计入双侧描边空间，默认细描边只
        # 增加少量高度，而“小字号 + 粗描边 + 多行”也能完整显示。
        stroke_padding = int(params.stroke_width * 2 * line_count)
        text_clip_margin_y = max(
            int(params.font_size * 0.3), int(params.stroke_width * 2)
        )
        # MoviePy 在 `method=label` 下会自动收缩文本框高度，遇到多行字幕、
        # 描边或背景色时，容易把最后一行的下半部分裁掉。这里显式传入
        # 一个更保守的高度，把行间距和额外上下留白一并算进去，保证字幕
        # 背景框与文字本身都能完整渲染出来。
        clip_h = int(
            txt_height
            + vertical_padding
            + (interline * line_count)
            + stroke_padding
        )

        if rounded_bg_enabled:
            # 圆角背景需要贴合文字宽度，而不是沿用 90% 视频宽度。这里先用
            # PIL 测量最长一行文字，再加水平内边距，避免短字幕出现过宽底板。
            try:
                font = ImageFont.truetype(font_path, params.font_size)
                text_w = max(
                    int(font.getbbox(line)[2] - font.getbbox(line)[0])
                    for line in wrapped_txt.split("\n")
                )
            except Exception as exc:
                logger.warning(
                    f"failed to measure subtitle text width, fallback to max width: {str(exc)}"
                )
                text_w = int(max_width)

            box_w = max(1, min(int(max_width), text_w + 2 * pad_x))
            radius = max(8, int(params.font_size * 0.4))
            text_clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=(box_w, None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            clip_h = max(clip_h, text_clip.h)
            bg_clip = _rounded_subtitle_background_clip(
                width=box_w,
                height=clip_h,
                color=bg_color,
                alpha=140,
                radius=radius,
            )
            text_position = _get_visible_center_position(text_clip, box_w, clip_h)
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=(box_w, clip_h),
            )
        elif bg_color:
            size = (
                int(max_width),
                clip_h,
            )
            text_clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=(int(max_width), None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            size = (size[0], max(size[1], text_clip.h))
            bg_clip = _rounded_subtitle_background_clip(
                width=size[0],
                height=size[1],
                color=bg_color,
                alpha=255,
                radius=0,
            )
            text_position = _get_visible_center_position(text_clip, size[0], size[1])
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=size,
            )
        else:
            size = (
                int(max_width),
                clip_h,
            )
            _clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=size,
                text_align="center",
            )
        duration = subtitle_item[0][1] - subtitle_item[0][0]
        _clip = _clip.with_start(subtitle_item[0][0])
        _clip = _clip.with_end(subtitle_item[0][1])
        _clip = _clip.with_duration(duration)

        # 弹跳动画只在用户显式选择时启用；默认 none 完全沿用原字幕渲染路径。
        anim_type = getattr(params, "subtitle_animation", "none")
        if anim_type in ("pop_spring", "spring", "pop"):
            _clip = _apply_subtitle_spring_animation(_clip, duration)

        return _clip

    def subtitle_origin(clip_w, clip_h, canvas_w, canvas_h):
        """Top-left pixel of a cue; the same placement rules as before."""
        x = (canvas_w - clip_w) / 2
        if params.subtitle_position == "bottom":
            y = video_height * 0.95 - clip_h
        elif params.subtitle_position == "top":
            y = video_height * 0.05
        elif params.subtitle_position in ("two_thirds_bottom", "two_thirds", "2/3_bottom"):
            # 2/3 from the bottom = 1/3 from the top: y = (video_height - h) * (1/3)
            y = (video_height - clip_h) / 3.0
        elif params.subtitle_position == "custom":
            # Ensure the subtitle is fully within the screen bounds
            margin = 10  # Additional margin, in pixels
            max_y = video_height - clip_h - margin
            min_y = margin
            custom_y = (video_height - clip_h) * (params.custom_position / 100)
            y = max(min_y, min(custom_y, max_y))
        else:  # center
            y = (canvas_h - clip_h) / 2
        return int(round(x)), int(round(y))

    def build_subtitle_cue(subtitle_item, canvas_w, canvas_h):
        cue_clip = create_text_clip(subtitle_item=subtitle_item)
        anim_type = getattr(params, "subtitle_animation", "none")
        animated = anim_type in ("pop_spring", "spring", "pop")
        x, y = subtitle_origin(cue_clip.w, cue_clip.h, canvas_w, canvas_h)
        start, end = subtitle_item[0]
        cue_duration = max(0.0, end - start)
        return _RenderedSubtitle(
            cue_clip,
            x,
            y,
            start,
            end,
            static_after=(
                min(_SUBTITLE_SPRING_DURATION_SECONDS, cue_duration) if animated else 0.0
            ),
        )

    # MoviePy 的 CompositeAudioClip.close() 不会关闭子 AudioFileClip。这里用
    # ExitStack 显式持有所有原始文件 reader，确保成功、字幕异常、混音失败和
    # 视频写入失败等路径都能释放 FFmpeg 子进程，尤其避免 Windows 文件被占用。
    with ExitStack() as clip_stack:
        source_video_clip = clip_stack.enter_context(
            _open_video_clip_quietly(video_path)
        )
        voice_source_clip = clip_stack.enter_context(AudioFileClip(audio_path))
        video_clip = source_video_clip
        audio_clip = voice_source_clip.with_effects(
            [afx.MultiplyVolume(1.0 if params.voice_volume is None else params.voice_volume)]
        )

        def make_textclip(text):
            return TextClip(
                text=text,
                font=font_path,
                font_size=params.font_size,
            )

        if params.subtitle_enabled and subtitle_path and os.path.exists(subtitle_path):
            sub = clip_stack.enter_context(
                SubtitlesClip(
                    subtitles=subtitle_cues or moviepy_subtitles(subtitle_path),
                    encoding="utf-8",
                    make_textclip=make_textclip,
                )
            )
            # Cues are rendered lazily while frames are encoded and only a few
            # stay cached; building them all here held every cue's text image,
            # background and mask in memory at once.
            canvas_w, canvas_h = (int(value) for value in video_clip.size)
            overlay = _LazySubtitleOverlay(
                sub.subtitles,
                lambda item: build_subtitle_cue(item, canvas_w, canvas_h),
            )
            clip_stack.callback(overlay.close)
            video_clip = video_clip.transform(overlay.apply)

        bgm_enabled = bgm_service.should_use_bgm(
            params.bgm_type, params.bgm_volume
        )
        if not bgm_enabled and params.bgm_type:
            # 所有 BGM 来源共用这一条短路规则。音量不大于 0 时不能解析随机或
            # 自定义文件，也不能加载提供商返回的文件，避免无意义的 IO 和混音。
            logger.info(
                f"skipping background music because volume is not positive: "
                f"type={params.bgm_type}, volume={params.bgm_volume}"
            )

        # 提供商配乐可由任务编排层直接传入对应文件。None 表示沿用随机/自定义
        # BGM 解析，空字符串明确禁用本条 BGM；但任何来源都必须先通过通用音量规则。
        bgm_file = ""
        if bgm_enabled:
            bgm_file = (
                bgm_file_override
                if bgm_file_override is not None
                else get_bgm_file(
                    bgm_type=params.bgm_type,
                    bgm_file=params.bgm_file,
                )
            )
        bgm_mix_succeeded = True
        if bgm_file:
            try:
                bgm_effects = [afx.MultiplyVolume(params.bgm_volume)]
                bgm_source_clip = clip_stack.enter_context(AudioFileClip(bgm_file))
                bgm_input = bgm_source_clip
                # 服务内解析的随机/自定义音乐可能比成片短，需要循环铺满；任务层
                # 通过 override 传入的文件表示提供商已经完成时长适配。这里依据
                # 文件来源决定是否循环，避免今后每增加一个提供商都修改名称白名单。
                if bgm_file_override is None:
                    bgm_effects.append(afx.AudioLoop(duration=video_clip.duration))
                else:
                    # Provider tracks may run past the video. Fade-out is relative
                    # to the clip end, so cut the track to the video first or the
                    # fade would happen after the video has already finished.
                    provider_length = getattr(bgm_source_clip, "duration", None)
                    if provider_length and provider_length > video_clip.duration:
                        bgm_input = bgm_source_clip.subclipped(0, video_clip.duration)
                if getattr(params, "bgm_ducking", True):
                    try:
                        curve = _voice_ducking_gain_curve(audio_clip)
                    except Exception as exc:
                        curve = None
                        logger.warning(f"skipping music ducking: {exc}")
                    if curve is not None:
                        bgm_effects.append(_VoiceDucking(*curve))
                # Fade the complete playback timeline, not each source loop.
                bgm_effects.append(afx.AudioFadeOut(3))
                bgm_clip = bgm_input.with_effects(bgm_effects)
                audio_clip = CompositeAudioClip([audio_clip, bgm_clip])
            except Exception:
                bgm_mix_succeeded = False
                # 记录完整堆栈和稳定上下文，便于区分文件解码、MoviePy 特效和
                # CompositeAudioClip 失败；文件内容与 API Key 不会进入日志。
                logger.exception(
                    f"failed to mix background music: type={params.bgm_type}, "
                    f"file={bgm_file}"
                )

        final_video_clip = video_clip.with_audio(audio_clip)
        clip_stack.callback(final_video_clip.close)
        # 显式沿用输入音频的采样率；如果取不到，再回退 MoviePy 默认的 44100Hz。
        # 这样可以减少不同环境，尤其 Docker 中再次重采样带来的音质波动。
        output_audio_fps = int(getattr(audio_clip, "fps", 0) or 44100)
        with _stage_heartbeat("final video render"):
            _write_videofile_with_codec_fallback(
                final_video_clip,
                output_file=output_file,
                codec=_get_configured_video_codec(),
                atomic_output=True,
                audio_codec=audio_codec,
                audio_fps=output_audio_fps,
                audio_bitrate=audio_bitrate,
                temp_audiofile_path=_get_temp_audio_dir(output_dir),
                threads=params.n_threads or 2,
                logger=None,
                fps=fps,
            )
        return bgm_mix_succeeded


# Largest canvas edge is 1920 px; keeping the photo at about 1.2x of that leaves
# room for the zoom and a cover-crop while avoiding 20+ megapixel frames.
_IMAGE_ZOOM_MIN_SIDE_PX = 2304
_IMAGE_ZOOM_MAX_SIDE_PX = 4608


def _image_zoom_working_size(width: int, height: int) -> tuple[int, int]:
    """Size to render a photo at: never larger than needed, never upscaled."""
    width, height = int(width), int(height)
    if min(width, height) <= 0:
        return width, height
    scale = 1.0
    if min(width, height) > _IMAGE_ZOOM_MIN_SIDE_PX:
        scale = _IMAGE_ZOOM_MIN_SIDE_PX / min(width, height)
    if max(width, height) * scale > _IMAGE_ZOOM_MAX_SIDE_PX:
        scale = _IMAGE_ZOOM_MAX_SIDE_PX / max(width, height)
    if scale >= 1.0:
        return width, height
    # H.264 needs even dimensions.
    return (
        max(2, int(round(width * scale / 2)) * 2),
        max(2, int(round(height * scale / 2)) * 2),
    )


def render_image_zoom_video(image_path: str, clip_duration: int = 5) -> str:
    """
    将单张本地图片渲染为带缓慢放大效果的 mp4 片段，返回输出文件路径。

    local 素材预处理和 OpenAI 兼容文生图素材共用这段"图片 → 片段"渲染
    逻辑：ImageClip 按 clip_duration 固定时长播放，并叠加每秒约 3% 的
    动态放大，避免静态画面在成片中显得呆板。渲染异常由调用方按各自
    素材源的失败约定处理。
    """
    clip, _ = _open_image_clip_with_fallback(image_path)
    clip = clip.with_duration(clip_duration).with_position("center")
    temp_path = ""
    zoom_source = clip
    try:
        # Phone photos are 12-48 MP. Every frame of the zoom resamples the whole
        # image, so downscale once up front to a size the canvas can use.
        source_size = getattr(clip, "size", None)
        working_size = (
            _image_zoom_working_size(*source_size) if source_size else None
        )
        if working_size and working_size != tuple(int(v) for v in source_size):
            logger.info(
                f"downscaling photo {clip.size[0]}x{clip.size[1]} -> "
                f"{working_size[0]}x{working_size[1]} before zoom render"
            )
            zoom_source = clip.resized(new_size=working_size)
        # Apply a zoom effect using the resize method.
        # A lambda function is used to make the zoom effect dynamic over time.
        # The zoom effect starts from the original size and gradually scales up to 120%.
        # t represents the current time, and clip.duration is the total duration of the clip.
        # Note: 1 represents 100% size, so 1.2 represents 120%.
        zoom_clip = zoom_source.resized(
            lambda t: 1 + (clip_duration * 0.03) * (t / clip.duration)
        )

        # Optionally, create a composite video clip containing the zoomed clip.
        # This is useful if you want to add other elements to the video.
        final_clip = CompositeVideoClip([zoom_clip], bg_color=(0, 0, 0))
        try:
            # The duration changes the rendered content, so it must be part of
            # the output identity. Different tasks may render the same image
            # concurrently; only publish a complete MP4 after MoviePy closes it.
            source_digest = hashlib.sha256(os.fsencode(os.path.abspath(image_path)))
            source_frame = clip.get_frame(0)
            source_digest.update(str(source_frame.shape).encode())
            source_digest.update(source_frame.tobytes())
            if clip.mask is not None:
                source_digest.update(clip.mask.get_frame(0).tobytes())
            source_identity = source_digest.hexdigest()
            duration_identity = hashlib.sha256(str(clip_duration).encode()).hexdigest()[:16]
            video_file = os.path.join(
                os.path.dirname(image_path), f"{source_identity}.zoom-{duration_identity}.mp4"
            )
            descriptor, temp_path = tempfile.mkstemp(
                prefix=".image-zoom-",
                suffix=".mp4",
                dir=os.path.dirname(os.path.abspath(video_file)),
            )
            os.close(descriptor)
            _write_videofile_with_codec_fallback(
                final_clip,
                temp_path,
                codec=_DEFAULT_VIDEO_CODEC,
                intermediate=True,
                fps=30,
                logger=None,
                ffmpeg_params=_BT709_FFMPEG_PARAMS,
            )
        finally:
            close_clip(final_clip)
        os.replace(temp_path, video_file)
        temp_path = ""
        return video_file
    finally:
        if zoom_source is not clip:
            close_clip(zoom_source)
        close_clip(clip)
        if temp_path:
            delete_files(temp_path)


def preprocess_video(materials: List[MaterialInfo], clip_duration=4):
    # WebUI 在某些二次生成场景下可能传入空素材列表，这里直接返回空结果，避免抛出 NoneType 异常。
    if not materials:
        return []

    # 仅返回通过预处理校验的素材，避免低分辨率图片继续进入后续的视频合成流程。
    valid_materials = []
    local_videos_dir = utils.storage_dir("local_videos", create=True)

    for material in materials:
        if not material.url:
            continue

        try:
            material_source_path = file_security.resolve_path_within_directory(
                local_videos_dir, material.url
            )
        except ValueError as exc:
            # local video_source 的素材路径来自 API 参数，必须限制在专用素材目录。
            # 允许用户传文件名，也兼容历史返回的绝对路径，但不允许逃逸到系统
            # 其他目录，避免任意文件读取或通过 MoviePy 探测本地敏感文件。
            logger.warning(
                f"skip unsafe local material: {material.url}, "
                f"local_videos_dir: {local_videos_dir}, error: {str(exc)}"
            )
            continue

        ext = utils.parse_extension(material_source_path)
        is_image = ext in const.FILE_TYPE_IMAGES
        try:
            # 图片素材直接按图片方式读取，避免先走 VideoFileClip 误判后触发不稳定的回退分支。
            if is_image:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
            else:
                clip = _open_video_clip_quietly(material_source_path)
        except Exception:
            # 非标准扩展名或探测失败时再回退到图片模式，兼容历史上直接传本地图片路径的情况。
            try:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
                # The successful decoder determines the material kind, even
                # when the uploaded filename has a video or unknown suffix.
                is_image = True
            except Exception as exc:
                logger.warning(
                    f"skip unreadable local material: {material.url}, error: {str(exc)}"
                )
                continue
        try:
            width = clip.size[0]
            height = clip.size[1]
            if not is_material_resolution_acceptable(width, height):
                logger.warning(
                    f"low resolution material: {width}x{height}, minimum "
                    f"{_MIN_MATERIAL_DIMENSION}x{_MIN_MATERIAL_DIMENSION} required "
                    f"(tolerance {_MIN_DIMENSION_TOLERANCE}px)"
                )
                # 探测到低分辨率素材后立即关闭资源，并且不要把该素材返回给后续流程。
                close_clip(clip)
                continue

            if is_image:
                logger.info(f"processing image: {material_source_path}")
                # 探测尺寸时已经打开过一次素材，这里先释放探测句柄，再渲染
                # 用于导出的图片片段。
                close_clip(clip)
                video_file = render_image_zoom_video(
                    material_source_path, clip_duration
                )
                material.url = video_file
                logger.success(f"image processed: {video_file}")
            else:
                # 普通视频素材只需要读取尺寸做校验，校验完成后立即释放句柄即可。
                close_clip(clip)
                # Update url to the resolved absolute path so that downstream
                # stages (combine_videos) can open the file without re-resolving.
                material.url = material_source_path
        except Exception:
            close_clip(clip)
            raise

        valid_materials.append(material)

    return valid_materials
