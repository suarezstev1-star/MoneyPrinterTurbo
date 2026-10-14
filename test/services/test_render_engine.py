"""Render engine: quality flags, watchdog, lazy subtitles, fonts, music, repeats."""

import json
import shutil
import subprocess
import threading
import time
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from moviepy import AudioClip, ColorClip, VideoClip, VideoFileClip
from PIL import Image

from app.config import config
from app.models.schema import VideoAspect, VideoConcatMode, VideoParams
from app.services import video

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg"), reason="native FFmpeg required"
)


def _ffmpeg(*arguments):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-threads", "1", *arguments],
        check=True,
    )


def _probe(path, entries, stream="v:0"):
    output = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", stream,
            "-show_entries", f"stream={entries}", "-of", "json", str(path),
        ],
        text=True,
    )
    return json.loads(output)["streams"][0]


@pytest.fixture(autouse=True)
def _restore_app_config():
    original = dict(config.app)
    yield
    config.app.clear()
    config.app.update(original)
    video._runtime_disabled_video_codecs.clear()


# --- encoder quality flags -------------------------------------------------


def test_quality_flags_use_crf_20_by_default_and_are_configurable():
    config.app.pop("video_crf", None)
    final = video._quality_ffmpeg_params("libx264")
    assert final[:2] == ["-crf", "20"]
    assert final[-2:] == ["-movflags", "+faststart"]
    intermediate = video._quality_ffmpeg_params("libx264", intermediate=True)
    assert intermediate == ["-crf", "17"]
    config.app["video_crf"] = 26
    assert video._quality_ffmpeg_params("libx264")[:2] == ["-crf", "26"]
    for invalid in ("bad", True, None):
        config.app["video_crf"] = invalid
        assert video._get_video_crf() == 20
    config.app["video_crf"] = 999
    assert video._get_video_crf() == 51


def test_hardware_encoders_get_equivalent_constant_quality_flags():
    for codec, flag in (
        ("h264_nvenc", "-cq"),
        ("h264_amf", "-qp_i"),
        ("h264_qsv", "-global_quality"),
        ("h264_videotoolbox", "-q:v"),
    ):
        params = video._quality_ffmpeg_params(codec)
        assert flag in params and "-crf" not in params
        assert params[-2:] == ["-movflags", "+faststart"]


def test_hardware_fallback_rebuilds_flags_for_libx264():
    seen = []

    class Clip:
        duration = 1

        def write_videofile(self, output_file, codec, **kwargs):
            seen.append((codec, kwargs["ffmpeg_params"]))
            if codec == "h264_nvenc":
                raise RuntimeError("no device")

    with patch.object(video, "_ffmpeg_encoder_exists", return_value=True):
        used = video._write_videofile_with_codec_fallback(
            Clip(), "/tmp/never-written.mp4", codec="h264_nvenc", logger=None, fps=30
        )
    assert used == "libx264"
    assert "-cq" in seen[0][1] and "-crf" not in seen[0][1]
    assert "-crf" in seen[1][1] and "-cq" not in seen[1][1]


def test_final_render_is_yuv420p_crf20_faststart(tmp_path):
    clip = ColorClip((64, 64), color=(200, 30, 30)).with_duration(1).with_fps(10)
    output = tmp_path / "out.mp4"
    video._write_videofile_with_codec_fallback(
        clip, str(output), codec="libx264", atomic_output=True, logger=None, fps=10
    )
    assert _probe(output, "pix_fmt")["pix_fmt"] == "yuv420p"
    data = output.read_bytes()
    assert b"crf=20.0" in data
    # faststart: the moov atom precedes the media data.
    assert data.index(b"moov") < data.index(b"mdat")


# --- render watchdog -------------------------------------------------------


def test_stalled_render_is_killed_with_a_clear_error_and_cleaned_up(tmp_path):
    config.app["render_timeout_seconds"] = 1.5

    def slow_frame(_t):
        time.sleep(0.4)
        return np.zeros((64, 64, 3), dtype=np.uint8)

    clip = VideoClip(slow_frame, duration=30).with_fps(10)
    output = tmp_path / "stalled.mp4"
    started = time.monotonic()
    with pytest.raises(video.RenderTimeoutError, match="render_timeout_seconds"):
        video._write_videofile_with_codec_fallback(
            clip, str(output), codec="libx264", logger=None, fps=10
        )
    assert time.monotonic() - started < 20
    assert list(tmp_path.iterdir()) == []


def test_atomic_render_timeout_leaves_no_partial_final_video(tmp_path):
    config.app["render_timeout_seconds"] = 1.5
    clip = VideoClip(
        lambda _t: (time.sleep(0.4), np.zeros((64, 64, 3), dtype=np.uint8))[1],
        duration=30,
    ).with_fps(10)
    output = tmp_path / "final.mp4"
    with pytest.raises(video.RenderTimeoutError):
        video._write_videofile_with_codec_fallback(
            clip, str(output), codec="libx264", atomic_output=True, logger=None, fps=10
        )
    assert list(tmp_path.iterdir()) == []


def test_render_timeout_default_scales_with_duration_and_validates_config():
    config.app.pop("render_timeout_seconds", None)
    assert video._get_render_timeout_seconds(10) == 900
    assert video._get_render_timeout_seconds(600) == 600 * 30 + 300
    config.app["render_timeout_seconds"] = 42
    assert video._get_render_timeout_seconds(600) == 42
    config.app["render_timeout_seconds"] = -5
    assert video._get_render_timeout_seconds(10) == 900


def test_watchdog_does_not_leak_into_other_threads(tmp_path):
    config.app["render_timeout_seconds"] = 60
    results = []

    def render(name):
        clip = ColorClip((32, 32), color=(0, 0, 255)).with_duration(0.5).with_fps(10)
        target = tmp_path / f"{name}.mp4"
        video._write_videofile_with_codec_fallback(
            clip, str(target), codec="libx264", logger=None, fps=10
        )
        results.append(target.exists())

    threads = [threading.Thread(target=render, args=(f"t{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [True, True]


# --- photos ----------------------------------------------------------------


def test_image_zoom_working_size_downscales_only_when_needed():
    assert video._image_zoom_working_size(1080, 1920) == (1080, 1920)
    assert video._image_zoom_working_size(2304, 2304) == (2304, 2304)
    width, height = video._image_zoom_working_size(6000, 4000)
    assert (width, height) == (3456, 2304)
    width, height = video._image_zoom_working_size(12000, 3000)
    assert max(width, height) <= 4608 and width % 2 == 0 and height % 2 == 0
    assert video._image_zoom_working_size(8000, 6000) == (3072, 2304)


def test_large_photo_is_rendered_at_working_size_not_full_resolution(tmp_path, monkeypatch):
    # Same logic as the production limits, scaled down so the test stays fast.
    monkeypatch.setattr(video, "_IMAGE_ZOOM_MIN_SIDE_PX", 400)
    monkeypatch.setattr(video, "_IMAGE_ZOOM_MAX_SIDE_PX", 800)
    photo = tmp_path / "big.png"
    Image.new("RGB", (1000, 600), (20, 120, 220)).save(photo)
    output = video.render_image_zoom_video(str(photo), clip_duration=0.1)
    stream = _probe(output, "width,height")
    assert (stream["width"], stream["height"]) == (666 // 2 * 2, 400)


def test_small_photo_keeps_native_size(tmp_path):
    photo = tmp_path / "small.png"
    Image.new("RGB", (640, 480), (200, 10, 10)).save(photo)
    output = video.render_image_zoom_video(str(photo), clip_duration=0.1)
    stream = _probe(output, "width,height")
    assert (stream["width"], stream["height"]) == (640, 480)


# --- lazy subtitles --------------------------------------------------------


class _FakeCue:
    closed = 0

    def __init__(self, index):
        self.index = index
        self.duration = 1.0
        self.mask = _FakeMask()

    def get_frame(self, _t):
        return np.full((4, 6, 3), 255, dtype=np.uint8)


class _FakeMask:
    def get_frame(self, _t):
        return np.ones((4, 6), dtype=float)


def test_lazy_overlay_renders_on_demand_and_bounds_live_cues():
    cues = [((i, i + 1), f"w{i}") for i in range(500)]
    built = []
    live = []
    peak = [0]

    def build(cue):
        index = cue[0][0]
        built.append(index)
        entry = video._RenderedSubtitle(_FakeCue(index), 2, 3, cue[0][0], cue[0][1], 0.0)
        live.append(entry)
        peak[0] = max(peak[0], len(live))
        original_close = entry.close

        def close():
            live.remove(entry)
            original_close()

        entry.close = close
        return entry

    overlay = video._LazySubtitleOverlay(cues, build, cache_size=3)
    base = np.zeros((20, 20, 3), dtype=np.uint8)
    for second in range(500):
        frame = overlay.apply(lambda _t: base, second + 0.5)
        assert frame[3:7, 2:8].min() == 255
        assert frame[0, 0].tolist() == [0, 0, 0]
    assert built == list(range(500))
    assert peak[0] <= 4
    overlay.close()
    assert live == []
    # Source frames are never modified in place.
    assert base.max() == 0


def test_lazy_overlay_leaves_frame_untouched_between_cues_and_clips_edges():
    cues = [((1.0, 2.0), "x")]
    entry = video._RenderedSubtitle(_FakeCue(0), 18, 18, 1.0, 2.0, 0.0)
    overlay = video._LazySubtitleOverlay(cues, lambda _cue: entry)
    base = np.zeros((20, 20, 3), dtype=np.uint8)
    assert overlay.apply(lambda _t: base, 0.5) is base
    frame = overlay.apply(lambda _t: base, 1.5)
    assert frame[19, 19].tolist() == [255, 255, 255]
    assert frame[17, 17].tolist() == [0, 0, 0]


def test_generate_video_burns_subtitles_lazily_into_real_output(tmp_path, monkeypatch):
    monkeypatch.setattr(VideoAspect, "to_resolution", lambda self: (320, 480))
    monkeypatch.setattr(video, "fps", 10)
    source = tmp_path / "in.mp4"
    voice = tmp_path / "voice.wav"
    _ffmpeg("-f", "lavfi", "-i", "color=c=black:s=320x480:r=10:d=3", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(source))
    _ffmpeg("-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "3", str(voice))
    srt = tmp_path / "s.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n\n2\n00:00:01,500 --> 00:00:02,500\nWorld\n\n",
        encoding="utf-8",
    )
    params = VideoParams(video_subject="t", bgm_type="", font_size=40)
    output = tmp_path / "out.mp4"
    warnings = []
    assert video.generate_video(
        str(source), str(voice), str(srt), str(output), params, warnings=warnings
    )
    assert warnings == []
    with VideoFileClip(str(output), audio=False) as rendered:
        shown = rendered.get_frame(0.5)
        gap = rendered.get_frame(1.25)
        later = rendered.get_frame(2.0)
    assert shown[300:480].max() > 200, "subtitle text must be visible"
    assert gap.max() < 40, "no subtitle between cues"
    assert later[300:480].max() > 200


# --- fonts -------------------------------------------------------------------


def test_default_font_is_replaced_by_script_appropriate_bundled_font():
    name, warnings = video._resolve_subtitle_font("STHeitiMedium.ttc", "Plain English text")
    assert name == "BeVietnamPro-Bold.ttf" and warnings == []
    name, _ = video._resolve_subtitle_font("", "Tiếng Việt có dấu")
    assert name == "BeVietnamPro-Bold.ttf"
    name, warnings = video._resolve_subtitle_font("STHeitiMedium.ttc", "你好，世界 hello")
    assert name == "STHeitiMedium.ttc" and warnings == []
    name, _ = video._resolve_subtitle_font("STHeitiMedium.ttc", "")
    assert name == "STHeitiMedium.ttc"


def test_explicit_font_missing_glyphs_is_swapped_with_a_warning():
    name, warnings = video._resolve_subtitle_font("UTM Kabel KT.ttf", "Señor Müller")
    assert name != "UTM Kabel KT.ttf"
    assert [w["code"] for w in warnings] == ["subtitle_font_replaced"]
    name, warnings = video._resolve_subtitle_font("BeVietnamPro-Medium.ttf", "Plain text")
    assert name == "BeVietnamPro-Medium.ttf" and warnings == []


def test_uncovered_scripts_warn_instead_of_silently_rendering_tofu():
    with patch.object(video, "_raqm_available", return_value=False):
        _, warnings = video._resolve_subtitle_font("STHeitiMedium.ttc", "مرحبا بالعالم")
    codes = [w["code"] for w in warnings]
    assert "subtitle_glyphs_missing" in codes
    assert "subtitle_complex_script_unshaped" in codes
    with patch.object(video, "_raqm_available", return_value=True):
        _, warnings = video._resolve_subtitle_font("STHeitiMedium.ttc", "שלום עולם")
    assert [w["code"] for w in warnings] == ["subtitle_glyphs_missing"]


def test_font_warnings_are_returned_through_generate_video(tmp_path, monkeypatch):
    monkeypatch.setattr(VideoAspect, "to_resolution", lambda self: (160, 240))
    monkeypatch.setattr(video, "fps", 5)
    source, voice = tmp_path / "in.mp4", tmp_path / "v.wav"
    _ffmpeg("-f", "lavfi", "-i", "color=c=black:s=160x240:r=5:d=1", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(source))
    _ffmpeg("-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "1", str(voice))
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:00,900\nשלום\n\n", encoding="utf-8")
    warnings = []
    video.generate_video(
        str(source), str(voice), str(srt), str(tmp_path / "o.mp4"),
        VideoParams(video_subject="t", bgm_type="", font_size=20), warnings=warnings,
    )
    assert "subtitle_glyphs_missing" in [w["code"] for w in warnings]


# --- background music ------------------------------------------------------


def _rms_profile(path):
    decoded = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", "44100", "-"],
        capture_output=True, check=True,
    )
    samples = np.frombuffer(decoded.stdout, dtype="<f4")

    def rms(start, end):
        window = samples[int(start * 44100) : int(end * 44100)]
        return float(np.sqrt(np.mean(window * window)))

    return rms


def test_provider_music_longer_than_video_still_fades_at_the_video_end(tmp_path):
    footage, speech, music, output = (
        tmp_path / n for n in ("in.mp4", "speech.wav", "music.wav", "out.mp4")
    )
    _ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x64:r=10:d=6", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(footage))
    _ffmpeg("-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "6", str(speech))
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=20", str(music))
    params = VideoParams(video_subject="m", subtitle_enabled=False, bgm_type="sonilo",
                         bgm_volume=1.0, n_threads=1)
    assert video.generate_video(str(footage), str(speech), "", str(output), params,
                                bgm_file_override=str(music))
    rms = _rms_profile(output)
    initial = rms(0.2, 0.4)
    assert initial > 0.02
    assert rms(2.0, 2.2) > initial * 0.85
    assert rms(5.7, 5.9) < initial * 0.25, "fade-out must land on the video end"


def test_ducking_curve_dips_during_speech_and_recovers():
    rate = 8000

    def voice(t):
        t = np.atleast_1d(t)
        tone = 0.3 * np.sin(2 * np.pi * 300 * t)
        return np.where(t < 2.0, tone, 0.0)[:, None]

    clip = AudioClip(voice, duration=5.0, fps=rate)
    times, gains = video._voice_ducking_gain_curve(clip)
    during = gains[(times > 0.6) & (times < 1.8)]
    after = gains[times > 4.0]
    assert during.max() < 0.45
    assert after.min() > 0.95
    silent = AudioClip(lambda t: np.zeros((np.atleast_1d(t).size, 1)), duration=2, fps=rate)
    assert video._voice_ducking_gain_curve(silent) is None


def test_music_is_quieter_under_voice_and_ducking_can_be_disabled(tmp_path):
    footage, voice, music = (tmp_path / n for n in ("in.mp4", "voice.wav", "music.wav"))
    _ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x64:r=10:d=8", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(footage))
    # Voice only in the first 3 seconds.
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=300:duration=3", "-af", "volume=0.5,apad=whole_dur=8",
            str(voice))
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=1200:duration=8", "-af", "volume=0.5", str(music))

    def music_level(ducking):
        out = tmp_path / f"out-{ducking}.mp4"
        params = VideoParams(video_subject="d", subtitle_enabled=False, bgm_type="custom",
                             bgm_volume=1.0, bgm_ducking=ducking, n_threads=1)
        with patch.object(video, "get_bgm_file", return_value=str(music)):
            assert video.generate_video(str(footage), str(voice), "", str(out), params)
        rms = _rms_profile(out)
        # Compare the music-only tail with the section under the voice.
        return rms(1.0, 2.5), rms(4.5, 5.5)

    ducked_under, ducked_free = music_level(True)
    plain_under, plain_free = music_level(False)
    assert ducked_free > 0
    assert plain_under / plain_free > 1.0
    assert ducked_under / ducked_free < plain_under / plain_free * 0.85


# --- crossfade and repeated footage ------------------------------------------


def _solid_clip(path, color, seconds=1.0, size="64x64"):
    _ffmpeg("-f", "lavfi", "-i", f"color=c={color}:s={size}:r=10:d={seconds}", "-c:v",
            "libx264", "-pix_fmt", "yuv420p", str(path))


def test_concat_crossfades_neighbouring_clips(tmp_path):
    red, blue, output = tmp_path / "r.mp4", tmp_path / "b.mp4", tmp_path / "o.mp4"
    _solid_clip(red, "red")
    _solid_clip(blue, "blue")
    video.concat_video_clips_with_ffmpeg(
        [str(red), str(blue)], str(output), 1, str(tmp_path), crossfade_seconds=0.4
    )
    with VideoFileClip(str(output), audio=False) as clip:
        assert clip.duration == pytest.approx(1.6, abs=0.15)
        start = clip.get_frame(0.1)[32, 32]
        middle = clip.get_frame(0.8)[32, 32]
        end = clip.get_frame(clip.duration - 0.15)[32, 32]
    assert start[0] > 200 and start[2] < 60
    assert middle[0] > 40 and middle[2] > 40, "mid-transition frames blend both clips"
    assert end[2] > 200 and end[0] < 60


def test_concat_without_crossfade_keeps_hard_cuts(tmp_path):
    red, blue, output = tmp_path / "r.mp4", tmp_path / "b.mp4", tmp_path / "o.mp4"
    _solid_clip(red, "red")
    _solid_clip(blue, "blue")
    video.concat_video_clips_with_ffmpeg([str(red), str(blue)], str(output), 1, str(tmp_path))
    with VideoFileClip(str(output), audio=False) as clip:
        assert clip.duration == pytest.approx(2.0, abs=0.15)


def _silence(path, seconds):
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\0\0" * int(24000 * seconds))


def test_short_footage_is_varied_before_repeating_and_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(VideoAspect, "to_resolution", lambda self: (64, 64))
    monkeypatch.setattr(video, "_get_effective_video_codec", lambda *args: "libx264")
    config.app["video_crossfade_seconds"] = 0
    source, audio, output = tmp_path / "s.mp4", tmp_path / "a.wav", tmp_path / "o.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=s=64x64:r=10:d=0.8", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(source))
    _silence(audio, 3.0)
    warnings = []
    video.combine_videos(
        str(output), [str(source)], str(audio), video_concat_mode=VideoConcatMode.sequential,
        threads=1, warnings=warnings,
    )
    assert [w["code"] for w in warnings] == ["footage_repeated"]
    assert warnings[0]["repeated_clips"] >= 3
    assert warnings[0]["variant_clips"] >= 1
    with VideoFileClip(str(output), audio=False) as clip:
        assert clip.duration == pytest.approx(3.0, abs=0.2)
        original = clip.get_frame(0.3)
        mirrored = clip.get_frame(0.8 + 0.3)
    assert not np.array_equal(original, mirrored), "first repeat must not replay identical frames"
    assert not list(tmp_path.glob("temp-clip-*"))


def test_sufficient_footage_reports_no_repeat_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(VideoAspect, "to_resolution", lambda self: (64, 64))
    monkeypatch.setattr(video, "_get_effective_video_codec", lambda *args: "libx264")
    source, audio, output = tmp_path / "s.mp4", tmp_path / "a.wav", tmp_path / "o.mp4"
    _solid_clip(source, "green", seconds=4)
    _silence(audio, 1.0)
    warnings = []
    video.combine_videos(str(output), [str(source)], str(audio), threads=1, warnings=warnings,
                         max_clip_duration=2)
    assert warnings == []


def test_default_crossfade_is_configurable():
    config.app.pop("video_crossfade_seconds", None)
    assert video._get_default_crossfade_seconds() == pytest.approx(0.3)
    config.app["video_crossfade_seconds"] = 0
    assert video._get_default_crossfade_seconds() == 0
    config.app["video_crossfade_seconds"] = "bad"
    assert video._get_default_crossfade_seconds() == pytest.approx(0.3)


# --- defaults ----------------------------------------------------------------


def test_subtitle_stroke_default_is_thicker_and_ducking_defaults_on():
    params = VideoParams(video_subject="x")
    assert params.stroke_width == 3.0
    assert params.bgm_ducking is True


# --- multi-scene project A/V alignment ---------------------------------------


def test_project_export_keeps_audio_and_video_aligned_across_scenes(tmp_path):
    from app.services.video_project import VideoProject

    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe required")
    scenes = []
    # Durations that are not frame- or packet-aligned accumulate drift when
    # scenes are stream-copied together.
    for index, seconds in enumerate((0.37, 0.53, 0.41, 0.59, 0.47, 0.61)):
        _solid_clip(tmp_path / f"v{index}.mp4", "red", seconds=1)
        _ffmpeg("-f", "lavfi", "-i", f"sine=frequency={300 + index * 50}:sample_rate=48000:duration={seconds}",
                str(tmp_path / f"a{index}.wav"))
        scenes.append({"id": f"s{index}", "footage": f"v{index}.mp4",
                       "audio": f"a{index}.wav", "narration": f"scene {index}"})
    request = {"project_id": "drift", "settings": {"width": 64, "height": 64, "fps": 10},
               "scenes": scenes}
    project = VideoProject(tmp_path / "project")
    revision = project.create_revision(request, relative_to=tmp_path)
    status = project.render(revision["revision_id"])
    export = Path(status["export"]["path"])
    video_seconds = float(_probe(export, "duration")["duration"])
    audio_seconds = float(_probe(export, "duration", stream="a:0")["duration"])
    expected = sum(round(s * 10 + 0.4999) / 10 for s in (0.37, 0.53, 0.41, 0.59, 0.47, 0.61))
    assert abs(video_seconds - audio_seconds) < 0.06
    assert abs(video_seconds - expected) < 0.15
    assert _probe(export, "pix_fmt")["pix_fmt"] == "yuv420p"
    assert b"crf=20.0" in export.read_bytes()
