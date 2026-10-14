import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from edge_tts.srt_composer import Subtitle

from app.services import subtitle, task as tm, voice
from app.utils.subtitle_reader import file_to_subtitles, moviepy_subtitles


def _maker(words):
    cues = [
        Subtitle(index=i + 1, start=timedelta(seconds=s), end=timedelta(seconds=e), content=w)
        for i, (w, s, e) in enumerate(words)
    ]
    return SimpleNamespace(cues=cues)


def test_mktimestamp_never_rolls_to_sixty_seconds():
    assert voice.mktimestamp(59.9996 * 10**7) == "00:01:00,000"
    assert voice.mktimestamp(3599.9996 * 10**7) == "01:00:00,000"
    assert voice.mktimestamp(1.5 * 10**7) == "00:00:01,500"


def test_unspeakable_line_does_not_drop_other_subtitles(tmp_path):
    maker = _maker([("Hello", 0, 1), (" world", 1, 2), ("Bye", 2, 3), (" now", 3, 4)])
    destination = tmp_path / "s.srt"
    result = voice.create_subtitle(maker, "Hello world.\n\U0001F680\nBye now.", str(destination))
    assert result["written"] and result["matched"] == result["total"] == 2
    assert [c[2] for c in file_to_subtitles(str(destination))] == ["Hello world", "Bye now"]


def test_unmatched_line_resyncs_and_reports_partial(tmp_path):
    maker = _maker([
        ("First", 0, 1), (" line", 1, 2),
        ("twenty", 2, 3), (" five", 3, 4),
        ("Last", 4, 5), (" one", 5, 6),
    ])
    destination = tmp_path / "s.srt"
    result = voice.create_subtitle(maker, "First line.\nPay $25 today.\nLast one.", str(destination))
    assert result["written"]
    assert result["matched"] == 2 and result["total"] == 3
    cues = file_to_subtitles(str(destination))
    assert [c[2] for c in cues] == ["First line", "Last one"]
    assert cues[1][1] == "00:00:04,000 --> 00:00:06,000"


def test_nothing_matching_reports_not_written(tmp_path):
    destination = tmp_path / "s.srt"
    result = voice.create_subtitle(_maker([("abc", 0, 1)]), "Totally different text here.", str(destination))
    assert not result["written"]


def test_whisper_merged_cue_keeps_later_lines_aligned(tmp_path):
    srt = (
        "1\n00:00:00,000 --> 00:00:04,000\nHello there my friend welcome\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\nto the show\n\n"
        "3\n00:00:06,000 --> 00:00:08,000\nGoodbye everyone\n\n"
    )
    path = tmp_path / "s.srt"
    path.write_text(srt, encoding="utf-8")
    subtitle.correct(str(path), "Hello there.\nMy friend welcome.\nTo the show.\nGoodbye everyone.")
    cues = file_to_subtitles(str(path))
    assert [c[2] for c in cues] == ["Hello there", "My friend welcome", "To the show", "Goodbye everyone"]
    assert cues[3][1] == "00:00:06,000 --> 00:00:08,000"
    for _, timing, _ in cues:
        start, end = timing.split(" --> ")
        assert start != end


def test_whisper_extra_script_lines_never_zero_duration(tmp_path):
    path = tmp_path / "s.srt"
    path.write_text("1\n00:00:00,100 --> 00:00:01,000\nWrong text\n\n", encoding="utf-8")
    subtitle.correct(str(path), "Expected sentence. Extra sentence. Third one.")
    cues = moviepy_subtitles(str(path))
    assert len(cues) == 3
    previous_end = 0
    for (start, end), _ in cues:
        assert end > start and start >= previous_end - 1e-9
        previous_end = end


def test_reader_accepts_period_separator_and_skips_malformed(tmp_path):
    path = tmp_path / "s.srt"
    path.write_text(
        "1\n00:00:01.500 --> 00:00:02.500\nDot cue\n\n"
        "2\n00:00:03,000\nBroken\n\n"
        "3\n00:00:04,000 --> 00:00:05,000\nGood\n\n",
        encoding="utf-8",
    )
    cues = moviepy_subtitles(str(path))
    assert [t for _, t in cues] == ["Dot cue", "Good"]
    assert cues[0][0] == (1.5, 2.5)


def test_edge_timeout_is_idle_not_total():
    class Slow:
        async def stream(self):
            for _ in range(6):
                await asyncio.sleep(0.1)
                yield {"type": "audio", "data": b"x"}

    chunks = []
    voice._stream_edge_tts_sync_with_timeout(Slow(), chunks.append, 0.35)
    assert len(chunks) == 6  # 0.6s total exceeds the 0.35s idle limit

    class Stalled:
        async def stream(self):
            yield {"type": "audio", "data": b"x"}
            await asyncio.sleep(5)
            yield {"type": "audio", "data": b"y"}

    with pytest.raises(TimeoutError):
        voice._stream_edge_tts_sync_with_timeout(Stalled(), lambda c: None, 0.2)


def test_estimated_timing_weights_cjk_heavier_than_latin():
    maker = voice.populate_legacy_submaker_with_full_text(
        voice.ensure_legacy_submaker_fields(SimpleNamespace()),
        "你好世界你好。Hello there.",
        10.0,
    )
    (s0, e0), (s1, e1) = maker.offset
    assert e0 > s0 and e1 == 10 * 10**7
    # 6 CJK chars take longer than 11 Latin letters under script-aware weights.
    assert (e0 - s0) > (e1 - s1) * 0.8


def _params():
    return SimpleNamespace(subtitle_enabled=True, subtitle_display_mode="sentence")


def test_generate_subtitle_warns_when_skipped_and_partial(tmp_path):
    with patch.object(tm.config, "app", {"subtitle_provider": "edge"}), \
         patch.object(tm.utils, "task_dir", return_value=str(tmp_path)):
        warnings = []
        out = tm.generate_subtitle(
            "t", _params(), "Nothing matches here.", _maker([("zzz", 0, 1)]), "a.mp3",
            warnings=warnings,
        )
        assert out == "" and warnings[0]["code"] == "subtitles_skipped"

        warnings = []
        maker = _maker([("Hello", 0, 1), ("Bye", 2, 3)])
        out = tm.generate_subtitle(
            "t", _params(), "Hello.\nUnheard line.\nBye.", maker, "a.mp3", warnings=warnings,
        )
        assert out and [w["code"] for w in warnings] == ["subtitles_partial"]
        assert warnings[0]["matched"] == 2 and warnings[0]["total"] == 3
