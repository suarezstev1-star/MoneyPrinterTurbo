"""Read SRT cue boundaries before adapting them to MoviePy timing tuples."""
import os
import re

from loguru import logger

# SRT uses a comma before the milliseconds; many tools emit a period (WebVTT
# style). Accept both so such cues are not silently dropped.
_TIMESTAMP = r"[0-9]+:[0-9]+:[0-9]+[,.][0-9]+"
_TIMESTAMP_RE = re.compile(f"({_TIMESTAMP})")


def _timestamps(line):
    return _TIMESTAMP_RE.findall(line)


def _is_timing_line(line):
    """A timing line carries a start and an end timestamp joined by ``-->``."""
    return "-->" in line and len(_timestamps(line)) >= 2


def file_to_subtitles(filename):
    if not filename or not os.path.isfile(filename):
        return []

    times_texts = []
    current_times = None
    current_text = ""
    index = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            # Once a cue has started, timestamps belong to its text until the
            # blank-line separator; they must not overwrite the cue's timing.
            if current_times is None:
                if _is_timing_line(line):
                    current_times = line
                elif _timestamps(line):
                    logger.warning(
                        f"skipping malformed subtitle timing line: {line.strip()!r}"
                    )
            elif line.strip() == "":
                index += 1
                times_texts.append((index, current_times.strip(), current_text.strip()))
                current_times, current_text = None, ""
            else:
                current_text += line

    # Flush the final block. SRT files whose last subtitle is not followed by a
    # trailing blank line never hit the blank-line branch above, so without this
    # the last subtitle would be silently dropped.
    if current_times:
        index += 1
        times_texts.append((index, current_times.strip(), current_text.strip()))
    return times_texts


def moviepy_subtitles(filename):
    """Convert only cue headers; timestamp-bearing text remains caption text."""
    from moviepy.tools import convert_to_seconds

    cues = []
    for _index, timing, text in file_to_subtitles(filename):
        stamps = _timestamps(timing)
        if len(stamps) < 2:
            logger.warning(f"skipping malformed subtitle cue: {timing!r}")
            continue
        try:
            start = convert_to_seconds(stamps[0].replace(",", "."))
            end = convert_to_seconds(stamps[1].replace(",", "."))
        except Exception as exc:
            logger.warning(f"skipping unparseable subtitle cue {timing!r}: {exc}")
            continue
        cues.append(((start, end), text))
    return cues
