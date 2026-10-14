import difflib
import json
import os.path
import re
import tempfile
import unicodedata
import threading
from timeit import default_timer as timer

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None
from loguru import logger

from app.config import config
from app.models import const
from app.utils import utils
from app.utils.subtitle_reader import file_to_subtitles as file_to_subtitles
from app.utils.subtitle_writer import write_subtitle_file

model_size = config.whisper.get("model_size", "large-v3")
device = config.whisper.get("device", "cpu")
compute_type = config.whisper.get("compute_type", "int8")
initial_prompt = config.whisper.get("initial_prompt", "") or None
model = None
_model_init_lock = threading.Lock()


def _ensure_model_loaded() -> bool:
    """Load the large Whisper model once, even when jobs start concurrently."""
    global model
    if model is not None:
        return True

    with _model_init_lock:
        if model is not None:
            return True
        model_path = f"{utils.root_dir()}/models/whisper-{model_size}"
        model_bin_file = f"{model_path}/model.bin"
        if not os.path.isdir(model_path) or not os.path.isfile(model_bin_file):
            model_path = model_size

        logger.info(
            f"loading model: {model_path}, device: {device}, compute_type: {compute_type}"
        )
        try:
            model = WhisperModel(
                model_size_or_path=model_path, device=device, compute_type=compute_type
            )
        except Exception as e:
            logger.error(
                f"failed to load model: {e} \n\n"
                f"********************************************\n"
                f"this may be caused by network issue. \n"
                f"please download the model manually and put it in the 'models' folder. \n"
                f"see [README.md FAQ](https://github.com/harry0703/MoneyPrinterTurbo) for more details.\n"
                f"********************************************\n\n"
            )
            return False
    return True


def create(
    audio_file,
    subtitle_file: str = "",
    word_level: bool = False,
    log_details: bool = True,
):
    if WhisperModel is None:
        logger.warning("faster_whisper not available, skipping whisper subtitle generation")
        return ""
    if not _ensure_model_loaded():
        return None

    if log_details:
        logger.info(f"start, output file: {subtitle_file}")
    if not subtitle_file:
        subtitle_file = f"{audio_file}.srt"

    segments, info = model.transcribe(
        audio_file,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        **({"initial_prompt": initial_prompt} if initial_prompt else {}),
    )

    logger.info(
        f"detected language: '{info.language}', probability: {info.language_probability:.2f}"
    )

    start = timer()
    subtitles = []

    def recognized(seg_text, seg_start, seg_end):
        seg_text = seg_text.strip()
        if not seg_text:
            return

        if log_details:
            msg = "[%.2fs -> %.2fs] %s" % (seg_start, seg_end, seg_text)
            logger.debug(msg)

        subtitles.append(
            {"msg": seg_text, "start_time": seg_start, "end_time": seg_end}
        )

    for segment in segments:
        if not segment.words:
            # Faster Whisper exposes words as Optional[List[Word]]. A segment
            # can still have usable text and timestamps when alignment yields
            # no words; keep that subtitle in both sentence and word modes.
            recognized(segment.text, segment.start, segment.end)
            continue

        if word_level:
            for word in segment.words:
                cleaned_word = word.word.strip()
                if cleaned_word:
                    recognized(cleaned_word, word.start, word.end)
            continue

        words_idx = 0
        words_len = len(segment.words)

        seg_start = 0
        seg_end = 0
        seg_text = ""

        if segment.words:
            is_segmented = False
            for word in segment.words:
                if not is_segmented:
                    seg_start = word.start
                    is_segmented = True

                seg_end = word.end
                # Accumulate words; only trailing punctuation ends a sentence.
                seg_text += word.word

                if word.word.rstrip().endswith(tuple(const.PUNCTUATIONS)):
                    # Punctuation inside a word (3.14, 1,000, 12:30) is not a
                    # sentence boundary. Remove only actual trailing delimiters.
                    seg_text = seg_text.rstrip().rstrip("".join(const.PUNCTUATIONS))
                    recognized(seg_text, seg_start, seg_end)

                    is_segmented = False
                    seg_text = ""

                if words_idx == 0 and segment.start < word.start:
                    seg_start = word.start
                if words_idx == (words_len - 1) and segment.end > word.end:
                    seg_end = word.end
                words_idx += 1

        if not seg_text:
            continue

        recognized(seg_text, seg_start, seg_end)

    end = timer()

    diff = end - start
    logger.info(f"complete, elapsed: {diff:.2f} s")

    idx = 1
    lines = []
    for subtitle in subtitles:
        text = subtitle.get("msg")
        if text:
            lines.append(
                utils.text_to_srt(
                    idx, text, subtitle.get("start_time"), subtitle.get("end_time")
                )
            )
            idx += 1

    sub = "\n".join(lines) + "\n"
    if not lines:
        logger.warning("transcription produced no subtitle cues")
        return
    write_subtitle_file(subtitle_file, sub)
    if log_details:
        logger.info(f"subtitle file created: {subtitle_file}")


def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Transcribe in-memory WAV audio and remove every temporary artifact."""
    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        return ""

    with tempfile.TemporaryDirectory(prefix="whisper-transcript-") as temp_dir:
        audio_file = os.path.join(temp_dir, "reference.wav")
        subtitle_file = os.path.join(temp_dir, "reference.srt")
        with open(audio_file, "wb") as output:
            output.write(audio_bytes)

        create_result = create(audio_file, subtitle_file, log_details=False)
        if create_result in (None, "") and not os.path.isfile(subtitle_file):
            return ""
        subtitle_items = file_to_subtitles(subtitle_file)
        return " ".join(item[2].strip() for item in subtitle_items if item[2].strip())


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity(a, b):
    distance = levenshtein_distance(a.lower(), b.lower())
    max_length = max(len(a), len(b))
    return 1 - (distance / max_length)


_TOKEN_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]|[^\W_]+"
)
_TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")
_MIN_CUE_SECONDS = 0.001


def _tokenize(text):
    """Words for alphabetic scripts, single characters for CJK; case/punctuation-free."""
    return [t.lower() for t in _TOKEN_RE.findall(unicodedata.normalize("NFKC", text or ""))]


def _token_weight(token):
    return 3.0 if len(token) == 1 and ord(token) > 0x2E80 else max(len(token), 1)


def _parse_srt_time(value):
    match = _TIME_RE.search(value or "")
    if not match:
        return None
    hours, minutes, seconds, fraction = match.groups()
    millis = int((fraction + "000")[:3])
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + millis / 1000.0


def _format_srt_time(seconds):
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def align_script_to_cues(script_lines, cues):
    """
    Assign a (start, end) in seconds to every script line, monotonically.

    ``cues`` is a list of ``(start, end, text)``. Script and transcript words are
    aligned with a longest-matching-blocks pass, so a cue that merged two script
    lines (or a line split over cues) no longer shifts every later subtitle. Lines
    without any matching word are spread proportionally over the audio that is
    left between their matched neighbours. No returned span has zero duration.
    """
    script_tokens, token_line = [], []
    for line_index, line in enumerate(script_lines):
        for token in _tokenize(line):
            script_tokens.append(token)
            token_line.append(line_index)

    # Every transcript token gets an interpolated time slice of its cue.
    cue_tokens, cue_spans = [], []
    for start, end, text in cues:
        tokens = _tokenize(text)
        if not tokens:
            continue
        step = (end - start) / len(tokens)
        for position, token in enumerate(tokens):
            cue_tokens.append(token)
            cue_spans.append((start + step * position, start + step * (position + 1)))

    audio_start = cues[0][0] if cues else 0.0
    audio_end = max((c[1] for c in cues), default=audio_start)
    spans = [None] * len(script_lines)

    if script_tokens and cue_tokens:
        matcher = difflib.SequenceMatcher(None, script_tokens, cue_tokens, autojunk=False)
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                line_index = token_line[block.a + offset]
                start, end = cue_spans[block.b + offset]
                current = spans[line_index]
                spans[line_index] = (
                    (start, end) if current is None
                    else (min(current[0], start), max(current[1], end))
                )

    line_weight = [
        sum(_token_weight(t) for t in _tokenize(line)) or 1.0 for line in script_lines
    ]

    result = []
    cursor = audio_start
    index = 0
    while index < len(script_lines):
        if spans[index] is not None:
            start = max(spans[index][0], cursor)
            end = max(spans[index][1], start + _MIN_CUE_SECONDS)
            result.append((start, end))
            cursor = end
            index += 1
            continue
        # A run of unmatched lines: share the room before the next anchor.
        run_end = index
        while run_end < len(script_lines) and spans[run_end] is None:
            run_end += 1
        limit = spans[run_end][0] if run_end < len(script_lines) else audio_end
        limit = max(limit, cursor + _MIN_CUE_SECONDS * (run_end - index))
        total = sum(line_weight[index:run_end])
        room = limit - cursor
        for position in range(index, run_end):
            share = room * line_weight[position] / total
            start, end = cursor, cursor + max(share, _MIN_CUE_SECONDS)
            result.append((start, end))
            cursor = end
        index = run_end
    return result


def correct(subtitle_file, video_script):
    subtitle_items = file_to_subtitles(subtitle_file)
    normalized_script = utils.normalize_script_for_subtitle_matching(video_script)
    script_lines = [
        line.strip()
        for line in utils.split_string_by_punctuations(normalized_script)
        if line.strip()
    ]
    if not script_lines or not subtitle_items:
        logger.warning("nothing to correct: empty script or transcription")
        return

    cues = []
    for _index, timing, text in subtitle_items:
        parts = re.split(r"\s*-->\s*", timing)
        start = _parse_srt_time(parts[0]) if parts else None
        end = _parse_srt_time(parts[1]) if len(parts) > 1 else None
        if start is None or end is None:
            continue
        cues.append((start, max(end, start), text))

    if [c[2].strip() for c in cues] == script_lines:
        logger.success("Subtitle is correct")
        return

    spans = align_script_to_cues(script_lines, cues)
    content = "".join(
        f"{i + 1}\n{_format_srt_time(start)} --> {_format_srt_time(end)}\n{line}\n\n"
        for i, (line, (start, end)) in enumerate(zip(script_lines, spans))
    )
    write_subtitle_file(subtitle_file, content)
    logger.info("Subtitle corrected")


if __name__ == "__main__":
    task_id = "c12fd1e6-4b0a-4d65-a075-c87abe35a072"
    task_dir = utils.task_dir(task_id)
    subtitle_file = f"{task_dir}/subtitle.srt"
    audio_file = f"{task_dir}/audio.mp3"

    subtitles = file_to_subtitles(subtitle_file)
    print(subtitles)

    script_file = f"{task_dir}/script.json"
    with open(script_file, "r") as f:
        script_content = f.read()
    s = json.loads(script_content)
    script = s.get("script")

    correct(subtitle_file, script)

    subtitle_file = f"{task_dir}/subtitle-test.srt"
    create(audio_file, subtitle_file)
