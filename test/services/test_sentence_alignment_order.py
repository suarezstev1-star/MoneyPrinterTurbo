import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.models.schema import VideoConcatMode
from app.services import task_artifacts


class TestSentenceAlignmentRecorded(unittest.TestCase):
    """对齐后的素材已按叙事排序，最终拼接不能再随机打乱。"""

    def _with_script(self, payload):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        script = Path(tmp.name) / "script.json"
        if payload is not None:
            script.write_text(payload, encoding="utf-8")
        return patch.object(task_artifacts, "_script_file", return_value=script)

    def test_recorded_alignment_is_detected(self):
        payload = json.dumps({"sentence_alignment": [{"local_file": "a.mp4"}]})
        with self._with_script(payload):
            self.assertTrue(task_artifacts.has_sentence_alignment("t1"))

    def test_empty_or_missing_alignment_is_not_detected(self):
        with self._with_script(json.dumps({"sentence_alignment": []})):
            self.assertFalse(task_artifacts.has_sentence_alignment("t1"))
        with self._with_script(json.dumps({"script": "x"})):
            self.assertFalse(task_artifacts.has_sentence_alignment("t1"))

    def test_unreadable_manifest_keeps_previous_behaviour(self):
        with self._with_script(None):  # file does not exist
            self.assertFalse(task_artifacts.has_sentence_alignment("t1"))
        with self._with_script("not json{"):
            self.assertFalse(task_artifacts.has_sentence_alignment("t1"))
        with self._with_script(json.dumps(["not", "an", "object"])):
            self.assertFalse(task_artifacts.has_sentence_alignment("t1"))


class TestConcatModeUsesAlignment(unittest.TestCase):
    def _mode_for(self, *, video_count, aligned, match=False, configured="random"):
        from app.models.schema import VideoParams
        from app.services import task as tm

        params = VideoParams(
            video_subject="Coffee",
            video_count=video_count,
            video_concat_mode=configured,
            match_materials_to_script=match,
        )
        captured = {}

        def fake_combine(*args, **kwargs):
            captured["mode"] = kwargs.get("video_concat_mode")
            raise RuntimeError("stop after capturing the mode")

        with (
            patch.object(task_artifacts, "has_sentence_alignment", return_value=aligned),
            patch.object(tm.video, "combine_videos", side_effect=fake_combine),
            patch.object(tm.sm, "state"),
            patch.object(tm.bgm_service, "should_use_bgm", return_value=False),
        ):
            try:
                tm.generate_final_videos(
                    "task-x", params, ["a.mp4"], "audio.mp3", "", 5.0
                )
            except Exception:
                pass
        return captured.get("mode")

    def test_single_aligned_video_is_concatenated_in_order(self):
        self.assertEqual(
            self._mode_for(video_count=1, aligned=True), VideoConcatMode.sequential
        )

    def test_unaligned_single_video_keeps_configured_mode(self):
        self.assertEqual(
            self._mode_for(video_count=1, aligned=False), VideoConcatMode.random
        )

    def test_batches_are_not_forced_sequential_by_alignment(self):
        self.assertEqual(
            self._mode_for(video_count=2, aligned=True), VideoConcatMode.random
        )


if __name__ == "__main__":
    unittest.main()
