import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect, VideoParams
from app.services import (
    cache_manager,
    elevenlabs_music,
    material,
    task as task_service,
    volcengine_seedance as seedance,
)
from app.utils import download_deadline


def _item(term, url, duration=5, provider="volcengine_seedance"):
    return MaterialInfo(
        provider=provider,
        url=url,
        duration=duration,
        source_info={
            "provider": provider,
            "search_term": term,
            "asset_id": f"asset-{url.rsplit('/', 1)[-1]}",
        },
    )


class _ConfigIsolation(unittest.TestCase):
    def setUp(self):
        self.original_app = dict(config.app)
        self.addCleanup(self._restore)

    def _restore(self):
        config.app.clear()
        config.app.update(self.original_app)


class TestAiClipPlanning(_ConfigIsolation):
    def test_terms_are_extended_to_cover_narration(self):
        planned = material._plan_ai_generation_terms(["a", "b"], 30, 5)
        self.assertGreaterEqual(len(planned), 6)
        self.assertEqual(planned[:2], ["a", "b"])
        self.assertEqual(len(set(planned)), len(planned))
        self.assertTrue(planned[2].startswith("a, "))

    def test_cap_limits_paid_clips(self):
        config.app["max_ai_clips_per_task"] = 4
        self.assertEqual(len(material._plan_ai_generation_terms(["a"], 600, 5)), 4)
        self.assertEqual(
            material._plan_ai_generation_terms(list("abcdef"), 10, 5), list("abcd")
        )

    def test_original_terms_always_come_first(self):
        # Extra planned terms are free: the loop stops once coverage is reached.
        planned = material._plan_ai_generation_terms(["a", "b", "c"], 10, 5)
        self.assertEqual(planned[:3], ["a", "b", "c"])

    def test_warning_only_when_short(self):
        warnings = []
        self.assertFalse(material._note_insufficient_materials(warnings, "x", 10, 10))
        self.assertTrue(material._note_insufficient_materials(warnings, "x", 10, 4))
        self.assertEqual(warnings[0]["code"], "insufficient_materials")
        self.assertEqual(warnings[0]["covered_duration"], 4.0)


class TestPaidGenerationLoops(_ConfigIsolation):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(material.utils, "task_dir", return_value=self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        persist = patch.object(material, "_persist_material_sources")
        persist.start()
        self.addCleanup(persist.stop)

    def _run(self, terms, audio, warnings=None, task_id="t1"):
        return material.download_videos(
            task_id=task_id,
            search_terms=terms,
            source="volcengine_seedance",
            audio_duration=audio,
            max_clip_duration=5,
            warnings=warnings,
        )

    def _saver(self):
        def save(url, directory, provider):
            path = os.path.join(directory, url.rsplit("/", 1)[-1])
            Path(path).write_bytes(b"clip")
            return path

        return save

    def test_terms_cycle_until_duration_is_covered(self):
        calls = []

        def generate(search_term, minimum_duration, video_aspect):
            calls.append(search_term)
            return [_item(search_term, f"https://cdn/{len(calls)}.mp4")]

        warnings = []
        with (
            patch.object(seedance, "generate_videos", side_effect=generate),
            patch.object(material, "_save_generated_video_with_retry", side_effect=self._saver()),
            patch.object(material, "_get_downloaded_video_duration", return_value=5.0),
        ):
            paths = self._run(["one", "two"], 20, warnings)

        self.assertEqual(len(paths), 4)
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[:2], ["one", "two"])
        self.assertEqual(warnings, [])

    def test_cap_leaves_insufficient_materials_warning(self):
        config.app["max_ai_clips_per_task"] = 2
        warnings = []
        with (
            patch.object(
                seedance,
                "generate_videos",
                side_effect=lambda **kw: [_item(kw["search_term"], f"https://cdn/{kw['search_term'][:3]}{len(kw['search_term'])}.mp4")],
            ),
            patch.object(material, "_save_generated_video_with_retry", side_effect=self._saver()),
            patch.object(material, "_get_downloaded_video_duration", return_value=5.0),
        ):
            paths = self._run(["one"], 60, warnings)

        self.assertEqual(len(paths), 2)
        self.assertEqual(warnings[0]["code"], "insufficient_materials")
        self.assertEqual(warnings[0]["required_duration"], 60.0)
        self.assertEqual(warnings[0]["covered_duration"], 10.0)

    def test_delivered_duration_is_counted_not_requested(self):
        generate_calls = []

        def generate(search_term, minimum_duration, video_aspect):
            generate_calls.append(search_term)
            return [_item(search_term, f"https://cdn/{len(generate_calls)}.mp4")]

        with (
            patch.object(seedance, "generate_videos", side_effect=generate),
            patch.object(material, "_save_generated_video_with_retry", side_effect=self._saver()),
            patch.object(material, "_get_downloaded_video_duration", return_value=2.0),
        ):
            # Requested 5s clips are really 2s: 10s of narration needs 5 clips.
            paths = self._run(["a", "b", "c", "d", "e", "f"], 10)

        self.assertEqual(len(paths), 5)

    def test_rejected_term_is_skipped_and_paid_clips_are_kept(self):
        outcomes = [
            [_item("one", "https://cdn/1.mp4")],
            seedance.VolcEngineSeedanceError(
                "Seedance task did not produce a video: "
                "detail=OutputVideoSensitiveContentDetected",
                task_id="cgt-bad",
            ),
            [_item("three", "https://cdn/3.mp4")],
        ]

        def generate(**kwargs):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with (
            patch.object(seedance, "generate_videos", side_effect=generate),
            patch.object(material, "_save_generated_video_with_retry", side_effect=self._saver()),
            patch.object(material, "_get_downloaded_video_duration", return_value=5.0),
        ):
            paths = self._run(["one", "two", "three"], 10)

        self.assertEqual([os.path.basename(p) for p in paths], ["1.mp4", "3.mp4"])

    def test_non_policy_error_still_aborts(self):
        with (
            patch.object(
                seedance,
                "generate_videos",
                side_effect=seedance.VolcEngineSeedanceError("bad api key"),
            ),
            self.assertRaises(seedance.VolcEngineSeedanceError),
        ):
            self._run(["one", "two"], 10)

    def test_provider_that_rejects_everything_stops_early(self):
        calls = []

        def generate(**kwargs):
            calls.append(kwargs["search_term"])
            raise seedance.VolcEngineSeedanceError("content policy violation")

        with patch.object(seedance, "generate_videos", side_effect=generate):
            paths = self._run(["a"], 100)

        self.assertEqual(paths, [])
        self.assertEqual(len(calls), material._MAX_CONSECUTIVE_MISSES)

    def test_retry_reuses_clips_already_paid_for(self):
        saved = self._saver()
        first_calls = []

        def generate_first(search_term, minimum_duration, video_aspect):
            first_calls.append(search_term)
            return [_item(search_term, f"https://cdn/{search_term}.mp4")]

        def save_then_fail(url, directory, provider):
            if url.endswith("two.mp4"):
                return ""
            return saved(url, directory, provider)

        with (
            patch.object(seedance, "generate_videos", side_effect=generate_first),
            patch.object(material, "_save_generated_video_with_retry", side_effect=save_then_fail),
            patch.object(material, "_get_downloaded_video_duration", return_value=5.0),
        ):
            with self.assertRaises(seedance.VolcEngineSeedanceDownloadError):
                self._run(["one", "two"], 10)
        self.assertEqual(first_calls, ["one", "two"])
        ledger = material._read_paid_ledger("t1")
        self.assertEqual(len(ledger), 1)
        self.assertEqual(next(iter(ledger.values()))["asset_id"], "asset-one.mp4")

        retry_calls = []

        def generate_retry(search_term, minimum_duration, video_aspect):
            retry_calls.append(search_term)
            return [_item(search_term, f"https://cdn/{search_term}.mp4")]

        with (
            patch.object(seedance, "generate_videos", side_effect=generate_retry),
            patch.object(material, "_save_generated_video_with_retry", side_effect=saved),
            patch.object(material, "_get_downloaded_video_duration", return_value=5.0),
        ):
            paths = self._run(["one", "two"], 10)

        self.assertEqual(retry_calls, ["two"])
        self.assertEqual([os.path.basename(p) for p in paths], ["one.mp4", "two.mp4"])


class TestDownloadDeadline(_ConfigIsolation):
    def test_configured_deadline_falls_back_on_bad_values(self):
        for bad in ("x", -1, 0, float("nan")):
            config.app["download_total_timeout_seconds"] = bad
            self.assertEqual(
                download_deadline.configured_deadline_seconds(),
                download_deadline.DEFAULT_DOWNLOAD_DEADLINE_SECONDS,
            )

    def test_save_video_aborts_and_removes_temp_file(self):
        config.app["download_total_timeout_seconds"] = 0.05

        class SlowResponse:
            headers = {}

            def raise_for_status(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def iter_content(self, chunk_size):
                import time

                while True:
                    time.sleep(0.03)
                    yield b"x"

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(material.requests, "get", return_value=SlowResponse()):
                with self.assertRaises(download_deadline.DownloadDeadlineExceeded):
                    material.save_video("https://cdn.example.com/slow.mp4", directory)
            self.assertEqual(os.listdir(directory), [])

    def test_elevenlabs_stream_enforces_total_deadline(self):
        config.app["download_total_timeout_seconds"] = 0.05

        class SlowResponse:
            def iter_content(self, chunk_size):
                import time

                while True:
                    time.sleep(0.03)
                    yield b"x"

        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "audio.mp3")
            with self.assertRaises(elevenlabs_music.ElevenLabsMusicError) as raised:
                elevenlabs_music._stream_audio(SlowResponse(), target)
        self.assertIn("total time limit", str(raised.exception))


class TestCacheSweep(_ConfigIsolation):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache_dir = Path(self.tmp.name)
        patcher = patch.object(
            cache_manager.utils, "storage_dir", return_value=str(self.cache_dir)
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        cache_manager._last_sweep_at = None
        self.addCleanup(setattr, cache_manager, "_last_sweep_at", None)
        self.now = 2_000_000_000.0

    def _file(self, name, size, age_seconds):
        path = self.cache_dir / name
        path.write_bytes(b"x" * size)
        mtime = self.now - age_seconds
        os.utime(path, (mtime, mtime))
        return path

    def test_stale_temp_files_are_removed_but_fresh_ones_kept(self):
        stale = self._file(f".vid-{'a' * 32}-abc123.mp4", 5, 7 * 3600)
        fresh = self._file(f".vid-{'b' * 32}-abc123.mp4", 5, 600)
        result = cache_manager.sweep(force=True, now=self.now)
        self.assertEqual(result.stale_temp_removed, 1)
        self.assertFalse(stale.exists())
        self.assertTrue(fresh.exists())

    def test_old_clips_expire_and_unmanaged_files_survive(self):
        old = self._file(f"vid-{'a' * 32}.mp4", 5, 40 * 86400)
        new = self._file(f"vid-{'b' * 32}.mp4", 5, 3600 * 5)
        other = self._file("personal.mp4", 5, 400 * 86400)
        result = cache_manager.sweep(force=True, now=self.now)
        self.assertEqual(result.expired_removed, 1)
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())
        self.assertTrue(other.exists())

    def test_size_limit_evicts_least_recently_used_first(self):
        config.app["cache_max_size_gb"] = 10 / 1024**3
        oldest = self._file(f"vid-{'a' * 32}.mp4", 6, 5 * 86400)
        middle = self._file(f"vid-{'b' * 32}.mp4", 6, 3 * 86400)
        newest = self._file(f"vid-{'c' * 32}.mp4", 6, 2 * 86400)
        result = cache_manager.sweep(force=True, now=self.now)
        self.assertEqual(result.evicted_removed, 2)
        self.assertFalse(oldest.exists())
        self.assertFalse(middle.exists())
        self.assertTrue(newest.exists())

    def test_recently_used_clip_is_never_evicted(self):
        config.app["cache_max_size_gb"] = 1 / 1024**3
        recent = self._file(f"vid-{'a' * 32}.mp4", 50, 60)
        cache_manager.sweep(force=True, now=self.now)
        self.assertTrue(recent.exists())

    def test_sweep_is_rate_limited_to_once_per_hour(self):
        first = cache_manager.sweep(now=self.now)
        again = cache_manager.sweep(now=self.now + 60)
        later = cache_manager.sweep(now=self.now + 3601)
        self.assertFalse(first.skipped)
        self.assertTrue(again.skipped)
        self.assertFalse(later.skipped)


class TestSentenceAlignedSelection(_ConfigIsolation):
    def _stock(self, term, count, duration=10, width=1080, height=1920):
        items = []
        for index in range(count):
            items.append(
                MaterialInfo(
                    provider="pexels",
                    url=f"https://cdn/{term}-{index}.mp4",
                    duration=duration,
                    source_info={
                        "provider": "pexels",
                        "search_term": term,
                        "asset_id": f"{term}-{index}",
                        "rendition": {"width": width, "height": height},
                    },
                )
            )
        return items

    def test_plan_normalization_accepts_strings_dicts_and_pairs(self):
        script = "First sentence. Second sentence."
        plan = material.normalize_sentence_plan(script, ["cats", "dogs"])
        self.assertEqual([e["queries"] for e in plan], [["cats"], ["dogs"]])
        self.assertEqual(plan[0]["sentence"], "First sentence.")
        plan = material.normalize_sentence_plan(
            script, [{"sentence": "x", "queries": ["a", "b"]}, ("y", "c"), {"bad": 1}]
        )
        self.assertEqual([e["queries"] for e in plan], [["a", "b"], ["c"]])
        self.assertIsNone(material.normalize_sentence_plan(script, []))
        self.assertIsNone(material.normalize_sentence_plan(script, "nope"))

    def test_planner_falls_back_when_generator_missing_or_failing(self):
        from app.services import llm

        with patch.object(llm, "generate_sentence_queries", None, create=True):
            self.assertIsNone(material.plan_sentence_queries("Hello there."))
        with patch.object(
            llm, "generate_sentence_queries", side_effect=RuntimeError("boom"), create=True
        ):
            self.assertIsNone(material.plan_sentence_queries("Hello there."))
        with patch.object(
            llm, "generate_sentence_queries", return_value=["sea"], create=True
        ):
            plan = material.plan_sentence_queries("Hello there.")
        self.assertEqual(plan[0]["queries"], ["sea"])

    def test_default_flag(self):
        config.app.pop("match_materials_to_script_default", None)
        self.assertTrue(material.sentence_alignment_enabled(False))
        config.app["match_materials_to_script_default"] = False
        self.assertFalse(material.sentence_alignment_enabled(False))
        self.assertTrue(material.sentence_alignment_enabled(True))

    def test_selection_keeps_narrative_order_and_never_repeats_a_clip(self):
        plan = [
            {"sentence": "a" * 10, "queries": ["sea"], "weight": 10.0},
            {"sentence": "b" * 10, "queries": ["sea", "city"], "weight": 10.0},
        ]
        results = {"sea": self._stock("sea", 2), "city": self._stock("city", 3)}
        selections = material.select_sentence_clips(
            plan, results, VideoAspect.portrait, 5, 20
        )
        self.assertEqual([s["sentence_index"] for s in selections], sorted(s["sentence_index"] for s in selections))
        keys = [material._material_key(s["item"]) for s in selections]
        self.assertEqual(len(keys), len(set(keys)))
        per_sentence = {}
        for selection in selections:
            per_sentence[selection["sentence_index"]] = (
                per_sentence.get(selection["sentence_index"], 0) + selection["target_duration"]
            )
        self.assertAlmostEqual(per_sentence[0], 10, delta=0.1)
        self.assertAlmostEqual(per_sentence[1], 10, delta=0.1)

    def test_ranking_prefers_resolution_and_fitting_duration(self):
        good = self._stock("q", 1, duration=10, width=1080, height=1920)[0]
        short = self._stock("q", 1, duration=1, width=1080, height=1920)[0]
        short.url = "https://cdn/short.mp4"
        short.source_info = dict(short.source_info, asset_id="short")
        low_res = self._stock("q", 1, duration=10, width=270, height=480)[0]
        low_res.url = "https://cdn/low.mp4"
        low_res.source_info = dict(low_res.source_info, asset_id="low")
        score = lambda item: material._score_sentence_candidate(  # noqa: E731
            item, 0, 3, 5, VideoAspect.portrait
        )
        self.assertGreater(score(good), score(short))
        self.assertGreater(score(good), score(low_res))

    def test_unused_clips_from_other_sentences_fill_gaps(self):
        plan = [
            {"sentence": "a" * 10, "queries": ["none"], "weight": 10.0},
            {"sentence": "b" * 10, "queries": ["sea"], "weight": 10.0},
        ]
        results = {"none": [], "sea": self._stock("sea", 4)}
        selections = material.select_sentence_clips(
            plan, results, VideoAspect.portrait, 5, 20
        )
        self.assertEqual(len(selections), 4)
        self.assertEqual(selections[0]["sentence_index"], 0)

    def test_download_returns_paths_and_saves_alignment(self):
        plan = [
            {"sentence": "a" * 10, "queries": ["sea"], "weight": 10.0},
            {"sentence": "b" * 10, "queries": ["city"], "weight": 10.0},
        ]
        catalogue = {"sea": self._stock("sea", 3), "city": self._stock("city", 3)}

        def search(search_term, minimum_duration, video_aspect):
            return catalogue[search_term]

        def save(video_url, save_dir=""):
            return f"/tmp/{video_url.rsplit('/', 1)[-1]}"

        warnings = []
        with (
            patch.object(material, "save_video", side_effect=save),
            patch.object(material, "_persist_material_sources"),
            patch.object(material.task_artifacts, "patch_script_data") as patch_script,
        ):
            paths = material._download_videos_by_sentences(
                task_id="aligned",
                sentence_plan=plan,
                search_videos=search,
                video_aspect=VideoAspect.portrait,
                audio_duration=20,
                max_clip_duration=5,
                material_directory="",
                warnings_out=warnings,
            )
        self.assertEqual(len(paths), 4)
        self.assertTrue(all("sea" in p for p in paths[:2]))
        self.assertTrue(all("city" in p for p in paths[2:]))
        alignment = patch_script.call_args.kwargs["sentence_alignment"]
        self.assertEqual([a["sentence_index"] for a in alignment], [0, 0, 1, 1])
        self.assertEqual(warnings, [])
        self.assertTrue(all(entry.get("aligned") for entry in plan))

    def test_download_videos_falls_back_to_keywords_when_nothing_found(self):
        plan = [{"sentence": "a", "queries": ["none"], "weight": 1.0}]
        with (
            patch.object(material, "_search_videos_with_cache", return_value=[]),
            patch.object(material, "_persist_material_sources"),
        ):
            paths = material.download_videos(
                task_id="fallback",
                search_terms=["x"],
                source="pexels",
                audio_duration=10,
                sentence_plan=plan,
            )
        self.assertEqual(paths, [])
        self.assertFalse(plan[0].get("aligned"))


class TestStockCoverageWarning(_ConfigIsolation):
    def test_short_stock_footage_adds_warning(self):
        warnings = []
        items = [
            MaterialInfo(
                provider="pexels",
                url="https://cdn/a.mp4",
                duration=5,
                source_info={"provider": "pexels", "asset_id": "a"},
            )
        ]
        with (
            patch.object(material, "_search_videos_with_cache", return_value=items),
            patch.object(material, "save_video", return_value="/tmp/a.mp4"),
            patch.object(material, "_persist_material_sources"),
        ):
            paths = material.download_videos(
                task_id="short",
                search_terms=["x"],
                source="pexels",
                audio_duration=30,
                match_script_order=False,
                warnings=warnings,
            )
        self.assertEqual(paths, ["/tmp/a.mp4"])
        self.assertEqual(warnings[0]["code"], "insufficient_materials")
        self.assertEqual(warnings[0]["source"], "pexels")


class TestGetVideoMaterialsPlumbing(_ConfigIsolation):
    def test_warnings_script_and_sentence_plan_are_forwarded(self):
        params = VideoParams(video_subject="s", video_source="pexels")
        seen = {}

        def fake_download(**kwargs):
            seen.update(kwargs)
            kwargs["warnings"].append({"code": "insufficient_materials"})
            for entry in kwargs["sentence_plan"]:
                entry["aligned"] = True
            return ["a.mp4"]

        plan = [{"sentence": "Hi.", "queries": ["hi"], "weight": 2.0}]
        warnings = []
        with (
            patch.object(task_service.cache_manager, "sweep") as sweep,
            patch.object(task_service.material, "plan_sentence_queries", return_value=plan),
            patch.object(task_service.material, "download_videos", side_effect=fake_download),
        ):
            result = task_service.get_video_materials(
                "plumb", params, ["term"], 10, video_script="Hi.", warnings=warnings
            )

        sweep.assert_called_once_with()
        self.assertEqual(result, ["a.mp4"])
        self.assertEqual(warnings, [{"code": "insufficient_materials"}])
        self.assertIs(seen["sentence_plan"], plan)
        # Aligned footage is already in narrative order: pin sequential concat.
        self.assertTrue(params.match_materials_to_script)

    def test_batches_and_ai_sources_skip_sentence_alignment(self):
        for overrides in ({"video_count": 2}, {"video_source": "ofox"}):
            params = VideoParams(video_subject="s", **{"video_source": "pexels", **overrides})
            with (
                patch.object(task_service.cache_manager, "sweep"),
                patch.object(task_service.material, "plan_sentence_queries") as planner,
                patch.object(task_service.material, "download_videos", return_value=["a.mp4"]) as download,
            ):
                task_service.get_video_materials(
                    "plumb2", params, ["term"], 10, video_script="Hi."
                )
            planner.assert_not_called()
            self.assertIsNone(download.call_args.kwargs["sentence_plan"])

    def test_sweep_failure_does_not_block_materials(self):
        params = VideoParams(video_subject="s", video_source="pexels")
        with (
            patch.object(task_service.cache_manager, "sweep", side_effect=OSError("disk")),
            patch.object(task_service.material, "plan_sentence_queries", return_value=None),
            patch.object(task_service.material, "download_videos", return_value=["a.mp4"]),
        ):
            self.assertEqual(
                task_service.get_video_materials("plumb3", params, ["t"], 10),
                ["a.mp4"],
            )


if __name__ == "__main__":
    unittest.main()
