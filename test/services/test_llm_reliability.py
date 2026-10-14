import json
import unittest
from unittest.mock import MagicMock, patch

from app.config import config
from app.services import llm, task, twelvelabs


class TestLLMErrorHandling(unittest.TestCase):
    def test_script_raises_cleaned_secret_redacted_reason(self):
        raw = (
            "Error: Error code: 401 - Incorrect API key provided: sk-abcdef1234567890. "
            "Authorization: Bearer abcdefghijkl"
        )
        with patch.object(llm, "_generate_response", return_value=raw) as gen:
            with self.assertRaises(llm.LLMError) as caught:
                llm.generate_script(video_subject="Coffee")

        message = str(caught.exception)
        self.assertNotIn("sk-abcdef1234567890", message)
        self.assertNotIn("abcdefghijkl", message)
        self.assertNotIn("Error: ", message)
        self.assertIn("Incorrect API key", message)
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(gen.call_count, 1)

    def test_transient_errors_are_retried_with_exponential_backoff(self):
        delays = []
        with (
            patch.object(
                llm,
                "_generate_response",
                side_effect=["Error: Request timed out."] * 2 + ["Hello there."],
            ) as gen,
            patch.object(llm, "_sleep", side_effect=delays.append),
            patch.object(llm.random, "uniform", return_value=1.0),
        ):
            result = llm.generate_script(video_subject="Coffee")

        self.assertEqual(result, "Hello there.")
        self.assertEqual(gen.call_count, 3)
        self.assertEqual(delays, [1.0, 2.0])

    def test_transient_failure_raises_last_error_after_all_attempts(self):
        with (
            patch.object(
                llm, "_generate_response", return_value="Error: Request timed out."
            ) as gen,
            patch.object(llm, "_sleep"),
        ):
            with self.assertRaisesRegex(llm.LLMError, "timed out"):
                llm.generate_script(video_subject="Coffee")
        self.assertEqual(gen.call_count, llm._max_retries)

    def test_retry_budget_stops_the_outer_loop(self):
        with (
            patch.object(
                llm, "_generate_response", return_value="Error: Request timed out."
            ) as gen,
            patch.object(llm, "_sleep"),
            patch.dict(config.app, {"llm_total_budget": 0.001}),
        ):
            with self.assertRaises(llm.LLMError):
                llm.generate_script(video_subject="Coffee")
        self.assertEqual(gen.call_count, 1)

    def test_terms_stop_immediately_on_quota_error(self):
        with patch.object(
            llm,
            "_generate_response",
            return_value="Error: Error code: 429 - insufficient_quota",
        ) as gen:
            with self.assertRaisesRegex(llm.LLMError, "insufficient_quota"):
                llm.generate_terms("topic", "script")
        self.assertEqual(gen.call_count, 1)

    def test_terms_retry_rate_limit_then_succeed(self):
        with (
            patch.object(
                llm,
                "_generate_response",
                side_effect=["Error: Error code: 429 - rate limit", '["a b", "c d"]'],
            ) as gen,
            patch.object(llm, "_sleep"),
        ):
            result = llm.generate_terms("topic", "script")
        self.assertEqual(result, ["a b", "c d"])
        self.assertEqual(gen.call_count, 2)

    def test_non_retryable_classification(self):
        for text in (
            "Error code: 401 - invalid_api_key",
            "Error code: 404 - model not found",
            "openai: api_key is not set, please set it in the config.toml file.",
            "You exceeded your current quota",
        ):
            self.assertTrue(llm.is_non_retryable_llm_error(text), text)
        for text in (
            "Request timed out.",
            "Connection error.",
            "Error code: 429 - Rate limit reached",
            "Error code: 503 - overloaded",
        ):
            self.assertFalse(llm.is_non_retryable_llm_error(text), text)


class TestRequestTimeouts(unittest.TestCase):
    def setUp(self):
        self._snapshot = dict(config.app)
        config.app.update(
            {
                "llm_provider": "openai",
                "openai_api_key": "k",
                "openai_base_url": "https://example.test/v1",
                "openai_model_name": "m",
            }
        )

    def tearDown(self):
        config.app.clear()
        config.app.update(self._snapshot)

    def _run(self, **extra):
        config.app.update(extra)
        with patch.object(llm, "OpenAI") as client:
            client.return_value.chat.completions.create.return_value = MagicMock()
            llm._generate_response("hi")
        return client.call_args.kwargs

    def test_pipeline_defaults(self):
        kwargs = self._run()
        self.assertEqual(kwargs["timeout"], 120.0)
        self.assertEqual(kwargs["max_retries"], 1)

    def test_timeout_is_configurable(self):
        self.assertEqual(self._run(llm_request_timeout=45)["timeout"], 45.0)

    def test_invalid_timeout_falls_back_to_default(self):
        self.assertEqual(self._run(llm_request_timeout="abc")["timeout"], 120.0)
        self.assertEqual(self._run(llm_request_timeout=-5)["timeout"], 120.0)

    def test_dashscope_call_has_timeout(self):
        config.app.update(
            {"llm_provider": "qwen", "qwen_api_key": "k", "qwen_model_name": "m"}
        )
        with patch("dashscope.Generation.call", return_value=None) as call:
            llm._generate_response("hi")
        self.assertEqual(call.call_args.kwargs["request_timeout"], 120.0)

    def test_connection_test_budget_is_unchanged(self):
        with patch.object(llm, "OpenAI") as client:
            client.return_value.chat.completions.create.return_value = MagicMock()
            llm._generate_response("hi", request_timeout=30.0, max_retries=0)
        self.assertEqual(client.call_args.kwargs["timeout"], 30.0)
        self.assertEqual(client.call_args.kwargs["max_retries"], 0)


class TestScriptPrompt(unittest.TestCase):
    def test_default_prompt_quality_rules(self):
        prompt = llm.DEFAULT_SCRIPT_SYSTEM_PROMPT
        self.assertIn("## Constraints:", prompt)
        self.assertNotIn("Constrains", prompt)
        for needle in (
            "HOOK",
            "first 3 seconds",
            "ONE IDEA PER SENTENCE",
            "call to action",
            "emojis",
        ):
            self.assertIn(needle, prompt)

    def test_length_target_is_inferred_from_paragraphs(self):
        prompt = llm.build_script_prompt("Coffee", language="en", paragraph_number=3)
        self.assertIn("- number of paragraphs: 3", prompt)
        self.assertIn("about 60 seconds", prompt)
        # 60s * 2.5 words/s = 150 words, +/-10%
        self.assertIn("135 to 165 words", prompt)

    def test_explicit_duration_overrides_paragraph_inference(self):
        prompt = llm.build_script_prompt(
            "Coffee", paragraph_number=5, target_duration_seconds=30
        )
        self.assertIn("about 30 seconds", prompt)
        self.assertIn("68 to 82 words", prompt)

    def test_cjk_length_is_in_characters(self):
        prompt = llm.build_script_prompt(
            "咖啡的历史", language="zh-CN", target_duration_seconds=30
        )
        self.assertIn("108 to 132 characters", prompt)
        auto = llm.build_script_prompt("咖啡的历史", target_duration_seconds=30)
        self.assertIn("characters", auto)

    def test_duration_is_clamped(self):
        self.assertEqual(llm.resolve_script_target_duration(1, 1), 10)
        self.assertEqual(llm.resolve_script_target_duration(1, 99999), 600)

    def test_generate_script_forwards_duration(self):
        captured = {}

        def fake(prompt, **_):
            captured["prompt"] = prompt
            return "Did you know? Coffee is a fruit."

        with patch.object(llm, "_generate_response", side_effect=fake):
            script = llm.generate_script("Coffee", target_duration_seconds=40)
        self.assertEqual(script, "Did you know? Coffee is a fruit.")
        self.assertIn("about 40 seconds", captured["prompt"])

    def test_custom_system_prompt_still_gets_runtime_targets(self):
        prompt = llm.build_script_prompt("Coffee", custom_system_prompt="Be funny.")
        self.assertTrue(prompt.startswith("Be funny."))
        self.assertIn("target length:", prompt)


class TestTermsPrompt(unittest.TestCase):
    def _prompt(self, **kwargs):
        captured = {}

        def fake(prompt, **_):
            captured["prompt"] = prompt
            return '["chef slicing tomatoes", "aerial city skyline"]'

        with patch.object(llm, "_generate_response", side_effect=fake):
            terms = llm.generate_terms("Cooking", "Chop. Serve.", **kwargs)
        return terms, captured["prompt"]

    def test_prompt_demands_concrete_visual_terms(self):
        terms, prompt = self._prompt(amount=5)
        self.assertEqual(terms, ["chef slicing tomatoes", "aerial city skyline"])
        self.assertIn("## Constraints:", prompt)
        self.assertNotIn("Constrains", prompt)
        self.assertIn("2-4 english words", prompt)
        self.assertIn("avoid abstract nouns", prompt)
        self.assertIn("follow the narrative of the script", prompt)
        self.assertIn("vary the shot", prompt)

    def test_ordered_mode_keeps_ordering_rule(self):
        _, prompt = self._prompt(amount=3, match_script_order=True)
        self.assertIn("same order as the script narration", prompt)

    def test_fenced_json_and_prose_are_still_parsed(self):
        with patch.object(
            llm,
            "_generate_response",
            return_value='Sure:\n```json\n["a b", "c d"]\n```',
        ):
            self.assertEqual(llm.generate_terms("s", "x"), ["a b", "c d"])


class TestSentenceQueries(unittest.TestCase):
    SCRIPT = (
        "Coffee started in Ethiopia. Farmers picked red cherries! "
        "Roasters turn them dark."
    )

    def test_split_sentences(self):
        self.assertEqual(
            llm.split_script_sentences(self.SCRIPT),
            [
                "Coffee started in Ethiopia.",
                "Farmers picked red cherries!",
                "Roasters turn them dark.",
            ],
        )
        self.assertEqual(llm.split_script_sentences(""), [])

    def test_one_query_per_sentence_aligned(self):
        queries = [
            "ethiopian coffee farm",
            "farmer picking cherries",
            "coffee roasting drum",
        ]
        with patch.object(
            llm, "_generate_response", return_value=json.dumps(queries)
        ) as gen:
            result = llm.generate_sentence_queries(self.SCRIPT, video_subject="Coffee")
        self.assertEqual(result, queries)
        self.assertIn("exactly 3 strings", gen.call_args.args[0])
        self.assertIn("1. Coffee started in Ethiopia.", gen.call_args.args[0])

    def test_provider_error_falls_back_to_term_list(self):
        with patch.object(
            llm, "_generate_response", return_value="Error: invalid API key"
        ) as gen:
            result = llm.generate_sentence_queries(
                self.SCRIPT, fallback_terms=["coffee beans", "roasting coffee"]
            )
        self.assertEqual(result, ["coffee beans", "coffee beans", "roasting coffee"])
        self.assertEqual(gen.call_count, 1)  # permanent error: no retry

    def test_length_mismatch_uses_fallback_not_misaligned_items(self):
        with patch.object(llm, "_generate_response", return_value='["only one"]'):
            result = llm.generate_sentence_queries(self.SCRIPT, fallback_terms=["x y"])
        self.assertEqual(result, ["x y", "x y", "x y"])

    def test_blank_items_are_filled_from_fallback(self):
        with patch.object(
            llm, "_generate_response", return_value='["a b", " ", "c d"]'
        ):
            result = llm.generate_sentence_queries(self.SCRIPT, fallback_terms=["z z"])
        self.assertEqual(result, ["a b", "z z", "c d"])

    def test_without_fallback_uses_sentence_keywords(self):
        with patch.object(llm, "_generate_response", return_value="not json"):
            result = llm.generate_sentence_queries("Coffee started in Ethiopia.")
        self.assertEqual(result, ["coffee started in ethiopia"])

    def test_empty_script_returns_empty_list_without_llm_call(self):
        with patch.object(llm, "_generate_response") as gen:
            self.assertEqual(llm.generate_sentence_queries("  "), [])
        gen.assert_not_called()


class TestTaskReportsRealReason(unittest.TestCase):
    def _params(self):
        params = MagicMock()
        params.video_script = ""
        params.video_terms = ""
        params.match_materials_to_script = False
        params.video_subject = "Coffee"
        return params

    def test_script_failure_carries_provider_reason(self):
        with (
            patch.object(
                task.llm, "generate_script", side_effect=llm.LLMError("invalid API key")
            ),
            patch.object(task, "_mark_task_failed") as failed,
        ):
            result = task.generate_script("t1", self._params())
        self.assertIsNone(result)
        failed.assert_called_once_with("t1", "script", "invalid API key")

    def test_terms_failure_carries_provider_reason(self):
        with (
            patch.object(
                task.llm, "generate_terms", side_effect=llm.LLMError("quota exceeded")
            ),
            patch.object(task, "_mark_task_failed") as failed,
        ):
            result = task.generate_terms("t1", self._params(), "script")
        self.assertIsNone(result)
        failed.assert_called_once_with("t1", "terms", "quota exceeded")


class TestTwelveLabsTimeouts(unittest.TestCase):
    def test_embedding_uses_short_timeout_and_no_sdk_retries(self):
        import httpx

        seen = {}

        class FakeClient(httpx.Client):
            def __init__(self, *args, **kwargs):
                seen["timeout"] = kwargs.get("timeout")
                super().__init__(*args, **kwargs)

        embed = MagicMock()
        embed.create.side_effect = RuntimeError("down")
        client = MagicMock(embed=embed)
        twelvelabs._embed_text_cached.cache_clear()
        with (
            patch("httpx.Client", FakeClient),
            patch.object(twelvelabs, "_client", return_value=client),
        ):
            with self.assertRaises(RuntimeError):
                twelvelabs._embed_text_cached("topic", "model")
        twelvelabs._embed_text_cached.cache_clear()
        self.assertEqual(seen["timeout"], 15.0)
        self.assertEqual(
            embed.create.call_args.kwargs["request_options"], {"max_retries": 0}
        )


if __name__ == "__main__":
    unittest.main()
