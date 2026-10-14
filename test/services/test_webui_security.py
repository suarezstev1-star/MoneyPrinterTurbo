import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.config import config
from app.models import const
from app.models.llm_provider import get_llm_provider
from app.models.schema import VideoParams
from app.services import llm, webui_task
from webui import access_gate

ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


class TestAccessGate(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(
            "os.environ",
            {},
            clear=False,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in (
            access_gate.PASSWORD_ENV,
            access_gate.HOST_ENV,
            access_gate.ALLOW_UNAUTHENTICATED_ENV,
        ):
            import os

            os.environ.pop(name, None)

    def test_loopback_detection(self):
        for host in ("127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2"):
            self.assertTrue(access_gate.is_loopback_host(host), host)
        for host in ("0.0.0.0", "", None, "192.168.1.5", "example.com", "::"):
            self.assertFalse(access_gate.is_loopback_host(host), host)

    def test_exposed_without_password_is_restricted(self):
        mode = access_gate.access_mode({}, lambda name: "0.0.0.0")
        self.assertEqual(mode, access_gate.MODE_RESTRICTED)
        self.assertFalse(access_gate.sensitive_features_enabled(mode))

    def test_unset_listen_address_is_treated_as_exposed(self):
        self.assertEqual(
            access_gate.access_mode({}, lambda name: None),
            access_gate.MODE_RESTRICTED,
        )

    def test_loopback_without_password_is_open(self):
        self.assertEqual(
            access_gate.access_mode({}, lambda name: "127.0.0.1"),
            access_gate.MODE_OPEN,
        )

    def test_env_host_used_when_streamlit_has_no_address(self):
        with patch.dict("os.environ", {access_gate.HOST_ENV: "127.0.0.1"}):
            self.assertEqual(
                access_gate.access_mode({}, lambda name: None),
                access_gate.MODE_OPEN,
            )

    def test_allow_unauthenticated_override(self):
        with patch.dict("os.environ", {access_gate.ALLOW_UNAUTHENTICATED_ENV: "1"}):
            self.assertEqual(
                access_gate.access_mode({}, lambda name: "0.0.0.0"),
                access_gate.MODE_OPEN,
            )

    def test_password_from_env_or_config_enables_gate(self):
        self.assertEqual(
            access_gate.access_mode({"webui_password": "pw"}, lambda name: "0.0.0.0"),
            access_gate.MODE_GATED,
        )
        with patch.dict("os.environ", {access_gate.PASSWORD_ENV: "secret"}):
            self.assertEqual(
                access_gate.access_mode({}, lambda name: "127.0.0.1"),
                access_gate.MODE_GATED,
            )
            self.assertEqual(access_gate.configured_password({"webui_password": "x"}), "secret")

    def test_verify_password(self):
        self.assertTrue(access_gate.verify_password("pw", "pw"))
        self.assertFalse(access_gate.verify_password("wrong", "pw"))
        self.assertFalse(access_gate.verify_password("", ""))
        self.assertFalse(access_gate.verify_password(None, "pw"))
        self.assertTrue(access_gate.verify_password("pässwörd", "pässwörd"))


class TestWebuiGateUi(unittest.TestCase):
    def _run(self, env):
        with patch.dict("os.environ", env, clear=False):
            app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
            app.run()
            return app

    def test_password_prompt_hides_the_ui_until_unlocked(self):
        env = {access_gate.PASSWORD_ENV: "correct horse"}
        with patch.dict("os.environ", env, clear=False):
            app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.text_input), 1)
            self.assertEqual(len(app.get("download_button")), 0)
            labels = [b.label for b in app.button]
            self.assertFalse(any("Settings" in label for label in labels))

            app.text_input[0].set_value("wrong")
            app.button[0].click()
            app.run()
            self.assertTrue(app.error)
            self.assertNotIn(access_gate.AUTH_SESSION_KEY, app.session_state)

            app.text_input[0].set_value("correct horse")
            app.button[0].click()
            app.run()
            self.assertTrue(app.session_state[access_gate.AUTH_SESSION_KEY])
            self.assertTrue(
                any(b.key == "open_settings_dialog_button" for b in app.button)
            )

    def test_exposed_without_password_disables_settings(self):
        with (
            patch.dict("os.environ", {access_gate.ALLOW_UNAUTHENTICATED_ENV: ""}),
            patch.object(
                access_gate, "server_host", lambda streamlit_option=None: "0.0.0.0"
            ),
        ):
            app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(app.warning)
            settings = [
                b for b in app.button if b.key == "open_settings_dialog_button"
            ]
            self.assertEqual(len(settings), 1)
            self.assertTrue(settings[0].disabled)


class TestSecretNames(unittest.TestCase):
    def test_token_names_are_treated_as_credentials(self):
        source = (ROOT_DIR / "webui" / "Main.py").read_text(encoding="utf-8")
        self.assertIn('"access_token"', source)
        self.assertIn('"refresh_token"', source)
        for name in ("kimi_code_access_token", "kimi_code_refresh_token"):
            self.assertTrue(
                name.endswith(
                    ("api_token", "access_token", "refresh_token")
                )
            )

    def test_hide_config_option_is_removed(self):
        self.assertNotIn(
            "hide_config", (ROOT_DIR / "webui" / "Main.py").read_text(encoding="utf-8")
        )
        self.assertNotIn(
            "hide_config", (ROOT_DIR / "config.example.toml").read_text(encoding="utf-8")
        )


class TestClaudeCliPathValidation(unittest.TestCase):
    def test_cli_path_field_is_read_only(self):
        provider = get_llm_provider("claude_code")
        field = next(f for f in provider.extra_fields if f.config_suffix == "cli_path")
        self.assertTrue(field.read_only)

    def test_rejects_other_executables(self):
        for value in ("/bin/sh", "bash", "C:\\Windows\\System32\\cmd.exe", "/tmp/claude-evil", "python"):
            with patch.object(llm.shutil, "which", return_value="/bin/x"), self.assertRaises(
                ValueError, msg=value
            ):
                llm.resolve_claude_cli_path(value)

    def test_rejects_resolved_path_with_other_basename(self):
        with patch.object(llm.shutil, "which", return_value="/usr/bin/sh"):
            with self.assertRaises(ValueError):
                llm.resolve_claude_cli_path("claude")

    def test_accepts_claude_names(self):
        with patch.object(llm.shutil, "which", return_value="/opt/bin/claude"):
            self.assertEqual(llm.resolve_claude_cli_path(""), "/opt/bin/claude")
            self.assertEqual(
                llm.resolve_claude_cli_path("/opt/bin/claude"), "/opt/bin/claude"
            )
        with patch.object(llm.shutil, "which", return_value=None), patch.object(
            llm.os.path, "isfile", return_value=True
        ):
            self.assertEqual(
                llm.resolve_claude_cli_path("C:\\tools\\Claude.CMD"),
                "C:\\tools\\Claude.CMD",
            )

    def test_missing_cli_reports_not_found(self):
        with patch.object(llm.shutil, "which", return_value=None), patch.object(
            llm.os.path, "isfile", return_value=False
        ):
            with self.assertRaisesRegex(ValueError, "not found"):
                llm.resolve_claude_cli_path("claude")

    def test_generate_response_never_executes_rejected_path(self):
        original = dict(config.app)
        try:
            config.app["llm_provider"] = "claude_code"
            config.app["claude_code_cli_path"] = "/bin/sh"
            with patch.object(llm.subprocess, "run") as run:
                result = llm._generate_response("hi")
            run.assert_not_called()
            self.assertIn("Error", result)
        finally:
            config.app.clear()
            config.app.update(original)


class TestSessionTaskCap(unittest.TestCase):
    def setUp(self):
        webui_task._session_tasks.clear()
        original = dict(config.app)
        self.addCleanup(lambda: (config.app.clear(), config.app.update(original)))
        config.app["webui_max_tasks_per_session"] = 2

    def test_cap_blocks_extra_tasks_and_releases_slots(self):
        gate = threading.Event()

        def fake_add_task(func, **kwargs):
            gate.wait(5)

        with (
            patch.object(webui_task._task_manager, "add_task", side_effect=fake_add_task),
            patch.object(webui_task.sm.state, "update_task"),
        ):
            params = VideoParams(video_subject="x")
            gate.set()
            webui_task.submit_generation("t1", params, session_id="s1")
            webui_task.submit_generation("t2", params, session_id="s1")
            with self.assertRaises(webui_task.SessionTaskLimitError) as ctx:
                webui_task.submit_generation("t3", params, session_id="s1")
            self.assertEqual(ctx.exception.limit, 2)
            # Another session is not affected.
            webui_task.submit_generation("t4", params, session_id="s2")

            webui_task._release_session_slot("s1", "t1")
            webui_task.submit_generation("t5", params, session_id="s1")

    def test_failed_scheduling_releases_slot(self):
        with (
            patch.object(
                webui_task._task_manager, "add_task", side_effect=RuntimeError("boom")
            ),
            patch.object(webui_task.sm.state, "update_task"),
        ):
            params = VideoParams(video_subject="x")
            for index in range(5):
                with self.assertRaises(RuntimeError):
                    webui_task.submit_generation(
                        f"t{index}", params, session_id="s1"
                    )
        self.assertNotIn("s1", webui_task._session_tasks)

    def test_refused_task_writes_no_state(self):
        with (
            patch.object(webui_task._task_manager, "add_task"),
            patch.object(webui_task.sm.state, "update_task") as update_task,
        ):
            params = VideoParams(video_subject="x")
            webui_task.submit_generation("a", params, session_id="s1")
            webui_task.submit_generation("b", params, session_id="s1")
            update_task.reset_mock()
            with self.assertRaises(webui_task.SessionTaskLimitError):
                webui_task.submit_generation("c", params, session_id="s1")
            update_task.assert_not_called()

    def test_worker_releases_slot_on_completion(self):
        webui_task._reserve_session_slot("s1", "t1")
        with patch.object(
            webui_task.tm, "start", return_value={"state": const.TASK_STATE_COMPLETE}
        ):
            webui_task._run_generation(
                "t1", VideoParams(video_subject="x"), False, session_id="s1"
            )
        self.assertNotIn("s1", webui_task._session_tasks)


if __name__ == "__main__":
    unittest.main()
