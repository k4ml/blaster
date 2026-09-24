#!/usr/bin/env python3
"""Tests for wheeljack. Run with:  python tests_wheeljack.py

Stdlib-only (no pytest). Covers the foundation contracts: the modal queue
(regression for the concurrent-confirm deadlock), the cancel path, the steer
queue, the plugin loader, slash dispatch, renderer selection, and piped-output
cleanliness. blaster's own suite lives in tests.py and is untouched.
"""
import io
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wheeljack as wj  # noqa: E402

REPO = Path(__file__).resolve().parent


class QuietRenderer(wj.BaseRenderer):
    """BaseRenderer with no drawing, for exercising the modal queue directly."""

    def _show_modal(self, req):
        pass

    def _hide_modal(self):
        pass


# ---------------------------------------------------------------------------
# Modal queue — the deadlock regression
# ---------------------------------------------------------------------------
class TestModalQueue(unittest.TestCase):
    def test_concurrent_confirms_all_return(self):
        # The old single-slot design orphaned every caller but the last, which
        # hung forever. All N must now come back.
        n = 5
        r = QuietRenderer()
        start = threading.Barrier(n + 1)
        results = {}
        lock = threading.Lock()

        def call(i):
            start.wait()
            ans = r.confirm(f"allow {i}?", default="n", timeout=5)
            with lock:
                results[i] = ans

        def responder():
            start.wait()
            time.sleep(0.2)  # let all N occupy the queue
            answered = 0
            deadline = time.time() + 5
            while answered < n and time.time() < deadline:
                if r.current_modal() is not None and r.answer_modal("n"):
                    answered += 1
                time.sleep(0.005)

        threads = [threading.Thread(target=call, args=(i,), daemon=True)
                   for i in range(n)]
        threads.append(threading.Thread(target=responder, daemon=True))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=6)

        self.assertEqual(len(results), n, f"some confirms hung: {results}")
        self.assertTrue(all(v == "n" for v in results.values()), results)

    def test_modal_queue_is_fifo(self):
        r = QuietRenderer()
        r._modals.append(wj.ModalRequest("first", ("y", "n"), "n"))
        r._active = r._modals[0]
        r._modals.append(wj.ModalRequest("second", ("y", "n"), "n"))
        self.assertEqual(r.current_modal().prompt, "first")
        r.answer_modal("y")
        self.assertEqual(r.current_modal().prompt, "second")

    def test_timeout_returns_default_without_hanging(self):
        r = QuietRenderer()
        t0 = time.time()
        self.assertEqual(r.confirm("q?", default="n", timeout=0.1), "n")
        self.assertLess(time.time() - t0, 2.0)
        self.assertIsNone(r.current_modal())

    def test_cancel_all_denies_pending(self):
        r = QuietRenderer()
        result = {}

        def call():
            result["ans"] = r.confirm("q?", default="n")

        t = threading.Thread(target=call, daemon=True)
        t.start()
        time.sleep(0.05)
        r.cancel_all()
        t.join(timeout=2)
        self.assertFalse(t.is_alive(), "cancel_all did not release the caller")
        self.assertEqual(result["ans"], "n")

    def test_invalid_key_does_not_answer(self):
        r = QuietRenderer()
        r._modals.append(wj.ModalRequest("q?", ("y", "n"), "n"))
        r._active = r._modals[0]
        self.assertFalse(r.answer_modal("x"))
        self.assertEqual(r.current_modal().prompt, "q?")

    def test_confirm_yes_no(self):
        r = QuietRenderer()

        def answer():
            time.sleep(0.05)
            r.answer_modal("y")

        t = threading.Thread(target=answer, daemon=True)
        t.start()
        self.assertTrue(r.confirm_yes_no("q?"))
        t.join(timeout=1)


# ---------------------------------------------------------------------------
# Cancel path
# ---------------------------------------------------------------------------
class TestCancel(unittest.TestCase):
    def _app(self):
        app = wj.WheeljackApp()
        app._renderer = wj.StdioRenderer(app, stream=io.StringIO())
        return app

    def test_cancelled_agent_returns_promptly(self):
        app = self._app()
        app.cancel.set()
        agent = wj.DemoAgent(app)
        agent.start()
        agent.join(timeout=3)
        self.assertFalse(agent.is_alive(), "agent ignored cancel")
        self.assertFalse(app.turn_active)

    def test_agent_finishes_and_clears_turn_active(self):
        app = self._app()
        agent = wj.DemoAgent(app)
        agent.start()
        agent.join(timeout=5)
        self.assertFalse(agent.is_alive())
        self.assertFalse(app.turn_active)

    def test_request_cancel_denies_pending_modal(self):
        app = self._app()
        result = {}
        t = threading.Thread(
            target=lambda: result.setdefault(
                "ans", app.renderer.confirm("q?", default="n")), daemon=True)
        t.start()
        time.sleep(0.05)
        app.request_cancel()
        t.join(timeout=2)
        self.assertFalse(t.is_alive())
        self.assertEqual(result["ans"], "n")


# ---------------------------------------------------------------------------
# Steer queue
# ---------------------------------------------------------------------------
class TestSteerQueue(unittest.TestCase):
    def test_submit_queues_and_emits(self):
        app = wj.WheeljackApp()
        seen = []
        app.on(wj.SteerQueued, lambda e: seen.append(e.pending_count))
        app.submit_line("hello")
        self.assertEqual(app.drain_steers(), ["hello"])
        self.assertEqual(seen, [1])

    def test_drain_is_destructive_and_ordered(self):
        app = wj.WheeljackApp()
        app.submit_line("one")
        app.submit_line("two")
        self.assertEqual(app.drain_steers(), ["one", "two"])
        self.assertEqual(app.drain_steers(), [])

    def test_blank_line_ignored(self):
        app = wj.WheeljackApp()
        app.submit_line("   ")
        self.assertEqual(app.drain_steers(), [])
        self.assertFalse(app.turn_active)


# ---------------------------------------------------------------------------
# Slash dispatch
# ---------------------------------------------------------------------------
class TestSlashDispatch(unittest.TestCase):
    def test_help_handled_not_queued(self):
        app = wj.WheeljackApp()
        app.submit_line("/help")
        self.assertEqual(app.drain_steers(), [])
        self.assertTrue(any("Wheeljack commands" in m for m in app._log))
        self.assertFalse(app.turn_active)

    def test_unknown_command_handled_with_message(self):
        app = wj.WheeljackApp()
        app.submit_line("/bogus")
        self.assertEqual(app.drain_steers(), [])
        self.assertTrue(any("unknown command" in m for m in app._log))

    def test_model_get_and_set(self):
        app = wj.WheeljackApp()
        app.submit_line("/model llama3")
        self.assertEqual(app.config.model, "llama3")
        app._log.clear()
        app.submit_line("/model")
        self.assertTrue(any("llama3" in m for m in app._log))

    def test_quit_sets_flag(self):
        app = wj.WheeljackApp()
        app.submit_line("/quit")
        self.assertTrue(app.quit_requested)

    def test_plain_text_not_a_command(self):
        app = wj.WheeljackApp()
        self.assertFalse(wj.dispatch_slash(app, "hello world"))

    def test_plugins_command(self):
        app = wj.WheeljackApp()
        app.plugin_names.append("wheeljack_tui")
        app.submit_line("/plugins")
        self.assertTrue(any("wheeljack_tui" in m for m in app._log))


# ---------------------------------------------------------------------------
# Plugin loader
# ---------------------------------------------------------------------------
class TestPluginLoader(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name, source):
        (self.dir / name).write_text(source)

    def test_convention_skip_and_version_warn(self):
        self._write("wheeljack_good.py",
                    "def register(app):\n"
                    "    app.add_tool('good_tool', {'d': 1}, lambda *a: None)\n")
        self._write("wheeljack_bad.py", "raise RuntimeError('boom')\n")
        self._write("wheeljack_noreg.py", "X = 1\n")
        self._write("wheeljack_old.py",
                    "API_VERSION = 0\n"
                    "def register(app):\n"
                    "    app.add_slash_command('oldcmd', lambda a, s: None)\n")
        # A stray sibling that a broad glob would have executed.
        self._write("blaster.py",
                    "import pathlib\n"
                    "pathlib.Path(__file__).with_name('stray_ran').write_text('yes')\n")

        app = wj.WheeljackApp()
        wj.load_plugins(app, self.dir)

        self.assertEqual(app.plugin_names, ["wheeljack_good", "wheeljack_old"])
        self.assertIn("good_tool", app.tools)
        self.assertIn("oldcmd", app.slash_commands)
        self.assertTrue(any("failed to load" in m for m in app._log))
        self.assertTrue(any("no register" in m for m in app._log))
        self.assertTrue(any("API v0" in m for m in app._log))
        # The stray file must never have been imported.
        self.assertFalse((self.dir / "stray_ran").exists())
        self.assertNotIn("wheeljack_plugin_wheeljack_bad", sys.modules)

    def test_missing_dir_is_noop(self):
        app = wj.WheeljackApp()
        wj.load_plugins(app, self.dir / "nope")
        self.assertEqual(app.plugin_names, [])

    def test_core_module_is_importable_by_plugins(self):
        self.assertIs(wj._core_module(), sys.modules["wheeljack"])


# ---------------------------------------------------------------------------
# Renderer selection + console writer
# ---------------------------------------------------------------------------
class TestRendererSelection(unittest.TestCase):
    def test_no_factory_falls_back_to_stdio(self):
        app = wj.WheeljackApp()
        self.assertIsInstance(app.make_renderer(prefer_tui=True), wj.StdioRenderer)

    def test_factory_none_falls_back_to_stdio(self):
        app = wj.WheeljackApp()
        app.set_renderer(lambda a: None)
        self.assertIsInstance(app.make_renderer(prefer_tui=True), wj.StdioRenderer)

    def test_factory_raises_falls_back_to_stdio(self):
        app = wj.WheeljackApp()

        def boom(a):
            raise RuntimeError("no curses")

        app.set_renderer(boom)
        self.assertIsInstance(app.make_renderer(prefer_tui=True), wj.StdioRenderer)
        self.assertTrue(any("renderer plugin failed" in m for m in app._log))

    def test_no_tui_prefers_stdio(self):
        app = wj.WheeljackApp()
        app.set_renderer(lambda a: wj.StdioRenderer(a))
        self.assertIsInstance(app.make_renderer(prefer_tui=False), wj.StdioRenderer)

    def test_console_writer_piped_has_no_escapes(self):
        buf = io.StringIO()
        w = wj.ConsoleWriter(buf)
        w.emit_line("hello")
        w.emit("partial")
        w.set_input(">> ", "typed")
        self.assertNotIn("\x1b", buf.getvalue())


# ---------------------------------------------------------------------------
# The bundled TUI plugin degrades off-TTY
# ---------------------------------------------------------------------------
class TestTuiPlugin(unittest.TestCase):
    def test_plugin_loads_and_degrades_when_piped(self):
        plugin_dir = REPO / "wheeljack_plugins"
        if not plugin_dir.is_dir():
            self.skipTest("no plugin dir")
        app = wj.WheeljackApp()
        wj.load_plugins(app, plugin_dir)
        self.assertIn("wheeljack_tui", app.plugin_names)
        # Piped (not a TTY) => factory returns None => stdio.
        self.assertIsInstance(app.make_renderer(prefer_tui=True), wj.StdioRenderer)


# ---------------------------------------------------------------------------
# End-to-end: piped runs stay clean and never hang
# ---------------------------------------------------------------------------
class TestPipedIntegration(unittest.TestCase):
    def _run(self, args, stdin_bytes=b""):
        return subprocess.run(
            [sys.executable, "wheeljack.py", *args],
            cwd=REPO, input=stdin_bytes,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)

    def test_prompt_mode_exits_clean(self):
        p = self._run(["-p", "demo"])
        out = p.stdout.decode("utf-8", "replace")
        self.assertEqual(p.returncode, 0, out)
        self.assertNotIn("\x1b", out)
        self.assertNotIn(">> ", out)
        self.assertIn("Done.", out)

    def test_slash_and_quit_pipe_clean(self):
        p = self._run(["--no-tui"], stdin_bytes=b"/help\n/quit\n")
        out = p.stdout.decode("utf-8", "replace")
        self.assertEqual(p.returncode, 0, out)
        self.assertNotIn("\x1b", out)
        self.assertNotIn(">> ", out)
        self.assertIn("Wheeljack commands", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
