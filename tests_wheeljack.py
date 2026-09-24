#!/usr/bin/env python3
"""Tests for wheeljack. Run with:  python tests_wheeljack.py

Stdlib-only (no pytest). Covers the foundation contracts: the modal queue
(regression for the concurrent-confirm deadlock), the cancel path, the steer
queue, the plugin loader, slash dispatch, renderer selection, and piped-output
cleanliness. blaster's own suite lives in tests.py and is untouched.
"""
import hashlib
import http.server
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime
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


class DenyRenderer(QuietRenderer):
    """Answers every confirm() with 'n' and records the prompts it saw.

    Gates are tested through the renderer (the plan's design), so tests need a
    deterministic responder instead of a real TTY. Without one, confirm() blocks
    for CONFIRM_TIMEOUT and the gate tests hang.
    """

    def __init__(self, app=None):
        super().__init__(app)
        self.prompts = []

    def confirm(self, prompt, **kw):
        self.prompts.append(prompt)
        return kw.get("default", "n")


class AllowRenderer(DenyRenderer):
    """Answers every confirm() with 'y'."""

    def confirm(self, prompt, **kw):
        self.prompts.append(prompt)
        return "y"


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
        self._write("good.py",
                    "def register(app):\n"
                    "    app.add_tool('good_tool', {'d': 1}, lambda *a: None)\n")
        self._write("bad.py", "raise RuntimeError('boom')\n")
        self._write("noreg.py", "X = 1\n")
        self._write("old.py",
                    "API_VERSION = 0\n"
                    "def register(app):\n"
                    "    app.add_slash_command('oldcmd', lambda a, s: None)\n")
        # Files starting with _ or wheeljack.py itself are skipped.
        self._write("_stray.py",
                    "import pathlib\n"
                    "pathlib.Path(__file__).with_name('stray_ran').write_text('yes')\n")
        self._write("wheeljack.py",
                    "import pathlib\n"
                    "pathlib.Path(__file__).with_name('wj_ran').write_text('yes')\n")

        app = wj.WheeljackApp()
        wj.load_plugins(app, self.dir)

        self.assertEqual(app.plugin_names, ["good", "old"])
        self.assertIn("good_tool", app.tools)
        self.assertIn("oldcmd", app.slash_commands)
        self.assertTrue(any("failed to load" in m for m in app._log))
        self.assertTrue(any("no register" in m for m in app._log))
        self.assertTrue(any("API v0" in m for m in app._log))
        self.assertFalse((self.dir / "stray_ran").exists())
        self.assertFalse((self.dir / "wj_ran").exists())
        self.assertNotIn("wheeljack_plugin_bad", sys.modules)

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

    def test_loader_skips_underscore_and_wheeljack_itself(self):
        """--plugins . must never import wheeljack.py or the _*-prefixed files."""
        app = wj.WheeljackApp()
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for name, marker in (("_side.py", "side_ran"),
                                 ("wheeljack.py", "wj_ran"),
                                 ("__init__.py", "init_ran")):
                (d / name).write_text(
                    "import pathlib\n"
                    f"pathlib.Path(__file__).with_name('{marker}').write_text('yes')\n")
            wj.load_plugins(app, d)
            self.assertEqual(app.plugin_names, [])
            for marker in ("side_ran", "wj_ran", "init_ran"):
                self.assertFalse((d / marker).exists(), marker)


# ---------------------------------------------------------------------------
# Tool-call parsing robustness
# ---------------------------------------------------------------------------
class TestToolCallParsing(unittest.TestCase):
    def _agent(self, td):
        app = wj.WheeljackApp()
        app._renderer = QuietRenderer()
        cfg = wj.Config(cwd=Path(td))
        return wj.Agent(app, cfg, Path(td))

    def test_tool_calls_parsed_without_finish_reason(self):
        """A server that omits finish_reason='tool_calls' must not have its
        tool call silently dropped and treated as a final answer."""
        with tempfile.TemporaryDirectory() as td:
            agent = self._agent(td)
            resp = wj.LLMResponse(
                choices=[{"message": {"tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "read_file",
                                  "arguments": '{"path":"a.txt"}'}}]},
                    "finish_reason": "stop"}],
                usage={}, streamed=True)
            calls = agent._parse_tool_calls(resp)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].id, "c1")

    def test_empty_and_malformed_tool_calls_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            agent = self._agent(td)
            cases = [
                {"message": {"content": "just text"}, "finish_reason": "stop"},
                {"message": {"tool_calls": []}, "finish_reason": "tool_calls"},
                {"message": {"tool_calls": [{"id": "x"}]}, "finish_reason": "tool_calls"},
                {"message": {}, "finish_reason": "tool_calls"},
            ]
            for choice in cases:
                resp = wj.LLMResponse(choices=[choice], usage={}, streamed=True)
                self.assertEqual(agent._parse_tool_calls(resp), [], choice)

    def test_llm_response_shorthands(self):
        resp = wj.LLMResponse(
            choices=[{"message": {"content": "hi",
                                  "tool_calls": [{"id": "1", "function": {}}]},
                      "finish_reason": "stop"}],
            usage={"prompt_tokens": 1, "completion_tokens": 2})
        self.assertEqual(resp.content, "hi")
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertFalse(resp.streamed)
        empty = wj.LLMResponse(choices=[], usage={})
        self.assertEqual(empty.content, "")
        self.assertEqual(empty.tool_calls, [])
        self.assertEqual(empty.message, {})


# ---------------------------------------------------------------------------
# The bundled TUI plugin degrades off-TTY
# ---------------------------------------------------------------------------
class TestTuiPlugin(unittest.TestCase):
    def test_plugin_loads_and_degrades_when_piped(self):
        plugin_dir = REPO / ".wheeljack" / "plugins"
        if not plugin_dir.is_dir():
            self.skipTest("no plugin dir")
        app = wj.WheeljackApp()
        wj.load_plugins(app, plugin_dir)
        self.assertIn("tui", app.plugin_names)
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
        p = self._run(["-p", "demo", "--demo"])
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


# ---------------------------------------------------------------------------
# Mock OpenAI chat completion server
# ---------------------------------------------------------------------------
def _sse(delta, finish=None):
    chunk = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(chunk)}\n\n".encode()


def _usage_event():
    return b'data: {"usage":{"prompt_tokens":3,"completion_tokens":5}}\n\n'


def _make_chat_handler(srv):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(n) if n else b""
            try:
                srv.requests.append(json.loads(raw))
            except Exception:
                srv.requests.append({})

            if srv.single_json is not None:
                body = json.dumps(srv.single_json).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for event in srv.events():
                    self.wfile.write(event)
                    self.wfile.flush()
                    if srv.delay:
                        time.sleep(srv.delay)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # Expected when a test cancels mid-stream (the client hangs up).
                pass

        def log_message(self, *a):
            pass

    return Handler


class MockChatServer:
    def __init__(self, scenario="plain", delay=0.0, single_json=None, scenarios=None):
        self.scenario = scenario
        self.scenarios = list(scenarios) if scenarios else None
        self.delay = delay
        self.single_json = single_json
        self.requests = []
        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), _make_chat_handler(self))
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def api_base(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def events(self):
        s = self.scenarios.pop(0) if self.scenarios else self.scenario
        out = []
        if s == "slow_content":
            for w in ["Hello", " from", " slow", " stream", "."]:
                out.append(_sse({"content": w}))
        elif s == "reasoning_content":
            for r in ["Thinking step 1\n", "Thinking step 2\n"]:
                out.append(_sse({"reasoning": r}))
            for c in ["Hello ", "world."]:
                out.append(_sse({"content": c}))
        elif s == "reasoning_only":
            for r in ["only reasoning here\n", "no content marker\n"]:
                out.append(_sse({"reasoning": r}))
        elif s == "toolcall":
            out.append(_sse({"tool_calls": [{"index": 0, "id": "call_1", "function": {
                "name": "read_file", "arguments": '{"path":"hello.txt"}'}}]}))
        elif s == "plain":
            for c in ["Plain ", "answer."]:
                out.append(_sse({"content": c}))
        out.append(_sse({}, "tool_calls" if s == "toolcall" else "stop"))
        out.append(_usage_event())
        return out

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


# ---------------------------------------------------------------------------
# LLM client & event emission
# ---------------------------------------------------------------------------
class TestLLMClient(unittest.TestCase):
    def test_streaming_emits_stream_delta(self):
        events = []
        with MockChatServer(scenario="plain") as srv:
            cfg = wj.Config(api_base=srv.api_base, request_timeout=5)
            client = wj.LLMClient(cfg, emit=events.append)
            resp = client.call([wj.Message("user", "hi")])
            self.assertEqual(resp.content, "Plain answer.")
            self.assertTrue(resp.streamed)
            stream_deltas = [e.text for e in events if isinstance(e, wj.StreamDelta)]
            self.assertEqual("".join(stream_deltas), "Plain answer.")

    def test_reasoning_and_content_events(self):
        events = []
        with MockChatServer(scenario="reasoning_content") as srv:
            cfg = wj.Config(api_base=srv.api_base, request_timeout=5)
            client = wj.LLMClient(cfg, emit=events.append)
            resp = client.call([wj.Message("user", "hi")])
            self.assertEqual(resp.content, "Hello world.")
            reasoning = [e.text for e in events if isinstance(e, wj.ReasoningDelta)]
            self.assertEqual("".join(reasoning), "Thinking step 1\nThinking step 2\n")

    def test_reasoning_only_edge_case(self):
        with MockChatServer(scenario="reasoning_only") as srv:
            cfg = wj.Config(api_base=srv.api_base, request_timeout=5)
            client = wj.LLMClient(cfg, emit=lambda e: None)
            resp = client.call([wj.Message("user", "hi")])
            self.assertIn("only reasoning here", resp.content)

    def test_single_json_fallback(self):
        body = {
            "choices": [{
                "message": {"role": "assistant", "content": "single json response"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}
        }
        with MockChatServer(single_json=body) as srv:
            cfg = wj.Config(api_base=srv.api_base, request_timeout=5)
            client = wj.LLMClient(cfg, emit=lambda e: None)
            resp = client.call([wj.Message("user", "hi")])
            self.assertEqual(resp.content, "single json response")
            self.assertFalse(resp.streamed)

    def test_cancel_between_chunks(self):
        cancel_ev = threading.Event()
        with MockChatServer(scenario="slow_content", delay=0.1) as srv:
            cfg = wj.Config(api_base=srv.api_base, request_timeout=5)
            client = wj.LLMClient(cfg, emit=lambda e: None, cancel=cancel_ev)

            def trigger_cancel():
                time.sleep(0.05)
                cancel_ev.set()

            t = threading.Thread(target=trigger_cancel, daemon=True)
            t.start()
            t0 = time.time()
            client.call([wj.Message("user", "hi")])
            self.assertLess(time.time() - t0, 1.5)
            self.assertTrue(cancel_ev.is_set())


# ---------------------------------------------------------------------------
# Core tools & safety gates
# ---------------------------------------------------------------------------
class TestCoreToolsAndGates(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "hello.txt").write_text("hello world\nline 2")
        self.app = wj.WheeljackApp()
        # A renderer that auto-denies every prompt: the modal queue is exercised
        # directly in TestModalQueue, so gates are tested without a real TTY and
        # without the 300s CONFIRM_TIMEOUT fallback.
        self.app._renderer = DenyRenderer()

    def tearDown(self):
        self._tmp.cleanup()

    def test_read_file_and_cap(self):
        cfg = wj.Config(cwd=self.root, max_file_size=5)
        wj.register_core_tools(self.app, cfg, self.root)
        read_fn = self.app.tools["read_file"][1]
        res = read_fn({"path": "hello.txt"})
        self.assertIn("File too large", res)
        self.assertIn("max 5", res)

    def test_read_file_returns_content(self):
        cfg = wj.Config(cwd=self.root)
        wj.register_core_tools(self.app, cfg, self.root)
        read_fn = self.app.tools["read_file"][1]
        res = read_fn({"path": "hello.txt"})
        self.assertIn("hello world", res)
        self.assertIn("--- end", res)

    def test_edit_blocked_without_write_flag(self):
        cfg = wj.Config(cwd=self.root, allow_edits=False)
        wj.register_core_tools(self.app, cfg, self.root)
        edit_fn = self.app.tools["edit_file"][1]
        res = edit_fn({"path": "hello.txt", "before": "hello", "after": "bye"})
        self.assertIn("[edit blocked: edit mode is off", res)
        self.assertEqual((self.root / "hello.txt").read_text(), "hello world\nline 2")
        self.assertFalse((self.root / "hello.txt.bak").exists())

    def test_edit_allowed_with_write_flag_and_creates_bak(self):
        cfg = wj.Config(cwd=self.root, allow_edits=True)
        wj.register_core_tools(self.app, cfg, self.root)
        edit_fn = self.app.tools["edit_file"][1]
        res = edit_fn({"path": "hello.txt", "before": "hello", "after": "bye"})
        self.assertIn("Successfully edited", res)
        self.assertEqual((self.root / "hello.txt").read_text(), "bye world\nline 2")
        self.assertTrue((self.root / "hello.txt.bak").exists())

    def test_edit_rejects_non_unique_before(self):
        (self.root / "dup.txt").write_text("aa aa")
        cfg = wj.Config(cwd=self.root, allow_edits=True)
        wj.register_core_tools(self.app, cfg, self.root)
        edit_fn = self.app.tools["edit_file"][1]
        res = edit_fn({"path": "dup.txt", "before": "aa", "after": "bb"})
        self.assertIn("not unique", res)
        self.assertEqual((self.root / "dup.txt").read_text(), "aa aa")

    def test_write_allowed_with_write_flag(self):
        cfg = wj.Config(cwd=self.root, allow_edits=True)
        wj.register_core_tools(self.app, cfg, self.root)
        write_fn = self.app.tools["write_file"][1]
        res = write_fn({"path": "new.txt", "content": "fresh data"})
        self.assertIn("Successfully wrote", res)
        self.assertEqual((self.root / "new.txt").read_text(), "fresh data")

    def test_write_blocked_without_write_flag(self):
        cfg = wj.Config(cwd=self.root, allow_edits=False)
        wj.register_core_tools(self.app, cfg, self.root)
        write_fn = self.app.tools["write_file"][1]
        res = write_fn({"path": "nope.txt", "content": "x"})
        self.assertIn("[edit blocked: edit mode is off", res)
        self.assertFalse((self.root / "nope.txt").exists())

    def test_run_shell_captures_output_and_code(self):
        cfg = wj.Config(cwd=self.root)
        wj.register_core_tools(self.app, cfg, self.root)
        shell_fn = self.app.tools["run_shell"][1]
        res = shell_fn({"command": "echo 'wheeljack test'"})
        self.assertIn("wheeljack test", res)
        self.assertIn("[exit code: 0]", res)

    def test_list_files_flat_and_recursive(self):
        (self.root / "sub").mkdir()
        (self.root / "sub" / "deep.txt").write_text("x")
        cfg = wj.Config(cwd=self.root)
        wj.register_core_tools(self.app, cfg, self.root)
        list_fn = self.app.tools["list_files"][1]
        flat = list_fn({"path": "."})
        self.assertIn("hello.txt", flat)
        self.assertNotIn("deep.txt", flat)
        tree = list_fn({"path": ".", "recursive": True})
        self.assertIn("deep.txt", tree)

    def test_destructive_command_auto_denies(self):
        cfg = wj.Config(cwd=self.root)
        wj.register_core_tools(self.app, cfg, self.root)
        shell_fn = self.app.tools["run_shell"][1]
        res = shell_fn({"command": "rm -rf /tmp/test-nonexistent"})
        self.assertIn("[command blocked", res)

    def test_destructive_command_decline_is_remembered(self):
        cfg = wj.Config(cwd=self.root)
        wj.register_core_tools(self.app, cfg, self.root)
        shell_fn = self.app.tools["run_shell"][1]
        shell_fn({"command": "rm -rf /tmp/test-nonexistent"})
        # A second attempt must not even prompt: it is declined immediately.
        self.app._renderer.prompts.clear()
        res = shell_fn({"command": "rm -rf /tmp/test-nonexistent"})
        self.assertIn("[command blocked", res)
        self.assertEqual(self.app._renderer.prompts, [])

    def test_unflagged_command_runs_without_prompting(self):
        cfg = wj.Config(cwd=self.root)
        wj.register_core_tools(self.app, cfg, self.root)
        shell_fn = self.app.tools["run_shell"][1]
        shell_fn({"command": "echo safe"})
        self.assertEqual(self.app._renderer.prompts, [])


# ---------------------------------------------------------------------------
# Tool Executor (read-only parallel, mutating serial, call-order preserved)
# ---------------------------------------------------------------------------
class TestToolExecutor(unittest.TestCase):
    def test_parallel_reads_overlap_and_mutating_serial(self):
        app = wj.WheeljackApp()
        app._renderer = QuietRenderer()
        cfg = wj.Config(tool_parallel_reads=True, parallel_workers=4)

        log = []
        lock = threading.Lock()

        def slow_read(args):
            with lock:
                log.append(f"start_read_{args['id']}")
            time.sleep(0.15)
            with lock:
                log.append(f"end_read_{args['id']}")
            return f"read_{args['id']}"

        def slow_write(args):
            with lock:
                log.append(f"start_write_{args['id']}")
            time.sleep(0.15)
            with lock:
                log.append(f"end_write_{args['id']}")
            return f"write_{args['id']}"

        app.add_tool("ro", {"parallel_safe": True}, slow_read)
        app.add_tool("mut", {"parallel_safe": False}, slow_write)

        with tempfile.TemporaryDirectory() as td:
            agent = wj.Agent(app, cfg, Path(td))
            calls = [
                wj.ToolCall("1", {"name": "ro", "arguments": '{"id": 1}'}),
                wj.ToolCall("2", {"name": "mut", "arguments": '{"id": 2}'}),
                wj.ToolCall("3", {"name": "ro", "arguments": '{"id": 3}'}),
                wj.ToolCall("4", {"name": "mut", "arguments": '{"id": 4}'}),
            ]
            t0 = time.time()
            results = agent.execute_tools(calls)
            elapsed = time.time() - t0

            self.assertEqual(results, ["read_1", "write_2", "read_3", "write_4"])
            self.assertLess(elapsed, 0.75)

    def test_no_parallel_flag_serializes_reads(self):
        app = wj.WheeljackApp()
        app._renderer = QuietRenderer()
        cfg = wj.Config(tool_parallel_reads=False)
        order = []

        def slow_read(args):
            order.append(f"start{args['id']}")
            time.sleep(0.1)
            order.append(f"end{args['id']}")
            return f"r{args['id']}"

        app.add_tool("ro", {"parallel_safe": True}, slow_read)
        with tempfile.TemporaryDirectory() as td:
            agent = wj.Agent(app, cfg, Path(td))
            calls = [wj.ToolCall(str(i), {"name": "ro", "arguments": json.dumps({"id": i})})
                     for i in (1, 2)]
            results = agent.execute_tools(calls)
            self.assertEqual(results, ["r1", "r2"])
            # Serial: each read completes before the next starts.
            self.assertEqual(order, ["start1", "end1", "start2", "end2"])

    def test_parallel_safe_plugin_tool_can_still_confirm(self):
        """A parallel_safe tool that prompts must work: the modal queue covers
        concurrent callers (the plan's watch item)."""
        app = wj.WheeljackApp()
        app._renderer = AllowRenderer()
        cfg = wj.Config(tool_parallel_reads=True, parallel_workers=4)

        def asking(args):
            ok = app.confirm(f"allow {args['id']}?", default="n",
                             timeout=5) == "y"
            return f"{args['id']}:{ok}"

        app.add_tool("ask", {"parallel_safe": True}, asking)
        with tempfile.TemporaryDirectory() as td:
            agent = wj.Agent(app, cfg, Path(td))
            calls = [wj.ToolCall(str(i), {"name": "ask", "arguments": json.dumps({"id": i})})
                     for i in range(4)]
            t0 = time.time()
            results = agent.execute_tools(calls)
            self.assertLess(time.time() - t0, 3.0)
            self.assertEqual(results, ["0:True", "1:True", "2:True", "3:True"])
            self.assertEqual(len(app._renderer.prompts), 4)

    def test_unknown_tool_returns_error_in_order(self):
        app = wj.WheeljackApp()
        app._renderer = QuietRenderer()
        cfg = wj.Config()
        with tempfile.TemporaryDirectory() as td:
            agent = wj.Agent(app, cfg, Path(td))
            calls = [
                wj.ToolCall("a", {"name": "nope", "arguments": "{}"}),
                wj.ToolCall("b", {"name": "nope2", "arguments": "{}"}),
            ]
            results = agent.execute_tools(calls)
            self.assertEqual(len(results), 2)
            self.assertTrue(results[0].startswith("Error: Unknown tool"))
            self.assertTrue(results[1].startswith("Error: Unknown tool"))


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
class TestAgentLoop(unittest.TestCase):
    def test_agent_turn_tools_and_final_answer(self):
        scenarios = ["toolcall", "plain"]
        with MockChatServer(scenarios=scenarios) as srv:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                (root / "hello.txt").write_text("agent loop content")
                app = wj.WheeljackApp()
                app._renderer = QuietRenderer()
                cfg = wj.Config(api_base=srv.api_base, cwd=root, request_timeout=5)
                wj.register_core_tools(app, cfg, root)
                agent = wj.Agent(app, cfg, root)
                ans = agent.run_once("read hello.txt")
                self.assertEqual(ans, "Plain answer.")
                tool_msgs = [m for m in agent.messages if m.role == "tool"]
                self.assertEqual(len(tool_msgs), 1)
                # read_file wraps the body in --- path --- markers.
                self.assertIn("agent loop content", tool_msgs[0].content)
                self.assertEqual(tool_msgs[0].tool_call_id, "call_1")
                # The assistant tool_call block that owns the result is present.
                owning = [m for m in agent.messages
                          if m.role == "assistant" and m.tool_calls]
                self.assertEqual(len(owning), 1)
                self.assertEqual(owning[0].tool_calls[0].id, "call_1")

    def test_agent_round_limit_enforced(self):
        with MockChatServer(scenario="toolcall") as srv:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                (root / "hello.txt").write_text("infinite loop")
                app = wj.WheeljackApp()
                app._renderer = QuietRenderer()
                cfg = wj.Config(api_base=srv.api_base, cwd=root, max_iterations=2, request_timeout=5)
                wj.register_core_tools(app, cfg, root)
                agent = wj.Agent(app, cfg, root)
                agent.run_once("loop")
                self.assertLessEqual(len([m for m in agent.messages if m.role == "tool"]), 3)

    def test_steer_mid_turn_reaches_next_llm_call(self):
        # The server always asks for a tool, so the turn keeps going; a steer
        # queued after the first round must appear as a user message.
        with MockChatServer(scenario="toolcall") as srv:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                (root / "hello.txt").write_text("steer me")
                app = wj.WheeljackApp()
                app._renderer = QuietRenderer()
                cfg = wj.Config(api_base=srv.api_base, cwd=root,
                                max_iterations=3, request_timeout=5)
                wj.register_core_tools(app, cfg, root)
                agent = wj.Agent(app, cfg, root)

                orig = agent.execute_tools

                def execute_and_steer(calls):
                    res = orig(calls)
                    app.submit_line("steered instruction")
                    return res

                agent.execute_tools = execute_and_steer
                agent.run_once("loop")
                self.assertTrue(any(m.role == "user"
                                    and m.content == "steered instruction"
                                    for m in agent.messages))
                # The steer must have been sent to the model, not just stored.
                sent = [m["content"] for req in srv.requests
                        for m in req.get("messages", [])]
                self.assertIn("steered instruction", sent)

    def test_cancel_stops_the_turn(self):
        with MockChatServer(scenario="slow_content", delay=0.05) as srv:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                app = wj.WheeljackApp()
                app._renderer = QuietRenderer()
                cfg = wj.Config(api_base=srv.api_base, cwd=root, request_timeout=5)
                agent = wj.Agent(app, cfg, root)
                app.request_cancel()
                t0 = time.time()
                agent.run_once("hi")
                self.assertLess(time.time() - t0, 2.0)
                self.assertFalse(app.turn_active)

    def test_close_detaches_the_reasoning_observer(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = wj.Config(cwd=Path(td))
            app = wj.WheeljackApp(cfg)
            agent = wj.Agent(app, cfg, Path(td))
            self.assertIn(agent._accumulate_reasoning,
                          app.observers.get(wj.ReasoningDelta, []))
            agent.close()
            self.assertNotIn(agent._accumulate_reasoning,
                             app.observers.get(wj.ReasoningDelta, []))
            self.assertIsNone(app.agent)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
class TestSessions(unittest.TestCase):
    def test_session_lifecycle_save_and_restore(self):
        with tempfile.TemporaryDirectory() as td:
            sdir = Path(td)
            cfg = wj.Config(sessions_dir=sdir, session_enabled=True)
            app = wj.WheeljackApp(cfg)
            app._renderer = QuietRenderer()

            agent = wj.Agent(app, cfg, sdir, session_name="test-session")
            self.assertEqual(agent.session_context.name, "test-session")
            sid = agent.session_context.session_id

            agent._add_user_message("user query")
            agent._save_session()

            session_file = sdir / f"{sid}.json"
            self.assertTrue(session_file.exists())
            data = json.loads(session_file.read_text(encoding="utf-8"))
            self.assertEqual(data["context"]["name"], "test-session")

            # Resuming session restores user query
            app2 = wj.WheeljackApp(cfg)
            app2._renderer = QuietRenderer()
            agent2 = wj.Agent(app2, cfg, sdir, session_name="test-session")
            self.assertEqual(agent2.session_context.session_id, sid)
            self.assertTrue(any(m.role == "user" and m.content == "user query" for m in agent2.messages))

    def test_tool_calls_survive_save_and_resume(self):
        """A resumed tool round must stay replayable: the assistant message
        carrying tool_calls is what owns the tool results that follow."""
        cfg = wj.Config(session_enabled=True)
        with tempfile.TemporaryDirectory() as td:
            sdir = Path(td) / "sessions"
            cfg.sessions_dir = sdir

            app = wj.WheeljackApp(cfg)
            app._renderer = QuietRenderer()
            agent = wj.Agent(app, cfg, sdir, session_name="tool-resume")
            call = wj.ToolCall("call_9", {"name": "read_file",
                                          "arguments": '{"path": "a.txt"}'})
            agent.messages.append(wj.Message(
                role="assistant", content="", tool_calls=[call],
                timestamp=datetime.now().isoformat()))
            agent._add_tool_response(call, "file body")
            agent._save_session()
            sid = agent.session_context.session_id

            app2 = wj.WheeljackApp(cfg)
            app2._renderer = QuietRenderer()
            agent2 = wj.Agent(app2, cfg, sdir, session_name="tool-resume")
            self.assertEqual(agent2.session_context.session_id, sid)

            owning = [m for m in agent2.messages if m.tool_calls]
            self.assertEqual(len(owning), 1)
            self.assertEqual(owning[0].tool_calls[0].id, "call_9")
            self.assertEqual(owning[0].tool_calls[0].function["name"], "read_file")

            # Replay through the wire format the API actually sees.
            payload = [agent2.llm._message_to_dict(m) for m in agent2.messages]
            tool_msg = [m for m in payload if m["role"] == "tool"]
            self.assertEqual(len(tool_msg), 1)
            self.assertEqual(tool_msg[0]["tool_call_id"], "call_9")
            owners = [m for m in payload if m.get("tool_calls")]
            self.assertEqual(len(owners), 1)
            self.assertEqual(owners[0]["tool_calls"][0]["id"], "call_9")

    def test_sessions_disabled_writes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            sdir = Path(td) / "sessions"
            cfg = wj.Config(sessions_dir=sdir, session_enabled=False)
            app = wj.WheeljackApp(cfg)
            app._renderer = QuietRenderer()
            agent = wj.Agent(app, cfg, Path(td))
            self.assertIsNone(agent.session_context)
            agent._add_user_message("hi")
            agent._save_session()
            self.assertFalse(sdir.exists())


# ---------------------------------------------------------------------------
# AGENTS.md
# ---------------------------------------------------------------------------
class TestAgentsMd(unittest.TestCase):
    def test_agents_md_injected_and_truncated(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "AGENTS.md").write_text("Special Instructions:\n" + ("x" * 25000))
            cfg = wj.Config(cwd=root, max_agents_md_chars=500,
                            session_enabled=False)
            app = wj.WheeljackApp(cfg)
            agent = wj.Agent(app, cfg, root)
            sys_msg = agent.messages[0].content
            self.assertIn("Special Instructions", sys_msg)
            self.assertIn("AGENTS.md truncated at 500 characters", sys_msg)
            # The marker must be present, and the payload really capped.
            self.assertLess(len(sys_msg), 25000)
            self.assertIn("more omitted", sys_msg)

    def test_agents_md_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "AGENTS.md").write_text("Should not appear")
            cfg = wj.Config(cwd=root, agents_md_enabled=False,
                            session_enabled=False)
            app = wj.WheeljackApp(cfg)
            agent = wj.Agent(app, cfg, root)
            sys_msg = agent.messages[0].content
            self.assertNotIn("Should not appear", sys_msg)

    def test_agents_md_found_in_parent_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "AGENTS.md").write_text("Parent rules here")
            child = root / "a" / "b"
            child.mkdir(parents=True)
            cfg = wj.Config(cwd=child, session_enabled=False)
            app = wj.WheeljackApp(cfg)
            agent = wj.Agent(app, cfg, child)
            self.assertIn("Parent rules here", agent.messages[0].content)

    def test_agents_md_explicit_path_override(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "AGENTS.md").write_text("discovered copy")
            custom = root / "custom.md"
            custom.write_text("explicit copy")
            cfg = wj.Config(cwd=root, agents_md_file=str(custom),
                            session_enabled=False)
            app = wj.WheeljackApp(cfg)
            agent = wj.Agent(app, cfg, root)
            sys_msg = agent.messages[0].content
            self.assertIn("explicit copy", sys_msg)
            self.assertNotIn("discovered copy", sys_msg)


# ---------------------------------------------------------------------------
# Two-tier plugin resolution
# ---------------------------------------------------------------------------
class TestTwoTierPlugins(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.inst_dir = self.tmp / "installed"
        self.proj_dir = self.tmp / "project"
        self.inst_dir.mkdir()
        self.proj_dir.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_project_overrides_installed_on_collision(self):
        (self.inst_dir / "myplug.py").write_text(
            "def register(app):\n"
            "    app.add_tool('test_tool', {}, lambda a: 'from_installed')\n")
        (self.proj_dir / "myplug.py").write_text(
            "def register(app):\n"
            "    app.add_tool('test_tool', {}, lambda a: 'from_project')\n")

        app = wj.WheeljackApp()
        wj.load_plugins(app, project_dir=self.proj_dir, installed_dir=self.inst_dir)
        self.assertEqual(app.plugin_names, ["myplug"])
        tool_fn = app.tools["test_tool"][1]
        self.assertEqual(tool_fn({}), "from_project")
        self.assertTrue(any("overridden by project tier" in m for m in app._log))

    def test_installed_only_plugin_loads(self):
        (self.inst_dir / "only_inst.py").write_text(
            "def register(app):\n"
            "    app.add_tool('inst_tool', {}, lambda a: 'ok')\n")
        app = wj.WheeljackApp()
        wj.load_plugins(app, project_dir=self.proj_dir, installed_dir=self.inst_dir)
        self.assertIn("only_inst", app.plugin_names)
        self.assertIn("inst_tool", app.tools)


# ---------------------------------------------------------------------------
# Plugin installer
# ---------------------------------------------------------------------------
def _make_file_server(files):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.lstrip("/")
            if path in files:
                content = files[path]
                if isinstance(content, str):
                    content = content.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a):
            pass

    return Handler


class TestPluginInstaller(unittest.TestCase):
    def test_https_required_by_default(self):
        with self.assertRaises(wj.InstallError) as ctx:
            wj.install_plugin("http://example.com/plugin.py")
        self.assertIn("HTTPS is required", str(ctx.exception))

        with self.assertRaises(wj.InstallError) as ctx:
            wj.install_plugin("plugin", repo_base="http://example.com")
        self.assertIn("HTTPS is required", str(ctx.exception))

    def test_install_verified_sidecar_and_uninstall(self):
        code = b"def register(app): pass\n"
        digest = hashlib.sha256(code).hexdigest()
        files = {
            ".wheeljack/plugins/demo.py": code,
            ".wheeljack/plugins/demo.py.sha256": digest.encode("utf-8"),
        }

        httpd = http.server.HTTPServer(("127.0.0.1", 0), _make_file_server(files))
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            repo_url = f"http://127.0.0.1:{httpd.server_address[1]}"
            with tempfile.TemporaryDirectory() as td:
                target_dir = Path(td) / "plugins"

                target = wj.install_plugin(
                    "demo", repo_base=repo_url, target_dir=target_dir,
                    insecure=True, confirm=lambda p: True, out=lambda s: None)
                self.assertIsNotNone(target)
                self.assertTrue(target.exists())
                self.assertEqual(target.read_bytes(), code)

                with self.assertRaises(wj.InstallError):
                    wj.install_plugin("demo", repo_base=repo_url, target_dir=target_dir,
                                      insecure=True, confirm=lambda p: True, out=lambda s: None)

                target = wj.install_plugin(
                    "demo", repo_base=repo_url, target_dir=target_dir,
                    force=True, insecure=True, confirm=lambda p: True, out=lambda s: None)
                self.assertTrue(target.exists())

                ok = wj.uninstall_plugin("demo", target_dir=target_dir, confirm=lambda p: True, out=lambda s: None)
                self.assertTrue(ok)
                self.assertFalse(target.exists())
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_explicit_sha256_and_decline_and_missing_sidecar(self):
        code = b"def register(app): pass\n"
        digest = hashlib.sha256(code).hexdigest()
        files = {".wheeljack/plugins/nocheck.py": code}   # no sidecar

        httpd = http.server.HTTPServer(("127.0.0.1", 0), _make_file_server(files))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            repo_url = f"http://127.0.0.1:{httpd.server_address[1]}"
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)

                # An explicit --sha256 that does not match => abort, no write.
                tdir = root / "mismatch"
                with self.assertRaises(wj.InstallError) as ctx:
                    wj.install_plugin("nocheck", repo_base=repo_url,
                                      target_dir=tdir,
                                      sha256_hex="0" * 64,
                                      insecure=True, confirm=lambda p: True,
                                      out=lambda s: None)
                self.assertIn("sha256 mismatch", str(ctx.exception))
                self.assertFalse((tdir / "nocheck.py").exists())

                # User declining the confirm => nothing written.
                tdir = root / "declined"
                target = wj.install_plugin("nocheck", repo_base=repo_url,
                                           target_dir=tdir,
                                           sha256_hex=digest, insecure=True,
                                           confirm=lambda p: False, out=lambda s: None)
                self.assertIsNone(target)
                self.assertFalse((tdir / "nocheck.py").exists())

                # Matching explicit --sha256 => installed.
                tdir = root / "verified"
                target = wj.install_plugin("nocheck", repo_base=repo_url,
                                           target_dir=tdir,
                                           sha256_hex=digest, insecure=True,
                                           confirm=lambda p: True, out=lambda s: None)
                self.assertTrue(target.exists())
                self.assertEqual(target.read_bytes(), code)

                # No sidecar + --insecure => unverified but installed (loud warn).
                tdir = root / "unverified"
                target = wj.install_plugin("nocheck", repo_base=repo_url,
                                           target_dir=tdir, insecure=True,
                                           confirm=lambda p: True, out=lambda s: None)
                self.assertTrue(target.exists())
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_no_sidecar_over_https_needs_sha256_or_insecure(self):
        """The 'no checksum available' branch is only reachable over HTTPS, so
        exercise it with a stubbed fetcher (the only way to test https://)."""
        code = b"def register(app): pass\n"
        digest = hashlib.sha256(code).hexdigest()
        base = "https://example.invalid/repo"
        plugin_path = "/repo/.wheeljack/plugins/plain.py"
        served = {plugin_path: code}

        real_get = wj._http_get

        def fake_get(url, timeout=30):
            from urllib.parse import urlparse
            path = urlparse(url).path
            if path in served:
                return served[path]
            raise Exception(f"404 stub for {url}")

        wj._http_get = fake_get
        try:
            with tempfile.TemporaryDirectory() as td:
                target_dir = Path(td) / "plugins"

                # No sidecar, not insecure => refuse, nothing written.
                with self.assertRaises(wj.InstallError) as ctx:
                    wj.install_plugin("plain", repo_base=base,
                                      target_dir=target_dir,
                                      confirm=lambda p: True, out=lambda s: None)
                self.assertIn("no checksum available", str(ctx.exception))
                self.assertFalse((target_dir / "plain.py").exists())

                # --sha256 supplied => verified and written.
                target = wj.install_plugin("plain", repo_base=base,
                                           target_dir=target_dir,
                                           sha256_hex=digest,
                                           confirm=lambda p: True, out=lambda s: None)
                self.assertEqual(target.read_bytes(), code)
        finally:
            wj._http_get = real_get

    def test_uninstall_absent_plugin_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(wj.InstallError):
                wj.uninstall_plugin("ghost", target_dir=Path(td) / "plugins",
                                    confirm=lambda p: True, out=lambda s: None)

    def test_checksum_mismatch_aborts_without_writing(self):
        code = b"def register(app): pass\n"
        files = {
            ".wheeljack/plugins/bad.py": code,
            ".wheeljack/plugins/bad.py.sha256": b"00000000000000000000000000000000",
        }
        httpd = http.server.HTTPServer(("127.0.0.1", 0), _make_file_server(files))
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            repo_url = f"http://127.0.0.1:{httpd.server_address[1]}"
            with tempfile.TemporaryDirectory() as td:
                target_dir = Path(td) / "plugins"
                with self.assertRaises(wj.InstallError) as ctx:
                    wj.install_plugin("bad", repo_base=repo_url, target_dir=target_dir,
                                      insecure=True, confirm=lambda p: True, out=lambda s: None)
                self.assertIn("sha256 mismatch", str(ctx.exception))
                self.assertFalse((target_dir / "bad.py").exists())
        finally:
            httpd.shutdown()
            httpd.server_close()


# ---------------------------------------------------------------------------
# TUI plugin updates (non-streamed answer and dim styling)
# ---------------------------------------------------------------------------
class TestTuiPluginUpdates(unittest.TestCase):
    def test_turn_ended_renders_non_streamed_answer(self):
        sys.path.insert(0, str(REPO / ".wheeljack" / "plugins"))
        import tui
        app = wj.WheeljackApp()
        renderer = tui.TuiRenderer(app)

        mock_agent = type("MockAgent", (), {
            "last_answer": "non-streamed answer",
            "last_streamed": False,
        })()
        app.agent = mock_agent

        renderer.turn_ended(wj.TurnEnded())
        with renderer._lock:
            lines = [txt for txt, dim in renderer._lines]
        self.assertIn("non-streamed answer", lines)

    def test_tool_started_and_finished_dim_styling(self):
        sys.path.insert(0, str(REPO / ".wheeljack" / "plugins"))
        import tui
        app = wj.WheeljackApp()
        renderer = tui.TuiRenderer(app)

        renderer.tool_started(wj.ToolCallStarted("1", "read_file", "{'path': 'a.txt'}"))
        renderer.tool_finished(wj.ToolCallFinished("1", "read_file", "ok", True))

        with renderer._lock:
            tool_lines = [(txt, dim) for txt, dim in renderer._lines if "[tool]" in txt or "[OK]" in txt]
        self.assertEqual(len(tool_lines), 2)
        self.assertTrue(all(dim for txt, dim in tool_lines))


if __name__ == "__main__":
    unittest.main(verbosity=2)
