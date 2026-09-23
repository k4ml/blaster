#!/usr/bin/env python3
"""Tests for blaster. Run with:  python tests.py

Stdlib-only (no pytest). Covers the safe-by-default edit gate, streamed
responses, the live reasoning panel, and the HTTP parsing paths. Mock LLM
servers are spun up in-process on ephemeral ports.
"""
import contextlib
import http.server
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blaster  # noqa: E402


# ---------------------------------------------------------------------------
# Mock OpenAI-compatible chat server
# ---------------------------------------------------------------------------
def _sse(delta, finish=None):
    chunk = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(chunk)}\n\n".encode()


def _usage_event():
    return b'data: {"usage":{"prompt_tokens":3,"completion_tokens":5}}\n\n'


def _make_handler(srv):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(n) if n else b""
            try:
                srv.requests.append(json.loads(raw))
            except ValueError:
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
            for event in srv.events():
                self.wfile.write(event)
                self.wfile.flush()
                if srv.delay:
                    time.sleep(srv.delay)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, *a):
            pass

    return Handler


class MockChatServer:
    """Streams a canned SSE response for a named scenario.

    Scenarios: slow_content, reasoning_content, reasoning_only, toolcall,
    plain. If `single_json` is set, a non-streaming JSON body is returned
    instead (to exercise the fallback path).
    """

    def __init__(self, scenario="plain", delay=0.0, single_json=None):
        self.scenario = scenario
        self.delay = delay
        self.single_json = single_json
        self.requests = []
        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), _make_handler(self))
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def api_base(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def events(self):
        s = self.scenario
        out = []
        if s == "slow_content":
            for w in ["Hello", " from", " streamed", " blaster", " test", "."]:
                out.append(_sse({"content": w}))
        elif s == "reasoning_content":
            for r in ["Let me think\n", "step one\n", "step two\n"]:
                out.append(_sse({"reasoning": r}))
            for c in ["Hello ", "world."]:
                out.append(_sse({"content": c}))
        elif s == "reasoning_only":
            for r in ["only reasoning here\n", "no content marker\n"]:
                out.append(_sse({"reasoning": r}))
        elif s == "toolcall":
            out.append(_sse({"tool_calls": [{"index": 0, "id": "call_1", "function": {
                "name": "read_file", "arguments": '{"path":"a.txt"}'}}]}))
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
# Helpers
# ---------------------------------------------------------------------------
def _make_agent(api_base=None, cwd=None, **overrides):
    """Build an agent with sessions off; stdout suppressed during construction."""
    cfg = blaster.Config()
    cfg.session_enabled = False
    cfg.non_interactive = True
    cfg.request_timeout = 10
    if api_base:
        cfg.api_base = api_base
    if cwd:
        cfg.cwd = cwd
    for key, val in overrides.items():
        setattr(cfg, key, val)
    with contextlib.redirect_stdout(io.StringIO()):
        return blaster.BasicCodingAgent(cfg)


class captured_io:
    """Redirect stdout/stdin and pin _USE_COLOR for one call.

    sys.stdin is swapped for a StringIO whose fileno() raises, so the panel's
    cbreak path can never touch (or leave in cbreak) a real terminal.
    """

    def __init__(self, color):
        self.color = color
        self.buf = io.StringIO()

    def __enter__(self):
        self._out, self._in, self._color = sys.stdout, sys.stdin, blaster._USE_COLOR
        sys.stdout = self.buf
        sys.stdin = io.StringIO()
        blaster._USE_COLOR = self.color
        return self.buf

    def __exit__(self, *a):
        sys.stdout, sys.stdin, blaster._USE_COLOR = self._out, self._in, self._color


def _content(resp):
    return resp.choices[0]["message"].get("content", "")


def _call(agent, color=False):
    """Call the LLM with stdout/stdin captured (never touches a real TTY).

    Returns (response, captured_output)."""
    with captured_io(color) as buf:
        resp = agent._call_llm([blaster.Message(role="user", content="hi")])
    return resp, buf.getvalue()


# ---------------------------------------------------------------------------
# Safe-by-default edit gate
# ---------------------------------------------------------------------------
class TestEditGate(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "target.txt").write_text("hello world\n")

    def tearDown(self):
        self._tmp.cleanup()

    def _agent(self, allow_edits):
        return _make_agent(cwd=self.root, allow_edits=allow_edits)

    def test_edit_blocked_without_write(self):
        result = self._agent(False)._edit_file("target.txt", "hello", "goodbye")
        self.assertTrue(result.startswith("[edit blocked"), result)
        self.assertEqual((self.root / "target.txt").read_text(), "hello world\n")

    def test_write_blocked_without_write(self):
        result = self._agent(False)._write_file("new.txt", "data")
        self.assertTrue(result.startswith("[edit blocked"), result)
        self.assertFalse((self.root / "new.txt").exists())

    def test_edit_allowed_with_write(self):
        result = self._agent(True)._edit_file("target.txt", "hello", "goodbye")
        self.assertTrue(result.startswith("Successfully edited"), result)
        self.assertEqual((self.root / "target.txt").read_text(), "goodbye world\n")

    def test_write_allowed_with_write(self):
        result = self._agent(True)._write_file("new.txt", "data")
        self.assertTrue(result.startswith("Successfully wrote"), result)
        self.assertEqual((self.root / "new.txt").read_text(), "data")


# ---------------------------------------------------------------------------
# Streaming transport
# ---------------------------------------------------------------------------
class TestStreaming(unittest.TestCase):
    def test_requests_stream_true(self):
        with MockChatServer("plain") as srv:
            _call(_make_agent(srv.api_base))
            self.assertIs(srv.requests[0].get("stream"), True)

    def test_slow_stream_survives_short_timeout(self):
        # 6 chunks x 0.4s = ~2.4s total, but timeout is 1.5s. Streaming keeps
        # each socket read under the timeout; a non-streamed read-all would fail.
        with MockChatServer("slow_content", delay=0.4) as srv:
            agent = _make_agent(srv.api_base, request_timeout=1.5)
            start = time.time()
            resp, _ = _call(agent)
            elapsed = time.time() - start
        self.assertEqual(_content(resp), "Hello from streamed blaster test.")
        self.assertGreater(elapsed, 1.5)

    def test_single_json_fallback(self):
        body = {"choices": [{"message": {"role": "assistant",
                "content": "single json answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2}}
        with MockChatServer(single_json=body) as srv:
            agent = _make_agent(srv.api_base)
            resp, _ = _call(agent)
        self.assertEqual(_content(resp), "single json answer")
        self.assertFalse(agent._content_streamed)


# ---------------------------------------------------------------------------
# Live reasoning panel + answer streaming
# ---------------------------------------------------------------------------
class TestThinkingPanel(unittest.TestCase):
    def test_reasoning_collapsed_and_answer_streams(self):
        with MockChatServer("reasoning_content") as srv:
            agent = _make_agent(srv.api_base)
            resp, out = _call(agent, color=True)
        self.assertEqual(_content(resp), "Hello world.")
        self.assertTrue(agent._content_streamed)
        self.assertIn("[Ctrl+O to expand]", out)          # collapsed counter
        self.assertIn("Hello world.", out)                # answer streamed inline
        self.assertNotIn("Let me think", _content(resp))  # reasoning kept separate

    def test_reasoning_only_surfaced_as_answer(self):
        # Ollama edge case: entire answer lands in reasoning, content empty.
        with MockChatServer("reasoning_only") as srv:
            agent = _make_agent(srv.api_base)
            resp, _ = _call(agent, color=True)
        self.assertTrue(_content(resp).startswith("only reasoning here"))
        self.assertFalse(agent._content_streamed)

    def test_toolcall_reassembled(self):
        with MockChatServer("toolcall") as srv:
            agent = _make_agent(srv.api_base)
            resp, _ = _call(agent)
        calls = resp.choices[0]["message"].get("tool_calls", [])
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(resp.choices[0].get("finish_reason"), "tool_calls")

    def test_no_reasoning_no_counter(self):
        with MockChatServer("plain") as srv:
            agent = _make_agent(srv.api_base)
            resp, out = _call(agent, color=True)
        self.assertEqual(_content(resp), "Plain answer.")
        self.assertNotIn("[Ctrl+O", out)

    def test_piped_is_silent_but_answer_streams(self):
        with MockChatServer("reasoning_content") as srv:
            agent = _make_agent(srv.api_base)
            resp, out = _call(agent, color=False)
        self.assertEqual(_content(resp), "Hello world.")
        self.assertNotIn("[Ctrl+O", out)          # no counter control codes
        self.assertIn("Hello world.", out)        # answer still streams


# ---------------------------------------------------------------------------
# AGENTS.md project instructions
# ---------------------------------------------------------------------------
class TestAgentsMd(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._old_depth = blaster.AGENTS_MD_MAX_DEPTH

    def tearDown(self):
        blaster.AGENTS_MD_MAX_DEPTH = self._old_depth
        self._tmp.cleanup()

    def _system(self, cwd, depth=1, **overrides):
        # Bound the upward search so tests stay hermetic (no /tmp or / AGENTS.md).
        blaster.AGENTS_MD_MAX_DEPTH = depth
        agent = _make_agent(cwd=cwd, **overrides)
        return agent, agent.messages[0].content

    def test_injected_from_cwd(self):
        (self.root / "AGENTS.md").write_text("Always use tabs.\n")
        agent, sys_prompt = self._system(self.root)
        self.assertEqual(agent.agents_md_path, self.root / "AGENTS.md")
        self.assertIn("Always use tabs.", sys_prompt)
        self.assertIn("AGENTS.md", sys_prompt)

    def test_found_in_parent_dir(self):
        (self.root / "AGENTS.md").write_text("Repo-wide rule.\n")
        child = self.root / "sub"
        child.mkdir()
        agent, sys_prompt = self._system(child, depth=2)
        self.assertEqual(agent.agents_md_path, self.root / "AGENTS.md")
        self.assertIn("Repo-wide rule.", sys_prompt)

    def test_absent_when_no_file(self):
        agent, sys_prompt = self._system(self.root)
        self.assertIsNone(agent.agents_md_path)
        self.assertNotIn("AGENTS.md", sys_prompt)

    def test_disabled_skips(self):
        (self.root / "AGENTS.md").write_text("Should be ignored.\n")
        agent, sys_prompt = self._system(self.root, agents_md_enabled=False)
        self.assertIsNone(agent.agents_md_path)
        self.assertNotIn("Should be ignored.", sys_prompt)

    def test_explicit_file_overrides_discovery(self):
        rules = self.root / "custom-rules.md"
        rules.write_text("Custom rules here.\n")
        other = self.root / "elsewhere"
        other.mkdir()
        (other / "AGENTS.md").write_text("Should not be used.\n")
        agent, sys_prompt = self._system(other, agents_md_file=str(rules))
        self.assertEqual(agent.agents_md_path, rules)
        self.assertIn("Custom rules here.", sys_prompt)
        self.assertNotIn("Should not be used.", sys_prompt)

    def test_missing_explicit_file_ignored(self):
        agent, sys_prompt = self._system(
            self.root, agents_md_file=str(self.root / "nope.md"))
        self.assertIsNone(agent.agents_md_path)
        self.assertNotIn("AGENTS.md", sys_prompt)

    def test_truncated_when_too_large(self):
        (self.root / "AGENTS.md").write_text("x" * 5000)
        _, sys_prompt = self._system(self.root, max_agents_md_chars=100)
        self.assertIn("AGENTS.md truncated", sys_prompt)


# ---------------------------------------------------------------------------
# Multiline input editor (redraw)
# ---------------------------------------------------------------------------
class TestEnhancedInputRedraw(unittest.TestCase):
    def _redraw(self, line, pos, cur_row, prev_rows, prompt=">> ", width=None):
        old_cols = os.environ.get("COLUMNS")
        if width is not None:
            os.environ["COLUMNS"] = str(width)
        try:
            with captured_io(color=False) as buf:
                ei = blaster.EnhancedInput()
                ei._prompt = prompt
                ei.line, ei.pos = line, pos
                ei._cur_row, ei._prev_rows = cur_row, prev_rows
                ei._redraw()
        finally:
            if old_cols is None:
                os.environ.pop("COLUMNS", None)
            else:
                os.environ["COLUMNS"] = old_cols
        return buf.getvalue(), ei

    def test_cursor_on_first_row_does_not_move_up(self):
        # Multiline buffer, cursor at the very top: redraw must NOT move up.
        # (The old code moved up prev_rows-1 from the cursor and smeared the
        # block above — the reported "partial text repeated below" bug.)
        out, ei = self._redraw("ab\ncd\nef", 0, 0, 3)
        self.assertFalse(out.startswith("\x1b["), out)
        self.assertEqual(ei._cur_row, 0)
        self.assertEqual(ei._prev_rows, 3)
        self.assertIn(">> ab", out)
        self.assertIn(">> ef", out)

    def test_cursor_on_middle_row_moves_up_by_that_row(self):
        # pos sits in "cd" (row 1), so redraw moves up exactly 1 row to reach top.
        out, ei = self._redraw("ab\ncd\nef", 4, 1, 3)
        self.assertTrue(out.startswith("\x1b[1A"), out)
        self.assertIn(">> cd", out)
        self.assertEqual(ei._cur_row, 1)

    def test_shrink_clears_leftover_rows(self):
        # Was 3 rows, now 1 (e.g. Ctrl+U): the two stale rows must be cleared.
        out, ei = self._redraw("x", 1, 2, 3)
        self.assertEqual(out.count("\r\n\x1b[K"), 2, out)
        self.assertEqual(ei._prev_rows, 1)
        self.assertEqual(ei._cur_row, 0)

    def test_wrapped_line_counts_extra_rows(self):
        # Width 10, line 16 chars + ">> " = 19 cols -> 2 screen rows; the end
        # cursor is on the 2nd row. Row tracking must account for the wrap.
        line = "abcdefghijklmnop"
        out, ei = self._redraw(line, len(line), 0, 1, width=10)
        self.assertEqual(ei._prev_rows, 2)
        self.assertEqual(ei._cur_row, 1)

    def test_moves_up_across_wrapped_lines(self):
        # Row 0 wraps to 2 screen rows, so a cursor on the 2nd logical line
        # (screen row 2) must move up 2 rows to reach the top.
        line = "aaaaaaaaaa\nb"
        out, ei = self._redraw(line, len(line), 2, 3, width=10)
        self.assertTrue(out.startswith("\x1b[2A"), out)
        self.assertEqual(ei._cur_row, 2)

    def test_non_tty_readline_reads_a_line(self):
        with captured_io(color=False):
            ei = blaster.EnhancedInput()
            sys.stdin = io.StringIO("hello world\n")
            line = ei.readline()
        self.assertEqual(line, "hello world")


if __name__ == "__main__":
    unittest.main(verbosity=2)
