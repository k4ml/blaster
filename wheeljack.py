#!/usr/bin/env python3
"""
Wheeljack — blaster's successor: a self-contained, stdlib-only server-ops and
coding agent with a bootstrap + plugin distribution model.

One file to fetch:
    wget https://raw.githubusercontent.com/k4ml/blaster/main/wheeljack.py
It runs a real agent with zero other files present. Everything else is
optional and fetched by the bootstrap (--install NAME) from the same repo.

Layout
------
    wheeljack.py                     self-contained core (bootstrap + agent)
    .wheeljack/plugins/              project tier (user-created next to wheeljack.py)
      tui.py                         the curses TUI (repo dogfoods this tier)
    ~/.wheeljack/                    state dir, created on first run
      plugins/                       installed tier (bootstrap-downloaded)
      sessions/                      session JSON
      config.json                    optional defaults (model, api_base, ...)

Plugins are two tiers, both always loaded (project overrides installed on a
name collision); a plugin is just <name>.py in one of those dirs exposing
register(app). Extended tools (web_fetch, grep, ...) ship as plugins, not core.

Usage:
    python wheeljack.py                       # curses TUI if a plugin provides one, else stdio
    python wheeljack.py --model qwen3:27b --api-base http://localhost:11434/v1 -p "list the files here"
    python wheeljack.py --install tui         # fetch + checksum + confirm from the repo
    python wheeljack.py --list-plugins        # installed vs project vs available
    python wheeljack.py --session my-task     # resume/create a named session
    python wheeljack.py --session             # list saved sessions
    python wheeljack.py --demo                # fake agent, no endpoint (TUI smoke test)
"""
from __future__ import annotations

import argparse
import atexit
import collections
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
import os
import queue
import random
import re
import select
import shutil
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# Bootstrap / distribution constants
# --------------------------------------------------------------------------
DEFAULT_PLUGIN_REPO = "https://raw.githubusercontent.com/k4ml/blaster/main"
# Official plugins shipped in our repo (the ones --install can fetch). The
# repo dogfoods the project tier; each is <repo>/.wheeljack/plugins/<name>.py.
REPO_PLUGINS = ("tui",)


def _home_dir() -> Path:
    """Wheeljack's state directory (~/.wheeljack by default)."""
    return Path(os.getenv("WHEELJACK_HOME", str(Path.home() / ".wheeljack")))


# Project instructions convention: an AGENTS.md file (looked up from the
# working directory upward) whose contents are injected into the system prompt.
AGENTS_MD_FILENAME = "AGENTS.md"
AGENTS_MD_MAX_DEPTH = 10   # how far up from the cwd to look for AGENTS.md


# Command patterns needing y/N approval (whole-word; harmless uses like
# `grep rm` are not flagged). Declined commands are remembered for the session.
DESTRUCTIVE_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+", r"\bmkfs(?:\.[a-z0-9]+)?\b",
    r"\bdd\b", r"\bmkswap\b|\bswapoff\b", r"\bparted\b|\bfdisk\b|\bsfdisk\b",
    r"\bkill\s+-9\b|\bpkill\s+-9\b", r"\bgit\s+push\s+(-f|--force)\b", r"\bmv\s+/\s+",
]
SUDO_PATTERN = re.compile(r"(^|[;&|]\s*)sudo\s+")

# Approval prompts must never be able to wedge a headless run: if a renderer
# cannot answer within this window, confirm() falls back to its default (deny).
CONFIRM_TIMEOUT = 300.0


# --------------------------------------------------------------------------
# Events — the vocabulary the agent loop speaks. Core code calls app.emit(event)
# instead of print(); the renderer decides how (or whether) to draw it.
# --------------------------------------------------------------------------

@dataclass
class TurnStarted:
    pass


@dataclass
class TurnEnded:
    pass


@dataclass
class ToolCallStarted:
    call_id: str
    name: str
    args: dict


@dataclass
class ToolCallFinished:
    call_id: str
    name: str
    result: str
    ok: bool


@dataclass
class StreamDelta:
    text: str


@dataclass
class ReasoningDelta:
    text: str


@dataclass
class SteerQueued:
    pending_count: int


@dataclass
class LogMessage:
    text: str


_EVENT_HANDLERS: Dict[type, str] = {
    TurnStarted: "turn_started",
    TurnEnded: "turn_ended",
    ToolCallStarted: "tool_started",
    ToolCallFinished: "tool_finished",
    StreamDelta: "stream_delta",
    ReasoningDelta: "reasoning_delta",
    SteerQueued: "steer_queued",
    LogMessage: "log_message",
}


# --------------------------------------------------------------------------
# ANSI color / markdown helpers (TTY-gated; never emit escapes on a pipe)
# --------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()
_CODES = {
    "red": "\x1b[31m", "green": "\x1b[32m", "yellow": "\x1b[33m",
    "blue": "\x1b[34m", "cyan": "\x1b[36m", "dim": "\x1b[2m",
    "bold": "\x1b[1m", "reset": "\x1b[0m",
}


def _c(text: str, style: str = "") -> str:
    """Wrap text in an ANSI style when stdout is a TTY; no-op otherwise."""
    code = _CODES.get(style, "")
    if not _USE_COLOR or not code:
        return text
    return f"{code}{text}\x1b[0m"


_GLOW_PATH = shutil.which("glow")  # None when glow is not installed.


def _render_markdown(text: str) -> None:
    """Render markdown via glow when it exists and stdout is a TTY."""
    if not _USE_COLOR or not _GLOW_PATH or not text.strip():
        print(text)
        return
    try:
        proc = subprocess.Popen([_GLOW_PATH], stdin=subprocess.PIPE)
        proc.communicate(text.encode(), timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        print(text)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass
class Config:
    api_base: str = os.getenv(
        "WHEELJACK_API_BASE", os.getenv("BLASTER_API_BASE", "http://localhost:11434/v1"))
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv(
        "WHEELJACK_MODEL", os.getenv("BLASTER_MODEL", "qwen3.8:27b"))
    max_tokens: int = 2000
    temperature: float = 0.2
    request_timeout: int = 300
    max_context_files: int = 20
    max_file_size: int = 100_000        # read_file cap (100KB)
    max_output_chars: int = 40_000      # run_shell output cap
    max_iterations: int = 50            # tool round limit per turn
    sessions_dir: Path = Path.home() / ".wheeljack" / "sessions"
    cwd: Path = Path.cwd()
    format_markdown: bool = True        # glow non-streamed answers (-x disables)
    session_enabled: bool = True        # -s/--no-session disables
    allow_edits: bool = False           # -w/--write enables write/edit tools
    non_interactive: bool = False       # -p single-shot mode
    agents_md_enabled: bool = True
    agents_md_file: Optional[str] = None
    max_agents_md_chars: int = 20_000
    tool_parallel_reads: bool = True    # --no-parallel disables
    parallel_workers: int = 8           # bounded read-only thread pool
    show_reasoning: bool = False        # --show-reasoning streams reasoning inline
    plugin_repo: str = os.getenv("WHEELJACK_PLUGIN_REPO", DEFAULT_PLUGIN_REPO)


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass
class Message:
    role: str
    content: str
    tool_calls: Optional[List["ToolCall"]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class ToolCall:
    id: str
    function: Dict[str, Any]
    type: str = "function"


@dataclass
class LLMResponse:
    choices: List[Dict]
    usage: Dict[str, int]
    # True when the answer arrived as stream deltas (False for the single-JSON
    # fallback); renderers use it to decide whether the final answer still
    # needs printing.
    streamed: bool = False

    @property
    def message(self) -> Dict:
        """The first choice's message ({} when the server sent none)."""
        if not self.choices:
            return {}
        return self.choices[0].get("message") or {}

    @property
    def content(self) -> str:
        """Assistant text, reassembled from the stream."""
        return self.message.get("content") or ""

    @property
    def tool_calls(self) -> List[Dict]:
        return self.message.get("tool_calls") or []


@dataclass
class SessionContext:
    session_id: str
    name: str
    created_at: str
    last_accessed: str
    message_count: int


@dataclass
class BashToolResult:
    """run_shell result: stdout/stderr with exit code."""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False

    def format(self) -> str:
        parts = []
        if self.stdout:
            parts.append(self.stdout.rstrip("\n"))
        if self.stderr:
            parts.append(f"[stderr]\n{self.stderr.rstrip(chr(10))}")
        parts.append(f"[exit code: {self.exit_code}]")
        if self.timed_out:
            parts.append("[command timed out and was terminated]")
        return "\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    """Truncate text to limit chars, keeping whole lines, with a marker."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if "\n" in cut:
        cut = cut.rsplit("\n", 1)[0]
    return f"{cut}\n... [output truncated, {len(text) - len(cut)} more chars]"


# --------------------------------------------------------------------------
# Terminal state guard: whatever sets cbreak must be able to restore it, even
# if a daemon thread is still blocked on a read when the process exits.
# --------------------------------------------------------------------------

_TTY_SAVED: List[Tuple[int, list]] = []


def _push_tty(fd: int, attrs: list) -> None:
    _TTY_SAVED.append((fd, attrs))


def _pop_tty(fd: int) -> None:
    for i in range(len(_TTY_SAVED) - 1, -1, -1):
        if _TTY_SAVED[i][0] == fd:
            del _TTY_SAVED[i]
            return


@atexit.register
def _restore_tty_at_exit() -> None:
    for fd, attrs in _TTY_SAVED:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
        except Exception:
            pass


def _term_width() -> int:
    """Terminal width in columns (COLUMNS overrides; fallback 80)."""
    try:
        return max(4, shutil.get_terminal_size((80, 24)).columns)
    except Exception:
        return 80


# --------------------------------------------------------------------------
# The modal / confirm() mechanics.
# --------------------------------------------------------------------------

class ModalRequest:
    def __init__(self, prompt: str, choices: Tuple[str, ...], default: str):
        self.prompt = prompt
        self.choices = choices
        self.default = default
        self.event = threading.Event()
        self.answer: Optional[str] = None


class BaseRenderer:
    """Shared confirm()/modal logic. Subclasses implement the actual drawing
    (_show_modal/_hide_modal) and their own event methods."""

    def __init__(self, app: Optional["WheeljackApp"] = None) -> None:
        self.app = app
        self._modal_lock = threading.Lock()
        self._modals: "collections.deque[ModalRequest]" = collections.deque()
        self._active: Optional[ModalRequest] = None
        self._shown: Optional[ModalRequest] = None

    # -- modal API -------------------------------------------------------
    def current_modal(self) -> Optional[ModalRequest]:
        with self._modal_lock:
            return self._active

    def confirm(self, prompt: str, *, choices: Tuple[str, ...] = ("y", "n"),
                default: str = "n", timeout: Optional[float] = None) -> str:
        """Ask a question and block the calling thread for the answer.

        Returns the chosen key, or `default` on timeout / cancel. Never blocks
        forever, and never orphans a concurrent caller.
        """
        req = ModalRequest(prompt, tuple(choices), default)
        with self._modal_lock:
            self._modals.append(req)
            if self._active is None:
                self._active = req
        self._refresh_modal_display()

        if not req.event.wait(timeout):
            # Timed out (or was withdrawn) before anyone answered.
            with self._modal_lock:
                if req in self._modals:
                    self._modals.remove(req)
                    if self._active is req:
                        self._active = self._modals[0] if self._modals else None
            self._refresh_modal_display()

        return req.answer if req.answer is not None else default

    def confirm_yes_no(self, prompt: str, timeout: Optional[float] = None) -> bool:
        return self.confirm(prompt, choices=("y", "n"), default="n",
                            timeout=timeout) == "y"

    def answer_modal(self, key: str) -> bool:
        """Called by whichever thread owns input, once it decides a keystroke
        answers the active modal. Returns True if there was a modal to answer."""
        with self._modal_lock:
            req = self._active
            if req is None or key not in req.choices:
                return False
            self._modals.remove(req)
            self._active = self._modals[0] if self._modals else None
            req.answer = key
            req.event.set()
        self._refresh_modal_display()
        return True

    def cancel_all(self) -> None:
        """Deny every pending modal (used when the user cancels a turn)."""
        with self._modal_lock:
            pending = list(self._modals)
            self._modals.clear()
            self._active = None
            for req in pending:
                req.answer = req.default
                req.event.set()
        self._refresh_modal_display()

    def _refresh_modal_display(self) -> None:
        """Show the current head modal, or hide when the queue drains."""
        with self._modal_lock:
            cur = self._active
            changed = cur is not self._shown
            self._shown = cur
        if not changed:
            return
        if cur is None:
            self._hide_modal()
        else:
            self._show_modal(cur)

    # -- event API, no-op defaults so a minimal renderer overrides just what
    # it cares about --
    def turn_started(self, e: TurnStarted) -> None: ...
    def turn_ended(self, e: TurnEnded) -> None: ...
    def tool_started(self, e: ToolCallStarted) -> None: ...
    def tool_finished(self, e: ToolCallFinished) -> None: ...
    def stream_delta(self, e: StreamDelta) -> None: ...
    def reasoning_delta(self, e: ReasoningDelta) -> None: ...
    def steer_queued(self, e: SteerQueued) -> None: ...
    def log_message(self, e: LogMessage) -> None: ...

    def _show_modal(self, req: ModalRequest) -> None: ...
    def _hide_modal(self) -> None: ...

    def start(self, app: "WheeljackApp") -> None:
        """Start whatever input-reading the renderer needs. Stdio spins up a
        background InputThread; the curses plugin reads input in its own loop."""

    def stop(self) -> None: ...


# --------------------------------------------------------------------------
# ConsoleWriter — the single owner of stdout for the stdio renderer.
# --------------------------------------------------------------------------

class ConsoleWriter:
    def __init__(self, stream=None) -> None:
        self._lock = threading.RLock()
        self._stream = stream if stream is not None else sys.stdout
        self._prompt = ""
        self._buf = ""
        self._visible = False      # an input line is currently drawn
        self._status = False       # a status line (thinking counter) is drawn
        self._line_start = True

    @property
    def is_tty(self) -> bool:
        try:
            return bool(self._stream.isatty())
        except Exception:
            return False

    @property
    def has_input(self) -> bool:
        with self._lock:
            return self._visible

    def emit(self, text: str) -> None:
        with self._lock:
            self._erase()
            self._stream.write(text)
            self._stream.flush()
            self._line_start = text.endswith("\n") or text == ""

    def emit_line(self, text: str = "") -> None:
        self.emit(text + "\n")

    def write_status(self, text: str) -> None:
        """Draw an overwrite-able status line (e.g. a thinking counter)."""
        with self._lock:
            self._erase()
            self._stream.write(text)
            self._stream.flush()
            self._status = True
            self._line_start = False

    def set_input(self, prompt: str, buf: str) -> None:
        """Draw (or remember) the in-progress input line."""
        with self._lock:
            self._prompt, self._buf = prompt, buf
            if not self.is_tty:
                return  # never write a prompt into piped output
            if not self._line_start:
                # Cursor is mid-line after a partial write; defer the draw so we
                # never erase streamed text.
                self._visible = False
                return
            self._erase()
            self._stream.write(prompt + buf)
            self._stream.flush()
            self._visible = True

    def clear_input(self) -> None:
        with self._lock:
            self._erase()
            self._prompt = self._buf = ""
            self._stream.flush()

    def _erase(self) -> None:
        if self._visible:
            self._stream.write("\r\x1b[K")
            self._visible = False
            self._line_start = True
        if self._status:
            self._stream.write("\r\x1b[K")
            self._status = False
            self._line_start = True


# --------------------------------------------------------------------------
# EnhancedInput — raw-mode line editor with history (ported from blaster).
# --------------------------------------------------------------------------

class EnhancedInput:
    def __init__(self) -> None:
        self.history: List[str] = []
        self.hist_i = 0
        self.line = ""
        self.pos = 0
        self._saved = ""
        self._prev_rows = 1
        self._cur_row = 0
        self._prompt = ">> "
        self._buf = b""
        if sys.stdin.isatty() and sys.stdout.isatty():
            sys.stdout.write("\x1b[?2004h")
            sys.stdout.flush()
            atexit.register(lambda: sys.stdout.write("\x1b[?2004l"))

    def _redraw(self) -> None:
        prompt = self._prompt
        plen = len(prompt)
        width = _term_width()
        lines = self.line.split("\n")
        before = self.line[:self.pos]
        cur_line = before.count("\n")
        cur_col = len(before) - (before.rfind("\n") + 1)

        def rows_of(text: str) -> int:
            return max(1, (plen + len(text) + width - 1) // width)

        total_rows = sum(rows_of(ln) for ln in lines)
        d = plen + cur_col
        cur_row = sum(rows_of(lines[i]) for i in range(cur_line)) + d // width
        cur_vis_col = d % width
        if d and d % width == 0:
            cur_row -= 1
            cur_vis_col = width - 1

        if self._cur_row:
            sys.stdout.write(f"\x1b[{self._cur_row}A")
        sys.stdout.write("\r")
        for i, ln in enumerate(lines):
            if i:
                sys.stdout.write("\r\n")
            sys.stdout.write("\x1b[K" + prompt + ln)
        extra = self._prev_rows - total_rows
        for _ in range(max(0, extra)):
            sys.stdout.write("\r\n\x1b[K")
        rows_below = max(total_rows, self._prev_rows) - 1 - cur_row
        if rows_below > 0:
            sys.stdout.write(f"\x1b[{rows_below}A")
        sys.stdout.write("\r")
        if cur_vis_col:
            sys.stdout.write(f"\x1b[{cur_vis_col}C")
        self._prev_rows = total_rows
        self._cur_row = cur_row
        sys.stdout.flush()

    def _paste(self) -> None:
        while True:
            ch = self._key()
            if ch == "\x1b":
                tail = (self._key() + self._key() + self._key()
                        + self._key() + self._key())
                if tail == "[201~":
                    break
                self._insert("\x1b" + tail)
                continue
            self._insert(ch)

    def _fill(self) -> None:
        fd = sys.stdin.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            chunk = os.read(fd, 65536)
        except (BlockingIOError, InterruptedError):
            chunk = b""
        except OSError:
            chunk = b""
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags)
        if chunk:
            self._buf += chunk

    def _key(self) -> str:
        while True:
            try:
                s = self._buf.decode("utf-8")
            except UnicodeDecodeError:
                s = ""
            if s:
                ch = s[0]
                self._buf = self._buf[len(ch.encode("utf-8")):]
                return ch
            r, _, _ = select.select([sys.stdin], [], [], None)
            if r:
                self._fill()

    def _insert(self, ch: str) -> None:
        self.line = self.line[:self.pos] + ch + self.line[self.pos:]
        self.pos += 1
        self._redraw()

    def _delete(self) -> None:
        if self.pos == 0:
            return
        self.line = self.line[:self.pos - 1] + self.line[self.pos:]
        self.pos -= 1
        self._redraw()

    def _hist(self, direction: int) -> None:
        if not self.history:
            return
        if direction < 0:
            if self.hist_i == 0:
                self._saved = self.line
            if self.hist_i < len(self.history):
                self.hist_i += 1
                self.line = self.history[-self.hist_i]
        else:
            if self.hist_i > 0:
                self.hist_i -= 1
                self.line = self.history[-self.hist_i] if self.hist_i else self._saved
        self.pos = len(self.line)
        self._redraw()

    def readline(self, prompt: str = ">> ") -> str:
        # Non-TTY (piped/scripted) fallback: no prompt, so piped output stays
        # clean (blaster printed the prompt here and polluted captured output).
        if not sys.stdin.isatty():
            try:
                line = input()
            except EOFError:
                raise KeyboardInterrupt
            if line and (not self.history or self.history[-1] != line):
                self.history.append(line)
            return line

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        _push_tty(fd, old)
        try:
            tty.setcbreak(fd)
            return self._readline_tty(prompt)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            _pop_tty(fd)

    def _readline_tty(self, prompt: str) -> str:
        self._prompt = prompt
        sys.stdout.write(prompt)
        sys.stdout.flush()
        self.line, self.pos, self.hist_i = "", 0, 0
        self._prev_rows = 1
        self._cur_row = 0
        self._buf = b""
        while True:
            ch = self._key()
            if ch in ("\r", "\n"):
                if self._buf:
                    self._insert("\n")
                    continue
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    self._fill()
                    self._insert("\n")
                    continue
                if self._cur_row < self._prev_rows - 1:
                    sys.stdout.write(f"\x1b[{self._prev_rows - 1 - self._cur_row}B")
                sys.stdout.write("\n")
                sys.stdout.flush()
                line = self.line.strip()
                if line and (not self.history or self.history[-1] != line):
                    self.history.append(line)
                return line
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch in ("\x7f", "\x08"):
                self._delete()
            elif ch == "\x15":
                self.line, self.pos = "", 0
                self._redraw()
            elif ch == "\x01":
                self.pos = 0
                self._redraw()
            elif ch == "\x05":
                self.pos = len(self.line)
                self._redraw()
            elif ch == "\x1b":
                c = self._key()
                if c == "[":
                    c2 = self._key()
                    if c2 == "2":
                        self._key(); self._key(); self._key()
                        self._paste()
                    elif c2 == "A":
                        self._hist(-1)
                    elif c2 == "B":
                        self._hist(1)
                    elif c2 == "C" and self.pos < len(self.line):
                        self.pos += 1
                        self._redraw()
                    elif c2 == "D" and self.pos > 0:
                        self.pos -= 1
                        self._redraw()
            elif ch >= " ":
                self._insert(ch)


# --------------------------------------------------------------------------
# Stdio renderer — EnhancedInput when idle, a raw single-line steer reader
# while a turn is in flight (so output and typing don't collide).
# --------------------------------------------------------------------------

class InputThread(threading.Thread):
    """The single thread allowed to read stdin. Routes each line to the slash
    dispatcher / steer_queue, and each keystroke during a turn to either the
    active modal or the steer buffer — never both, never a race."""

    def __init__(self, renderer: "StdioRenderer", app: "WheeljackApp"):
        super().__init__(daemon=True)
        self.renderer = renderer
        self.app = app
        self._stop = False
        self._eof = False
        self._editor: Optional[EnhancedInput] = None

    def run(self) -> None:
        interactive = sys.stdin.isatty()
        while not self._stop:
            if interactive and self.app.turn_active:
                line = self._read_steer_line()
            else:
                line = self._read_idle_line()
            if line is None:
                # Stdin is gone for good. The agent may still raise a confirm()
                # later — it must not hang just because nobody can answer.
                self._deny_modals_until_stopped()
                return
            if line.strip():
                self.app.submit_line(line)
                if self.app.quit_requested:
                    # No more input is coming, but the running turn may still
                    # raise a confirm() — answer those so it can finish.
                    self._deny_modals_until_stopped()
                    return

    def _read_idle_line(self) -> Optional[str]:
        if not sys.stdin.isatty():
            try:
                return input()
            except EOFError:
                return None
        if self._editor is None:
            self._editor = EnhancedInput()
        try:
            return self._editor.readline(">> ")
        except (EOFError, KeyboardInterrupt):
            return None

    def _read_char(self, fd: int, timeout: float) -> Optional[str]:
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            return None
        try:
            data = os.read(fd, 1)
        except OSError:
            return None
        if not data:
            self._eof = True
            return None
        return data.decode("utf-8", "replace")

    def _read_steer_line(self) -> Optional[str]:
        """Raw single-line read used while a turn is running."""
        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
        except Exception:
            return self._read_idle_line()
        buf = ""
        _push_tty(fd, old)
        try:
            tty.setcbreak(fd)
            while not self._stop:
                if self.app.cancel.is_set():
                    self.renderer.writer.clear_input()
                    return ""
                ch = self._read_char(fd, 0.1)
                if ch is None:
                    if self._eof:
                        return None
                    continue
                modal = self.renderer.current_modal()
                if modal is not None:
                    key = ch.lower() if ch.isalpha() else ch
                    if key in modal.choices:
                        self.renderer.answer_modal(key)
                    elif ch in ("\r", "\n", "\x1b"):
                        self.renderer.answer_modal(modal.default)
                    continue
                if ch in ("\r", "\n"):
                    self.renderer.writer.clear_input()
                    return buf
                if ch in ("\x7f", "\x08"):
                    buf = buf[:-1]
                    self.renderer.writer.set_input(">> ", buf)
                elif ch == "\x03":
                    self.app.request_cancel()
                    return ""
                elif ch >= " ":
                    buf += ch
                    self.renderer.writer.set_input(">> ", buf)
            return ""
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass
            _pop_tty(fd)

    def _deny_modals_until_stopped(self) -> None:
        while not self._stop:
            if self.renderer.current_modal() is not None:
                self.renderer.answer_modal(
                    self.renderer.current_modal().default)
            time.sleep(0.05)

    def stop(self) -> None:
        self._stop = True


class StdioRenderer(BaseRenderer):
    def __init__(self, app: Optional["WheeljackApp"] = None, stream=None) -> None:
        super().__init__(app)
        self.writer = ConsoleWriter(stream)
        self._input_thread: Optional[InputThread] = None
        self._reasoning_lines = 0
        self._reasoning_shown = False

    def start(self, app: "WheeljackApp") -> None:
        self.app = app
        self._input_thread = InputThread(self, app)
        self._input_thread.start()

    def stop(self) -> None:
        if self._input_thread:
            self._input_thread.stop()
        self.writer.clear_input()

    def confirm(self, prompt: str, **kw) -> str:
        # A piped/scripted run has nobody to answer a prompt; auto-deny so the
        # turn can never hang waiting on stdin (mirrors blaster's rule).
        if not sys.stdin.isatty():
            default = kw.get("default", "n")
            self.writer.emit_line(f"[no tty] auto-denying: {prompt}")
            return default
        return super().confirm(prompt, **kw)

    # -- non-streamed final answers --------------------------------------
    def _format_markdown_enabled(self) -> bool:
        config = self.app.config if self.app is not None else None
        return bool(getattr(config, "format_markdown", True))

    def _glow(self, text: str) -> None:
        self.writer.clear_input()
        try:
            proc = subprocess.Popen([_GLOW_PATH], stdin=subprocess.PIPE)
            proc.communicate(text.encode(), timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self.writer.emit_line(text)

    def turn_started(self, e: TurnStarted) -> None:
        self.writer.emit_line("")
        self._reasoning_lines = 0
        if self._reasoning_shown:
            self.writer.clear_input()
            self._reasoning_shown = False

    def turn_ended(self, e: TurnEnded) -> None:
        agent = self.app.agent if self.app is not None else None
        if agent is None:
            return
        answer = getattr(agent, "last_answer", "") or ""
        if answer == "(no response)":
            self.writer.emit_line(_c("🤖 (no response from model)", "dim"))
            return
        if answer and not getattr(agent, "last_streamed", False):
            if (self._format_markdown_enabled() and self.writer.is_tty
                    and _GLOW_PATH):
                self._glow(answer)
            else:
                self.writer.emit_line(f"\n🤖 {_c(answer, 'green')}")

    def tool_started(self, e: ToolCallStarted) -> None:
        self.writer.emit_line(f"  \U0001F527 {e.name}({e.args}) ...")

    def tool_finished(self, e: ToolCallFinished) -> None:
        mark = "\u2713" if e.ok else "\u2717"
        self.writer.emit_line(f"  {mark} {e.name}: {e.result}")

    def stream_delta(self, e: StreamDelta) -> None:
        self.writer.emit(e.text)

    def reasoning_delta(self, e: ReasoningDelta) -> None:
        config = self.app.config if self.app is not None else None
        if getattr(config, "show_reasoning", False):
            self.writer.emit(e.text)
            return
        # Collapsed counter — the scrollback guard. Skipped when piped (no
        # control codes) or while the user is typing a steer prompt.
        if not self.writer.is_tty or self.writer.has_input:
            return
        n = self._count_segments(e.text)
        self._reasoning_lines += n
        label = "line" if self._reasoning_lines == 1 else "lines"
        self.writer.write_status(
            f"\r🤖 Thinking ... ({self._reasoning_lines} {label})")
        self._reasoning_shown = True

    def _count_segments(self, text: str) -> int:
        n = text.count("\n")
        if text and not text.endswith("\n"):
            n += 1
        return max(n, 1)

    def steer_queued(self, e: SteerQueued) -> None:
        self.writer.emit_line(f"  \u21B3 queued ({e.pending_count} pending)")

    def log_message(self, e: LogMessage) -> None:
        self.writer.emit_line(f"[wheeljack] {e.text}")

    def _show_modal(self, req: ModalRequest) -> None:
        opts = "/".join(req.choices)
        self.writer.emit_line(f"\n\u26A0\uFE0F  {req.prompt} [{opts}] ")

    def _hide_modal(self) -> None:
        pass


class NonInteractiveRenderer(StdioRenderer):
    """Single-shot (-p) mode: no input thread at all, confirm() always denies."""

    def start(self, app: "WheeljackApp") -> None:
        pass

    def stop(self) -> None:
        pass

    def confirm(self, prompt: str, **kw) -> str:
        default = kw.get("default", "n")
        self.writer.emit_line(f"[non-interactive] auto-denying: {prompt}")
        return default

    def turn_ended(self, e: TurnEnded) -> None:
        pass  # main() renders the non-streamed answer for -p mode


# --------------------------------------------------------------------------
# Slash commands
# --------------------------------------------------------------------------

def dispatch_slash(app: "WheeljackApp", text: str) -> bool:
    """Handle a leading `/name args` line. Returns True if it was a command
    (so the caller must NOT also send it to the agent as a turn)."""
    text = text.strip()
    if not text.startswith("/"):
        return False
    name, _, args = text[1:].partition(" ")
    name = name.strip()
    handler = app.slash_commands.get(name)
    if handler is None:
        app.log(f"unknown command: /{name} (try /help)")
        return True
    try:
        handler(app, args.strip())
    except Exception as exc:
        app.log(f"/{name} failed: {exc}")
    return True


_BUILTIN_COMMANDS = ("help", "quit", "exit", "tools", "plugins", "model",
                     "context", "sessions", "reasoning")


def _cmd_help(app: "WheeljackApp", args: str) -> None:
    lines = [
        "Wheeljack commands:",
        "  /help            show this help",
        "  /quit, /exit     exit",
        "  /tools           list registered tools",
        "  /plugins         list loaded plugins",
        "  /model [NAME]    show or set the model",
        "  /context         show recent conversation",
        "  /sessions        list saved sessions",
        "  /reasoning       show the captured reasoning block",
    ]
    for name in sorted(app.slash_commands):
        if name not in _BUILTIN_COMMANDS:
            lines.append(f"  /{name}")
    app.log("\n".join(lines))


def _cmd_quit(app: "WheeljackApp", args: str) -> None:
    app.quit_requested = True


def _cmd_tools(app: "WheeljackApp", args: str) -> None:
    app.log("tools: " + (", ".join(sorted(app.tools)) if app.tools else "(none)"))


def _cmd_plugins(app: "WheeljackApp", args: str) -> None:
    app.log("plugins: " + (", ".join(app.plugin_names) if app.plugin_names else "(none)"))


def _cmd_model(app: "WheeljackApp", args: str) -> None:
    if args:
        app.config.model = args
        app.log(f"model set to {args}")
    else:
        app.log(f"model: {app.config.model}")


def _cmd_context(app: "WheeljackApp", args: str) -> None:
    agent = app.agent
    if agent is None:
        app.log("no conversation yet")
        return
    past = [m for m in getattr(agent, "messages", [])
            if m.role in ("user", "assistant") and m.content
            and m.content != "(no response)"
            and not m.content.startswith("(system)")]
    if not past:
        app.log("no conversation yet")
        return
    lines = []
    for m in past[-20:]:
        who = "you" if m.role == "user" else "wheeljack"
        text = m.content.strip().replace("\n", " ")
        if len(text) > 300:
            text = text[:300] + "…"
        lines.append(f"  {who}: {text}")
    app.log("conversation context:\n" + "\n".join(lines))


def _cmd_sessions(app: "WheeljackApp", args: str) -> None:
    dirs = getattr(app.config, "sessions_dir", None)
    if dirs is None:
        app.log("sessions disabled")
        return
    app.log("\n".join(_sessions_table(dirs)))


def _cmd_reasoning(app: "WheeljackApp", args: str) -> None:
    agent = app.agent
    text = (getattr(agent, "reasoning_text", "") or "") if agent else ""
    if not text.strip():
        app.log("no reasoning captured this turn (use --show-reasoning to stream it)")
        return
    app.log("thinking:\n" + text.rstrip("\n"))


# --------------------------------------------------------------------------
# Sessions table
# --------------------------------------------------------------------------

def _sessions_table(sessions_dir: Path) -> List[str]:
    """Format saved sessions (most recently used first) for /sessions."""
    sessions_dir = Path(sessions_dir)
    files = sorted(sessions_dir.glob("*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return [f"No saved sessions (stored in {sessions_dir})."]
    hdr = f"{'NAME':<22} {'ID':<14} {'MSGS':>5}  {'CREATED':<10}  {'LAST USED':<10}"
    lines = [hdr, "-" * len(hdr)]
    for f in files:
        try:
            ctx = json.loads(f.read_text(encoding="utf-8")).get("context", {})
            name = ctx.get("name", ctx.get("session_id", f.stem))
            sid = ctx.get("session_id", f.stem)[:12]
            msgs = ctx.get("message_count", "?")
            created = (ctx.get("created_at") or "?")[:10]
            last = (ctx.get("last_accessed") or "?")[:10]
        except Exception:
            name, sid, msgs, created, last = f.stem, f.stem[:12], "?", "?", "?"
        lines.append(f"{name:<22} {sid:<14} {str(msgs):>5}  {created:<10}  {last:<10}")
    lines.append("")
    lines.append("Resume one with: python wheeljack.py --session NAME")
    return lines


# --------------------------------------------------------------------------
# Plugin loader — two tiers, both always loaded. A plugin is any <name>.py in
# the dedicated plugin dirs (a dedicated dir is safe to glob wholesale), with
# a register(app) function. Project overrides installed on a name collision.
# --------------------------------------------------------------------------

PLUGIN_API_VERSION = 1


def _core_module():
    return sys.modules.get("wheeljack") or sys.modules.get(__name__)


def load_plugins(app: "WheeljackApp", project_dir: Path,
                 installed_dir: Optional[Path] = None) -> None:
    """Load plugins from the installed tier (if given) and the project tier.
    On a name collision the project copy wins (the more local, more specific
    tier always overrides)."""
    tiers: List[Tuple[str, Path]] = []
    if installed_dir is not None:
        tiers.append(("installed", Path(installed_dir)))
    tiers.append(("project", Path(project_dir)))

    chosen: Dict[str, Tuple[str, Path]] = {}
    for tier, directory in tiers:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_") or path.name == "wheeljack.py":
                continue
            name = path.stem
            if name in chosen and tier == "project":
                app.log(f"plugin '{name}' overridden by project tier")
            chosen[name] = (tier, path)

    for name, (tier, path) in sorted(chosen.items()):
        _load_one(app, path, name, tier)


def _load_one(app: "WheeljackApp", path: Path, name: str, tier: str) -> None:
    modname = f"wheeljack_plugin_{name}"
    try:
        spec = importlib.util.spec_from_file_location(modname, path)
        if spec is None or spec.loader is None:
            app.log(f"plugin {name} could not be loaded; skipping")
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[modname] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(modname, None)
        app.log(f"plugin {path.name} failed to load ({exc}); skipping")
        return

    declared = getattr(module, "API_VERSION", PLUGIN_API_VERSION)
    if declared != PLUGIN_API_VERSION:
        app.log(f"plugin {path.name} targets API v{declared}, "
                f"core is v{PLUGIN_API_VERSION}; loading anyway")

    register = getattr(module, "register", None)
    if not callable(register):
        app.log(f"plugin {path.name} has no register(app); skipping")
        return
    try:
        register(app)
    except Exception as exc:
        app.log(f"plugin {path.name} register() failed ({exc}); skipping")
        return
    app.plugin_names.append(name)
    app.log(f"loaded plugin: {path.name} [{tier}]")


# --------------------------------------------------------------------------
# Bootstrap + plugin installer
# --------------------------------------------------------------------------

class InstallError(Exception):
    pass


def ensure_home(home: Optional[Path] = None) -> Path:
    """Create ~/.wheeljack/{plugins,sessions} on first run. Returns the home."""
    home = Path(home) if home is not None else _home_dir()
    (home / "plugins").mkdir(parents=True, exist_ok=True)
    (home / "sessions").mkdir(parents=True, exist_ok=True)
    return home


def _http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _default_confirm(prompt: str) -> bool:
    """Plain stdin y/N prompt (no renderer exists for install commands)."""
    sys.stdout.write(prompt + " [y/N] ")
    sys.stdout.flush()
    try:
        return sys.stdin.readline().strip().lower() in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        sys.stdout.write("\n")
        return False


def install_plugin(name: str, repo_base: str = DEFAULT_PLUGIN_REPO,
                   target_dir: Optional[Path] = None, *,
                   force: bool = False, sha256_hex: Optional[str] = None,
                   insecure: bool = False, confirm: Optional[Callable[[str], bool]] = None,
                   out: Callable[[str], None] = print,
                   home: Optional[Path] = None) -> Optional[Path]:
    """Fetch a plugin over HTTPS, verify the sha256 sidecar, confirm, and write
    atomically into the installed tier. Returns the target path, or None when
    the user declined. Raises InstallError on any abort."""
    target_dir = Path(target_dir) if target_dir else ensure_home(home) / "plugins"
    target_dir.mkdir(parents=True, exist_ok=True)

    if name.startswith(("http://", "https://")):
        url = name
        plugin_name = url.rstrip("/").rsplit("/", 1)[-1]
        if plugin_name.endswith(".py"):
            plugin_name = plugin_name[:-3]
        if not plugin_name:
            raise InstallError("could not derive plugin name from URL")
    else:
        plugin_name = name[:-3] if name.endswith(".py") else name
        url = f"{repo_base.rstrip('/')}/.wheeljack/plugins/{plugin_name}.py"

    if not url.startswith("https://") and not insecure:
        raise InstallError("HTTPS is required for plugin installation; use --insecure to override")

    try:
        body = _http_get(url)
    except urllib.error.HTTPError as exc:
        raise InstallError(f"download failed: HTTP {exc.code} for {url}") from exc
    except Exception as exc:
        raise InstallError(f"download failed: {exc}") from exc

    digest = hashlib.sha256(body).hexdigest()
    sidecar: Optional[str] = None
    try:
        raw = _http_get(url + ".sha256").decode("utf-8", "replace").strip()
        sidecar = raw.split()[0] if raw else None
    except Exception:
        sidecar = None

    if sha256_hex:
        if sha256_hex.lower() != digest:
            raise InstallError(f"sha256 mismatch: expected {sha256_hex}, got {digest}")
    elif sidecar:
        if sidecar.lower() != digest:
            raise InstallError(f"sha256 mismatch: expected {sidecar}, got {digest}")
    elif insecure:
        out("WARNING: no checksum available and --insecure given; installing unverified.")
    else:
        raise InstallError("no checksum available (no .sha256 sidecar); "
                           "pass --sha256 HEX or --insecure")

    target = target_dir / f"{plugin_name}.py"
    if target.exists() and not force:
        raise InstallError(f"{target} already exists; use --force to overwrite")

    ask = confirm if confirm is not None else _default_confirm
    if not ask(f"Install plugin '{plugin_name}' ({len(body)} bytes, "
               f"sha256 {digest[:16]}…) to {target}?"):
        out(f"aborted: {plugin_name} not installed")
        return None

    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(body)
    os.replace(tmp, target)
    return target


def uninstall_plugin(name: str, target_dir: Optional[Path] = None, *,
                     confirm: Optional[Callable[[str], bool]] = None,
                     out: Callable[[str], None] = print,
                     home: Optional[Path] = None) -> bool:
    """Remove an installed plugin (installed tier only)."""
    target_dir = Path(target_dir) if target_dir else ensure_home(home) / "plugins"
    plugin_name = name[:-3] if name.endswith(".py") else name
    target = target_dir / f"{plugin_name}.py"
    if not target.exists():
        raise InstallError(f"plugin '{plugin_name}' is not installed ({target})")
    ask = confirm if confirm is not None else _default_confirm
    if not ask(f"Uninstall plugin '{plugin_name}' ({target})?"):
        out(f"aborted: {plugin_name} kept")
        return False
    target.unlink()
    return True


def list_plugins(project_dir: Path, home: Optional[Path] = None,
                 repo_base: str = DEFAULT_PLUGIN_REPO) -> List[str]:
    """installed vs project vs available (from the repo)."""
    project_dir = Path(project_dir)
    installed = _home_dir() if home is None else Path(home)
    installed = installed / "plugins"

    def _names(d: Optional[Path]) -> List[str]:
        if d is None or not d.is_dir():
            return []
        return sorted(p.stem for p in d.glob("*.py")
                      if not p.name.startswith("_") and p.name != "wheeljack.py")

    inst = [p.stem for p in (installed.glob("*.py") if installed.is_dir() else [])
            if not p.name.startswith("_") and p.name != "wheeljack.py"]
    proj = _names(project_dir)
    lines = [
        "Installed (~/.wheeljack/plugins):",
        "  " + (", ".join(inst) if inst else "(none)"),
        f"Project ({project_dir}):",
        "  " + (", ".join(proj) if proj else "(none)"),
        f"Available (from {repo_base}):",
        "  " + ", ".join(REPO_PLUGINS),
        "",
        "Install one with: python wheeljack.py --install NAME",
    ]
    return lines


# --------------------------------------------------------------------------
# Tool registering: Core tools + safety gates. Extended tools are plugins.
# Spec shape: {"description", "parameters", "required", "parallel_safe"}.
# --------------------------------------------------------------------------

class ToolContext:
    """Per-agent tool state shared by the core handlers (config, project root,
    and the session-scoped set of declined dangerous commands)."""

    def __init__(self, app: "WheeljackApp", config: Config, project_root: Path):
        self.app = app
        self.config = config
        self.project_root = Path(project_root).expanduser().resolve()
        self._declined_commands: List[str] = []


def _tools_schema(app: "WheeljackApp") -> List[Dict]:
    """Build the OpenAI function-calling schema from every registered tool
    (core and plugins alike)."""
    tools = []
    for name, (spec, _handler) in sorted(app.tools.items()):
        params = spec.get("parameters", {})
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": spec.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": {p: {"type": t, "description": d}
                                    for p, (t, d) in params.items()},
                    "required": spec.get("required", []),
                },
            },
        })
    return tools


def _resolve_path(ctx: ToolContext, path: str) -> Path:
    """Resolve a (possibly relative) path against the project root."""
    if path.startswith("/"):
        return Path(path)
    return (ctx.project_root / path).resolve()


def _gate_edit(ctx: ToolContext, file_path: str, action: str) -> str:
    """Return '' to allow an edit, or a message explaining the block.

    Safe by default: when edit mode is off (no -w/--write), file-modifying
    tools are blocked. The confirm goes through app.confirm, so a non-TTY /
    -p run auto-denies via the renderer (no hang, no prompt pollution).
    """
    if ctx.config.allow_edits:
        return ""
    approved = ctx.app.confirm(
        f"\U0001F512 Edit mode is off (-w/--write to enable permanently). "
        f"{action}: {file_path}\nAllow this edit?",
        default="n", timeout=CONFIRM_TIMEOUT) == "y"
    if approved:
        return ""
    return ("[edit blocked: edit mode is off. Re-run with -w/--write "
            "to allow file edits. Tell the user and do not retry.]")


def _confirm_dangerous_command(ctx: ToolContext, command: str) -> bool:
    """Return True if a command may run without (further) approval."""
    reasons = []
    if any(re.search(p, command) for p in DESTRUCTIVE_PATTERNS):
        reasons.append("destructive pattern")
    if SUDO_PATTERN.search(command):
        reasons.append("uses sudo")
    if not reasons:
        return True
    # A previously-declined command stays blocked for this session, so the
    # model cannot keep retrying the same destructive/sudo operation.
    if command in ctx._declined_commands:
        return False
    approved = ctx.app.confirm(
        f"\u26A0\uFE0F  This command is flagged as {', '.join(reasons)}:\n"
        f"    {command}\nRun it?",
        default="n", timeout=CONFIRM_TIMEOUT) == "y"
    if not approved:
        ctx._declined_commands.append(command)
    return approved


def _read_file(ctx: ToolContext, path: str) -> str:
    """Read file content (truncated to max_file_size)."""
    file_path = _resolve_path(ctx, path)
    if not file_path.exists():
        return f"Error: File not found: {file_path}"
    if file_path.is_dir():
        return f"Error: {file_path} is a directory, not a file"
    file_size = file_path.stat().st_size
    if file_size > ctx.config.max_file_size:
        return (f"Error: File too large ({file_size} bytes, "
                f"max {ctx.config.max_file_size})")
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return (f"Error: {file_path} is not valid UTF-8 text "
                f"({file_size} bytes); use run_shell to inspect it")
    except Exception as exc:
        return f"Error reading file: {exc}"
    return f"--- {file_path} ---\n{content}\n--- end {file_path} ---"


def _write_file(ctx: ToolContext, path: str, content: str) -> str:
    """Write content to file, creating a .bak backup if the file exists."""
    file_path = _resolve_path(ctx, path)
    gate = _gate_edit(ctx, str(file_path), "write_file")
    if gate:
        return gate
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    if file_path.exists():
        try:
            shutil.copy2(file_path, backup_path)
        except Exception as exc:
            return f"Error backing up existing file: {exc}"
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        return f"Error writing file: {exc}"
    msg = f"Successfully wrote {len(content)} bytes to {file_path}"
    if backup_path.exists():
        msg += f" (backup: {backup_path.name})"
    return msg


def _edit_file(ctx: ToolContext, path: str, before: str, after: str) -> str:
    """Edit a file by replacing a unique substring."""
    file_path = _resolve_path(ctx, path)
    gate = _gate_edit(ctx, str(file_path), "edit_file")
    if gate:
        return gate
    if not file_path.exists():
        return f"Error: File not found: {file_path}"
    if file_path.is_dir():
        return f"Error: {file_path} is a directory"
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Error reading file: {exc}"
    if before not in content:
        return (f"Error: 'before' text not found in {file_path}. "
                f"Use read_file to get the exact current content first.")
    if content.count(before) > 1:
        return (f"Error: 'before' text is not unique in {file_path} "
                f"({content.count(before)} matches). Include more context.")
    new_content = content.replace(before, after, 1)
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    try:
        shutil.copy2(file_path, backup_path)
    except Exception as exc:
        return f"Error backing up existing file: {exc}"
    try:
        file_path.write_text(new_content, encoding="utf-8")
    except Exception as exc:
        shutil.copy2(backup_path, file_path)
        return f"Error writing file: {exc}"
    return (f"Successfully edited {file_path} "
            f"({len(before)} chars replaced, backup: {backup_path.name})")


def _list_files(ctx: ToolContext, path: str = ".", recursive: bool = False) -> str:
    """List files in a directory, optionally as a recursive tree."""
    dir_path = _resolve_path(ctx, path)
    if not dir_path.is_dir():
        return f"Error: {dir_path} is not a directory"
    try:
        if not recursive:
            files = []
            for item in sorted(dir_path.iterdir()):
                if item.is_file():
                    files.append(f"  - {item.name} ({item.stat().st_size} bytes)")
                elif item.is_dir():
                    files.append(f"  - {item.name}/")
            return f"Files in {dir_path}:\n" + "\n".join(files)

        # Recursive tree with depth limit to avoid flooding context.
        lines = [f"{dir_path}/"]

        def _walk(d: Path, prefix: str, depth: int, count: List[int]):
            if count[0] >= 500:
                lines.append(f"{prefix}... [too many entries]")
                return
            entries = sorted(d.iterdir(),
                             key=lambda p: (not p.is_dir(), p.name.lower()))
            for i, item in enumerate(entries):
                if count[0] >= 500:
                    break
                count[0] += 1
                is_last = (i == len(entries) - 1)
                branch = "└── " if is_last else "├── "
                child_prefix = prefix + ("    " if is_last else "│   ")
                if item.is_dir():
                    lines.append(f"{prefix}{branch}{item.name}/")
                    if depth < 4:
                        _walk(item, child_prefix, depth + 1, count)
                else:
                    try:
                        size = item.stat().st_size
                    except OSError:
                        size = 0
                    lines.append(f"{prefix}{branch}{item.name} ({size} bytes)")

        _walk(dir_path, "", 0, [0])
        return "\n".join(lines)
    except Exception as exc:
        return f"Error listing files: {exc}"


def _run_shell(ctx: ToolContext, command: str, timeout: int = 30) -> str:
    """Run a shell command, capturing output. Returns text for the LLM."""
    if not command.strip():
        return "Error: empty command"
    # Safety: destructive or sudo commands need explicit y/N approval.
    if not _confirm_dangerous_command(ctx, command):
        return ("[command blocked: user declined approval. "
                "Abort or rephrase without destructive/sudo operations.]")
    timeout = max(1, min(int(timeout), 600))
    try:
        proc = subprocess.Popen(
            command, shell=True, executable="/bin/bash",
            cwd=str(ctx.project_root),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception as exc:
        return f"Error starting command: {exc}"
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        timed_out = True
    result = BashToolResult(
        stdout=stdout or "",
        stderr=stderr or "",
        exit_code=proc.returncode if proc.returncode is not None else -1,
        timed_out=timed_out,
    )
    return _truncate(result.format(), ctx.config.max_output_chars)


def _run_interactive(ctx: ToolContext, command: str) -> str:
    """Run an interactive program using the user's own TTY."""
    if not command.strip():
        return "Error: empty command"
    try:
        ctx.app.log(f"[starting interactive: {command}]")
        # Run without pipes so the child gets the real terminal.
        subprocess.call(command, shell=True, executable="/bin/bash",
                        cwd=str(ctx.project_root))
        return f"[interactive command finished: {command}]"
    except Exception as exc:
        return f"Error running interactive command: {exc}"


def register_core_tools(app: "WheeljackApp", config: Config,
                        project_root: Path) -> None:
    """Register blaster's basic six tools (read-only ones parallel_safe=True).
    Any extended tool (web_fetch, grep, ...) is a plugin that adds itself."""
    ctx = ToolContext(app, config, project_root)

    def wrap(handler: Callable) -> Callable:
        return lambda args: handler(ctx, args)

    app.add_tool("read_file", {
        "description": "Read a file's content. Use when the user asks about "
                       "a file, config, log, or file content.",
        "parameters": {"path": ("string", "File path, absolute or relative to cwd")},
        "required": ["path"],
        "parallel_safe": True,
    }, wrap(_h_read_file))

    app.add_tool("write_file", {
        "description": "Create a new file or overwrite an existing one. "
                       "Blocked unless edit mode is enabled (-w/--write); in an "
                       "interactive terminal a single write may be approved on "
                       "the spot.",
        "parameters": {"path": ("string", "File path, absolute or relative to cwd"),
                       "content": ("string", "Full new content of the file")},
        "required": ["path", "content"],
        "parallel_safe": False,
    }, wrap(_h_write_file))

    app.add_tool("edit_file", {
        "description": "Edit an existing file by replacing a unique substring "
                       "(before) with new text (after). A .bak backup is made. "
                       "Blocked unless edit mode is enabled (-w/--write); in an "
                       "interactive terminal a single edit may be approved on "
                       "the spot.",
        "parameters": {"path": ("string", "File path, absolute or relative to cwd"),
                       "before": ("string", "Exact existing text to find (must be unique)"),
                       "after": ("string", "Replacement text")},
        "required": ["path", "before", "after"],
        "parallel_safe": False,
    }, wrap(_h_edit_file))

    app.add_tool("list_files", {
        "description": "List files and directories under a path "
                       "(recursive=true for a tree).",
        "parameters": {"path": ("string", "Directory to list, default '.'"),
                       "recursive": ("boolean", "List recursively as a tree")},
        "required": [],
        "parallel_safe": True,
    }, wrap(_h_list_files))

    app.add_tool("run_shell", {
        "description": "Run a shell command (bash -c) on this machine, "
                       "returning stdout/stderr and exit code. Use for server "
                       "ops: service status, logs, package installs, process "
                       "checks, systemctl, docker, disk/network, etc. "
                       "Destructive or sudo commands require the user's y/N "
                       "approval.",
        "parameters": {"command": ("string", "The shell command to run"),
                       "timeout": ("integer", "Seconds to wait before killing (default 30)")},
        "required": ["command"],
        "parallel_safe": False,
    }, wrap(_h_run_shell))

    app.add_tool("run_interactive", {
        "description": "Run an interactive terminal program (editor, pager, "
                       "dialog, top). Output is not captured; the user operates "
                       "it directly.",
        "parameters": {"command": ("string", "Command to run interactively")},
        "required": ["command"],
        "parallel_safe": False,
    }, wrap(_h_run_interactive))


def _h_read_file(ctx: ToolContext, args: dict) -> str:
    return _read_file(ctx, args.get("path", ""))


def _h_write_file(ctx: ToolContext, args: dict) -> str:
    return _write_file(ctx, args.get("path", ""), args.get("content", ""))


def _h_edit_file(ctx: ToolContext, args: dict) -> str:
    return _edit_file(ctx, args.get("path", ""), args.get("before", ""),
                      args.get("after", ""))


def _h_list_files(ctx: ToolContext, args: dict) -> str:
    return _list_files(ctx, args.get("path", "."), bool(args.get("recursive", False)))


def _h_run_shell(ctx: ToolContext, args: dict) -> str:
    return _run_shell(ctx, args.get("command", ""), int(args.get("timeout", 30)))


def _h_run_interactive(ctx: ToolContext, args: dict) -> str:
    return _run_interactive(ctx, args.get("command", ""))


# --------------------------------------------------------------------------
# The app: registry + event bus + renderer selection + input routing.
# --------------------------------------------------------------------------

class WheeljackApp:
    """Surface a plugin's register(app) can call into."""

    API_VERSION = PLUGIN_API_VERSION

    def __init__(self, config=None) -> None:
        self.core = _core_module()
        self.config = config if config is not None else Config()
        self.agent = None
        self.tools: Dict[str, tuple] = {}
        self.slash_commands: Dict[str, Callable] = {}
        self.observers: Dict[type, List[Callable]] = {}
        self.plugin_names: List[str] = []
        self.steer_queue: "queue.Queue[str]" = queue.Queue()
        self.cancel = threading.Event()
        self.turn_active = False
        self.quit_requested = False
        self._renderer: Optional[BaseRenderer] = None
        self._renderer_factory: Optional[Callable] = None
        self._log: List[str] = []
        self.reasoning_buffer: List[str] = []
        self._register_builtins()

    # -- renderer --------------------------------------------------------
    def set_renderer(self, factory: Callable) -> None:
        """Register a factory(app) -> BaseRenderer. Returning None means 'not
        applicable' (e.g. no TTY), and core falls back to stdio."""
        self._renderer_factory = factory

    @property
    def renderer(self) -> Optional[BaseRenderer]:
        return self._renderer

    def make_renderer(self, prefer_tui: bool = True) -> BaseRenderer:
        if prefer_tui and self._renderer_factory is not None:
            try:
                r = self._renderer_factory(self)
            except Exception as exc:
                self._log.append(f"renderer plugin failed ({exc}); using stdio")
                r = None
            if r is not None:
                self._renderer = r
                return r
        self._renderer = StdioRenderer(self)
        return self._renderer

    # -- events ----------------------------------------------------------
    def emit(self, event) -> None:
        r = self._renderer
        if r is not None:
            name = _EVENT_HANDLERS.get(type(event))
            if name:
                try:
                    getattr(r, name)(event)
                except Exception as exc:
                    self.log(f"renderer error: {exc}")
        for handler in list(self.observers.get(type(event), ())):
            try:
                handler(event)
            except Exception as exc:
                self.log(f"observer error: {exc}")

    def on(self, event_type: type, handler: Callable) -> None:
        self.observers.setdefault(event_type, []).append(handler)

    def off(self, event_type: type, handler: Callable) -> None:
        handlers = self.observers.get(event_type)
        if handlers and handler in handlers:
            handlers.remove(handler)

    # -- tools / slash ---------------------------------------------------
    def add_tool(self, name: str, spec: dict, handler: Callable) -> None:
        self.tools[name] = (spec, handler)

    def add_slash_command(self, name: str, handler: Callable) -> None:
        self.slash_commands[name] = handler

    def _register_builtins(self) -> None:
        self.add_slash_command("help", _cmd_help)
        self.add_slash_command("quit", _cmd_quit)
        self.add_slash_command("exit", _cmd_quit)
        self.add_slash_command("tools", _cmd_tools)
        self.add_slash_command("plugins", _cmd_plugins)
        self.add_slash_command("model", _cmd_model)
        self.add_slash_command("context", _cmd_context)
        self.add_slash_command("sessions", _cmd_sessions)
        self.add_slash_command("reasoning", _cmd_reasoning)

    # -- input / steering ------------------------------------------------
    def submit_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        if line.startswith("/") and dispatch_slash(self, line):
            return
        # A submitted prompt implies a turn is (or is about to be) running, so
        # the input thread switches to the steer reader immediately.
        self.turn_active = True
        self.steer_queue.put(line)
        self.emit(SteerQueued(self.steer_queue.qsize()))

    def drain_steers(self) -> List[str]:
        out: List[str] = []
        while True:
            try:
                out.append(self.steer_queue.get_nowait())
            except queue.Empty:
                return out

    def request_cancel(self) -> None:
        self.cancel.set()
        if self._renderer is not None:
            self._renderer.cancel_all()

    def confirm(self, prompt: str, **kw) -> str:
        if self._renderer is None:
            return kw.get("default", "n")
        return self._renderer.confirm(prompt, **kw)

    # -- logging ---------------------------------------------------------
    def log(self, msg: str) -> None:
        if self._renderer is None:
            self._log.append(msg)
            return
        self.emit(LogMessage(msg))

    def flush_logs(self) -> None:
        if self._renderer is None:
            return
        pending, self._log = self._log, []
        for m in pending:
            self.emit(LogMessage(m))


# --------------------------------------------------------------------------
# LLM streaming client — emits StreamDelta / ReasoningDelta events instead of
# printing; the renderer decides how to draw them.
# --------------------------------------------------------------------------

class LLMClient:
    def __init__(self, config: Config, emit: Callable,
                 cancel: Optional[threading.Event] = None,
                 tools_schema: Optional[Callable[[], List[Dict]]] = None):
        self.config = config
        self.emit_fn = emit
        self.cancel = cancel
        self.tools_schema_fn = tools_schema or (lambda: [])
        self.content_streamed = False

    # -- public ----------------------------------------------------------
    def call(self, messages: List[Message], label: str = "Thinking") -> LLMResponse:
        """Call an OpenAI-compatible chat completions endpoint, streaming.

        Streams: true keeps long generations alive; tokens flow as
        StreamDelta / ReasoningDelta events. Falls back to a single-JSON parse
        when the server ignores stream:true. Returns a reassembled LLMResponse.
        """
        history = self._trim_messages(messages)
        need_tools = self._conversation_needs_tools(history)
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [self._message_to_dict(m) for m in history],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }
        if need_tools:
            payload["tools"] = self.tools_schema_fn()
            payload["tool_choice"] = "auto"

        url = f"{self.config.api_base.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        self.content_streamed = False
        try:
            with urllib.request.urlopen(req, timeout=self.config.request_timeout) as response:
                data = self._read_stream(response)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", "replace") if exc.fp else str(exc)
            raise Exception(f"API Error {exc.code}: {error_body}") from exc
        except Exception as exc:
            raise Exception(f"Request failed: {exc}") from exc

        if data.get("error"):
            raise Exception(f"API error: {data['error']}")
        if not data.get("choices"):
            raise Exception(f"Empty/unexpected API response: {str(data)[:500]}")
        return LLMResponse(
            choices=data["choices"],
            usage=data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0}),
            streamed=bool(self.content_streamed))

    # -- stream reading --------------------------------------------------
    def _read_stream(self, response) -> Dict:
        """Reassemble an SSE/NDJSON stream, emitting content/reasoning events.

        Detects reasoning tokens (delta.reasoning / reasoning_content /
        thinking) and content (delta.content). tool_calls are accumulated
        silently. Returns (data-like dict). Honors cancel between chunks.
        """
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls_acc: Dict[int, Dict[str, Any]] = {}
        finish_reason = None
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        saw_data = False
        raw_buf: List[bytes] = []
        content_streamed = False
        cancelled = False

        for raw_line in response:
            if self.cancel is not None and self.cancel.is_set():
                # Cancelled mid-stream: keep whatever the model produced so far
                # rather than raising, so the caller can persist it.
                cancelled = True
                break
            raw_buf.append(raw_line)
            line = raw_line.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            saw_data = True
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if not chunk.get("choices"):
                if chunk.get("usage"):
                    usage = chunk["usage"]
                continue
            choice = chunk["choices"][0]
            delta = choice.get("delta", {}) or {}

            rtext = (delta.get("reasoning") or delta.get("reasoning_content")
                     or delta.get("thinking"))
            if rtext:
                reasoning_parts.append(rtext)
                self.emit_fn(ReasoningDelta(rtext))
                continue

            ctext = delta.get("content")
            if ctext:
                content_parts.append(ctext)
                self.emit_fn(StreamDelta(ctext))
                content_streamed = True

            for tc in delta.get("tool_calls", []) or []:
                idx = tc.get("index", 0)
                acc = tool_calls_acc.setdefault(
                    idx, {"id": "", "type": "function",
                          "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    acc["id"] = tc["id"]
                fn = tc.get("function", {}) or {}
                if fn.get("name"):
                    acc["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    acc["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

        self.content_streamed = content_streamed

        # Non-streaming fallback: server ignored stream:true (single JSON body).
        # Never taken when the turn was cancelled — partial output from the
        # stream is what the caller wants to keep.
        if not saw_data and not cancelled:
            text = "".join(r.decode("utf-8", "replace") for r in raw_buf)
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = self._parse_streaming_ndjson(text)
            return data

        # Edge case: the whole answer routed into reasoning with empty content
        # (observed on Ollama /v1 with qwen3). Surface the reasoning as answer.
        if not content_parts and reasoning_parts:
            content_parts = ["".join(reasoning_parts)]

        message: Dict[str, Any] = {}
        if content_parts:
            message["content"] = "".join(content_parts)
        elif cancelled:
            message["content"] = "(cancelled)"
        if tool_calls_acc:
            message["tool_calls"] = [tool_calls_acc[k] for k in sorted(tool_calls_acc)]
        return {"choices": [{"message": message, "finish_reason": finish_reason}],
                "usage": usage}

    def _parse_streaming_ndjson(self, text: str) -> Dict:
        """Reassemble an OpenAI-style response from NDJSON stream chunks."""
        content_parts: List[str] = []
        tool_calls_acc: Dict[int, Dict[str, Any]] = {}
        finish_reason = None
        usage = {"prompt_tokens": 0, "completion_tokens": 0}

        for line in text.splitlines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if not chunk.get("choices"):
                if chunk.get("usage"):
                    usage = chunk.get("usage")
                continue
            delta = chunk["choices"][0].get("delta", {}) or {}
            if delta.get("content"):
                content_parts.append(delta["content"])
            for tc in delta.get("tool_calls", []) or []:
                idx = tc.get("index", 0)
                acc = tool_calls_acc.setdefault(
                    idx, {"id": "", "type": "function",
                          "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    acc["id"] = tc["id"]
                fn = tc.get("function", {}) or {}
                if fn.get("name"):
                    acc["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    acc["function"]["arguments"] += fn["arguments"]
            if chunk["choices"][0].get("finish_reason"):
                finish_reason = chunk["choices"][0]["finish_reason"]

        message: Dict[str, Any] = {}
        if content_parts:
            message["content"] = "".join(content_parts)
        if tool_calls_acc:
            message["tool_calls"] = [
                tool_calls_acc[k] for k in sorted(tool_calls_acc)]
        return {
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage,
        }

    # -- context / payload helpers ---------------------------------------
    def _trim_messages(self, messages: List[Message]) -> List[Message]:
        """Drop the oldest messages once the conversation grows too long.

        Keeps the system prompt, the most recent N messages, and never splits
        an assistant tool_call block away from the tool results that follow it.
        """
        MAX_HISTORY = 100
        if len(messages) <= MAX_HISTORY:
            return messages
        keep = list(messages[-MAX_HISTORY:])
        # If the first kept message is a tool result, its assistant tool_call
        # predecessor was dropped; walk forward until we find a non-tool
        # message so the block is well-formed.
        while keep and keep[0].role == "tool":
            keep.pop(0)
        # Ensure the system prompt is always present.
        if keep and keep[0].role != "system":
            keep.insert(0, messages[0])
        return keep

    def _conversation_needs_tools(self, history: List[Message]) -> bool:
        """True if the request should advertise the tool schema."""
        if len(history) >= 2:
            return True
        return any("Available Tools" in (m.content or "") for m in history[:1])

    def _message_to_dict(self, message: Message) -> Dict:
        result: Dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            result["tool_calls"] = [{"id": tc.id, "type": tc.type, "function": tc.function}
                                    for tc in message.tool_calls]
        if message.tool_call_id:
            result["tool_call_id"] = message.tool_call_id
        if message.name:
            result["name"] = message.name
        return result


# --------------------------------------------------------------------------
# The real Agent loop + tool executor.
# --------------------------------------------------------------------------

def _tool_call_args(tool_call: ToolCall) -> dict:
    raw = tool_call.function.get("arguments", "")
    try:
        return json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {"raw": raw}


def _tool_result_ok(result: str) -> bool:
    return not (result.startswith("Error") or result.startswith("[edit blocked")
                or result.startswith("[command blocked") or result.startswith("[edit declined"))


class Agent(threading.Thread):
    """Real agent loop: wait for a steer when idle, append it as a user
    message, run a turn (LLM stream + bounded/parallel tool execution), save."""

    def __init__(self, app: "WheeljackApp", config: Config, project_root: Path,
                 session_name: Optional[str] = None):
        super().__init__(daemon=True)
        self.app = app
        self.config = config
        self.project_root = Path(project_root).expanduser().resolve()
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.messages: List[Message] = []
        self._declined_commands: List[str] = []
        self.session_context: Optional[SessionContext] = None
        self.conversation_summary = ""
        self.total_tokens_used = 0
        self.last_answer = ""
        self.last_streamed = False
        self.reasoning_text = ""
        self.llm = LLMClient(
            config, app.emit, app.cancel, lambda: _tools_schema(app))

        # Resolve AGENTS.md once; injected into the system prompt.
        self.agents_md_path: Optional[Path] = None
        if config.agents_md_enabled:
            if config.agents_md_file:
                candidate = Path(config.agents_md_file).expanduser()
                if candidate.is_file():
                    self.agents_md_path = candidate
                else:
                    app.log(f"AGENTS.md file not found: {candidate}")
            else:
                self.agents_md_path = self._find_agents_md()

        if not config.session_enabled:
            self.session_context = None
            app.log("Sessions disabled (-s/--no-session); conversation will not be saved.")
        else:
            config.sessions_dir.mkdir(parents=True, exist_ok=True)
            if session_name:
                self._load_or_create_session(session_name)
            else:
                self._create_new_session(self._generate_session_name())

        self._initialize_system_prompt()
        self.app.agent = self
        app.on(ReasoningDelta, self._accumulate_reasoning)

    # -- thread entry point ----------------------------------------------
    def run(self) -> None:
        app = self.app
        try:
            while not app.quit_requested:
                try:
                    line = app.steer_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if app.quit_requested:
                    break
                if not line.strip():
                    continue
                self._add_user_message(line)
                app.turn_active = True
                try:
                    self.run_turn()
                except Exception as exc:
                    app.log(f"error: {exc}")
                finally:
                    app.turn_active = False
                    self._save_session()
        finally:
            app.off(ReasoningDelta, self._accumulate_reasoning)
            if self.session_context:
                self._save_session()

    def run_once(self, prompt: str) -> str:
        """Non-interactive mode: run a single prompt and return the answer."""
        app = self.app
        self._add_user_message(prompt)
        app.turn_active = True
        try:
            self.run_turn()
        finally:
            app.turn_active = False
            app.off(ReasoningDelta, self._accumulate_reasoning)
            self._save_session()
        return self.last_answer

    def close(self) -> None:
        """Detach this agent's observers from the app.

        run_once() detaches when it returns; a long-lived agent (run()) does it
        on thread exit. Call this for an agent that was never started, so a
        later agent on the same app is not fed this one's events.
        """
        self.app.off(ReasoningDelta, self._accumulate_reasoning)
        if self.app.agent is self:
            self.app.agent = None

    # -- one turn --------------------------------------------------------
    def run_turn(self) -> None:
        app = self.app
        self.reasoning_text = ""
        app.emit(TurnStarted())
        try:
            response = self.llm.call(self.messages)
            tool_rounds = 0
            while True:
                if app.cancel.is_set():
                    break
                tool_calls = self._parse_tool_calls(response)
                if not tool_calls:
                    break
                tool_rounds += 1
                if tool_rounds > self.config.max_iterations:
                    app.log(f"Tool round limit reached ({self.config.max_iterations}) "
                            "- stopping to avoid a loop.")
                    self.messages.append(Message(
                        role="user",
                        content=f"(system) Tool round limit reached "
                                f"({self.config.max_iterations}). Stop calling "
                                f"tools and answer now.",
                        timestamp=datetime.now().isoformat()))
                    response = self.llm.call(self.messages)
                    break
                self.messages.append(Message(
                    role="assistant", content="",
                    tool_calls=[tc for tc in tool_calls],
                    timestamp=datetime.now().isoformat()))
                results = self.execute_tools(tool_calls)
                for tool_call, result in zip(tool_calls, results):
                    self._add_tool_response(tool_call, result)
                # Steers typed mid-round reach the next LLM call.
                for steer in app.drain_steers():
                    self._add_user_message(steer)
                if app.cancel.is_set():
                    break
                response = self.llm.call(self.messages, "Getting next response")

            choice = response.choices[0] if response.choices else {}
            content = (choice.get("message") or {}).get("content") or ""
            if not content:
                content = "(no response)"
            self.last_streamed = response.streamed
            self.last_answer = content
            self.messages.append(Message(
                role="assistant", content=content,
                timestamp=datetime.now().isoformat()))
            usage = response.usage
            self.total_tokens_used += (usage.get("prompt_tokens", 0)
                                       + usage.get("completion_tokens", 0))
            # Refresh the recap now that the exchange is complete, so a resumed
            # session shows what it was about even if the newest turn is trimmed.
            self.conversation_summary = self._generate_conversation_summary()
        finally:
            # Drain once more before TurnEnded so a steer arriving in the
            # final round is not dropped.
            for steer in app.drain_steers():
                self._add_user_message(steer)
            app.emit(TurnEnded())

    # -- tool executor ---------------------------------------------------
    def execute_tools(self, tool_calls: List[ToolCall]) -> List[str]:
        """Partition tools by parallel_safe; read-only tools run in a bounded
        pool, mutating tools serialize. Results are returned in call order."""
        app = self.app
        parallel: List[ToolCall] = []
        serial: List[ToolCall] = []
        for tc in tool_calls:
            entry = app.tools.get(tc.function["name"])
            safe = bool(entry and entry[0].get("parallel_safe"))
            (parallel if safe else serial).append(tc)

        for tc in tool_calls:
            app.emit(ToolCallStarted(tc.id, tc.function["name"],
                                     _tool_call_args(tc)))

        results: Dict[str, str] = {}
        if parallel and self.config.tool_parallel_reads:
            workers = min(len(parallel), self.config.parallel_workers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(self._execute_tool, tc) for tc in parallel]
                for tc, fut in zip(parallel, futures):
                    res = fut.result()
                    results[tc.id] = res
                    app.emit(ToolCallFinished(tc.id, tc.function["name"],
                                              res, _tool_result_ok(res)))
        else:
            for tc in parallel:
                res = self._execute_tool(tc)
                results[tc.id] = res
                app.emit(ToolCallFinished(tc.id, tc.function["name"],
                                          res, _tool_result_ok(res)))

        for tc in serial:
            if app.cancel.is_set():
                res = "[cancelled: turn cancelled before this ran]"
            else:
                res = self._execute_tool(tc)
            results[tc.id] = res
            app.emit(ToolCallFinished(tc.id, tc.function["name"],
                                      res, _tool_result_ok(res)))

        return [results[tc.id] for tc in tool_calls]

    def _execute_tool(self, tool_call: ToolCall) -> str:
        name = tool_call.function["name"]
        args_raw = tool_call.function.get("arguments", "")
        try:
            args = json.loads(args_raw) if args_raw.strip() else {}
        except json.JSONDecodeError:
            return f"Error: Invalid tool arguments: {args_raw}"
        entry = self.app.tools.get(name)
        if entry is None:
            return f"Error: Unknown tool {name}"
        try:
            return entry[1](args)
        except Exception as exc:
            return f"Error executing {name}: {exc}"

    def _parse_tool_calls(self, response: LLMResponse) -> List[ToolCall]:
        tool_calls = []
        if response.choices:
            choice = response.choices[0]
            # Don't require finish_reason == "tool_calls": some servers send the
            # calls with "stop" or omit the field, and dropping them silently
            # would turn a tool round into a final answer.
            raw_calls = choice.get("message", {}).get("tool_calls", [])
            for tc in raw_calls or []:
                if isinstance(tc, dict) and "id" in tc and "function" in tc:
                    tool_calls.append(ToolCall(id=tc["id"], function=tc["function"]))
        return tool_calls

    def _add_tool_response(self, tool_call: ToolCall, result: str) -> None:
        self.messages.append(Message(
            role="tool", content=result, tool_call_id=tool_call.id,
            name=tool_call.function["name"], timestamp=datetime.now().isoformat()))

    def _add_user_message(self, text: str) -> None:
        self.messages.append(Message(role="user", content=text,
                                     timestamp=datetime.now().isoformat()))

    def _accumulate_reasoning(self, event: ReasoningDelta) -> None:
        if getattr(self.app, "agent", None) is self:
            self.reasoning_text += event.text

    # -- sessions --------------------------------------------------------
    @staticmethod
    def _generate_session_name() -> str:
        adjectives = [
            "amber", "brave", "clever", "cosmic", "crimson", "daring", "echo",
            "electric", "fierce", "golden", "gentle", "hushed", "jade", "lively",
            "lunar", "mighty", "nimble", "noble", "oceanic", "quiet", "radiant",
            "restless", "silver", "silent", "steel", "swift", "tender", "vivid",
            "wandering", "winter",
        ]
        nouns = [
            "badger", "beacon", "bear", "beetle", "breeze", "cascade", "comet",
            "condor", "coyote", "crane", "dolphin", "eagle", "falcon", "fox",
            "gazelle", "gecko", "glacier", "heron", "horizon", "jaguar", "lantern",
            "lynx", "maple", "meadow", "otter", "pine", "raven", "river", "sable",
            "salamander", "sojourn", "sparrow", "summit", "thunder", "voyage",
            "willow", "wren", "zephyr",
        ]
        return f"{random.choice(adjectives)}-{random.choice(nouns)}"

    @staticmethod
    def _generate_session_id() -> str:
        return uuid.uuid4().hex[:12]

    def _load_or_create_session(self, name: str) -> None:
        target = self._find_session_by_name(name)
        if target is None:
            self._create_new_session(name)
            return
        session_file, session_data = target
        context_data = session_data.get("context", {})
        self.session_context = SessionContext(
            session_id=context_data.get("session_id", session_file.stem),
            name=context_data.get("name", name),
            created_at=context_data.get("created_at", datetime.now().isoformat()),
            last_accessed=datetime.now().isoformat(),
            message_count=context_data.get("message_count", 0),
        )
        self.conversation_summary = session_data.get("conversation_summary", "")
        self.total_tokens_used = session_data.get("total_tokens", 0)
        saved_messages = session_data.get("messages", [])
        if saved_messages:
            for msg_data in saved_messages[-100:]:
                role = msg_data.get("role", "user")
                content = msg_data.get("content", "")
                if role == "system":
                    continue  # a fresh system prompt is added when the agent starts
                raw_calls = msg_data.get("tool_calls") or []
                tool_calls = [
                    ToolCall(id=c.get("id", ""), type=c.get("type", "function"),
                             function=c.get("function", {}))
                    for c in raw_calls if isinstance(c, dict) and c.get("function")
                ]
                if role == "tool":
                    self.messages.append(Message(
                        role="tool", content=content,
                        tool_call_id=msg_data.get("tool_call_id", "legacy"),
                        name=msg_data.get("name", "tool"),
                        timestamp=msg_data.get("timestamp")))
                else:
                    self.messages.append(Message(
                        role=role, content=content, tool_calls=tool_calls or None,
                        timestamp=msg_data.get("timestamp")))
        self.app.log(f"Loaded session '{self.session_context.name}' "
                     f"({len(saved_messages)} previous messages, saved on quit)")
        self._show_past_messages()

    def _find_session_by_name(self, name: str):
        for session_file in sorted(self.config.sessions_dir.glob("*.json"),
                                   key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(session_file.read_text(encoding="utf-8"))
                ctx = data.get("context", {})
                saved_name = ctx.get("name", ctx.get("session_id", session_file.stem))
                if saved_name == name:
                    return session_file, data
            except Exception:
                continue
        return None

    def _create_new_session(self, name: Optional[str] = None) -> None:
        if not name:
            name = self._generate_session_name()
        self.session_context = SessionContext(
            session_id=self._generate_session_id(),
            name=name,
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            message_count=0,
        )
        self.app.log(f"Created new session '{name}' (will be saved on quit)")

    def _save_session(self) -> None:
        if not self.session_context:
            return
        sc = self.session_context
        sc.last_accessed = datetime.now().isoformat()
        sc.message_count = len(self.messages)
        session_data = {
            "context": dict(vars(sc), project_root=str(self.project_root)),
            "conversation_summary": self.conversation_summary,
            "total_tokens": self.total_tokens_used,
            "messages": [
                {"role": m.role, "content": m.content,
                 "timestamp": m.timestamp or datetime.now().isoformat(),
                 "tool_call_id": m.tool_call_id, "name": m.name,
                 # Kept so a resumed conversation replays a well-formed
                 # assistant tool_call block; without it the tool results that
                 # follow have no owner and the API rejects the request.
                 "tool_calls": ([{"id": tc.id, "type": tc.type,
                                  "function": tc.function} for tc in m.tool_calls]
                                if m.tool_calls else None)}
                for m in self.messages
            ],
        }
        try:
            path = self.config.sessions_dir / f"{sc.session_id}.json"
            path.write_text(json.dumps(session_data, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        except Exception as exc:
            self.app.log(f"Could not save session: {exc}")

    def _generate_conversation_summary(self) -> str:
        """One-line recap of the most recent user/assistant exchanges.

        Heuristic, not an LLM call (keeps the transport out of it): feeds the
        system prompt's 'Recent:' line and the resume banner, so a long session
        stays legible once `_trim_messages` starts dropping the oldest turns.
        Long messages are truncated rather than skipped, so they still count.
        """
        recent = []
        for msg in self.messages[-6:]:
            if msg.role not in ("user", "assistant"):
                continue
            text = (msg.content or "").strip().replace("\n", " ")
            if not text or text == "(no response)" or text.startswith("(system)"):
                continue
            if len(text) > 100:
                text = text[:100] + "…"
            who = "User" if msg.role == "user" else "Assistant"
            recent.append(f"{who}: {text}")
        return " | ".join(recent[-3:])

    def _show_past_messages(self) -> None:
        past = [m for m in self.messages if m.role in ("user", "assistant")
                and m.content]
        summary = (self.conversation_summary or "").strip()
        if len(past) <= 1 and not summary:
            return  # nothing restored (just the system prompt)
        lines = []
        if summary:
            lines.append("— summary of the previous session —")
            lines.append(f"  {summary}")
        if len(past) > 1:
            n = len(past) - 1  # exclude the (empty) initial context
            lines.append(f"— previous conversation ({n} messages) —")
            for m in past:
                who = "you" if m.role == "user" else "wheeljack"
                text = m.content.strip().replace("\n", " ")
                if len(text) > 300:
                    text = text[:300] + "…"
                lines.append(f"  {who}: {text}")
            lines.append("— end of previous conversation —")
        self.app.log("\n".join(lines))

    # -- AGENTS.md / system prompt ---------------------------------------
    def _find_agents_md(self) -> Optional[Path]:
        """Return the nearest AGENTS.md from the working directory upward."""
        d = self.project_root
        for _ in range(AGENTS_MD_MAX_DEPTH):
            candidate = d / AGENTS_MD_FILENAME
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                pass
            if d.parent == d:
                break
            d = d.parent
        return None

    def _get_agents_md(self) -> str:
        if not self.agents_md_path:
            return ""
        try:
            text = self.agents_md_path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
        if not text:
            return ""
        if len(text) > self.config.max_agents_md_chars:
            dropped = len(text) - self.config.max_agents_md_chars
            text = (text[:self.config.max_agents_md_chars].rstrip()
                    + f"\n\n... [AGENTS.md truncated at "
                      f"{self.config.max_agents_md_chars} characters; "
                      f"{dropped} more omitted]")
        return (f"Project instructions from {self.agents_md_path} (AGENTS.md — "
                f"follow these; they take precedence for this project):\n\n{text}")

    def _initialize_system_prompt(self) -> None:
        project_info = self._get_project_info()
        session_info = self._get_session_info()
        agents_md = self._get_agents_md()
        agents_block = f"{agents_md}\n\n" if agents_md else ""
        tool_names = ", ".join(sorted(self.app.tools)) or "(none)"

        system_prompt = f"""You are Wheeljack, a server operations and coding assistant running directly on this machine. You help set up, maintain, configure, and debug servers and code, and you execute actions via tools.

You operate in the working directory: {self.project_root}
The current date is {datetime.now().strftime('%Y-%m-%d')}.
Edit mode: {("ENABLED (-w/--write)" if self.config.allow_edits else "OFF — safe/read-only mode; write_file/edit_file are blocked unless the user approves")}.

{agents_block}{project_info}

{session_info}

TOOL USE RULES (critical):
- Prefer using tools over merely describing what to do. You can actually change the system.
- FILE EDITS: when edit mode is OFF, write_file and edit_file are blocked. If you need to change a file, ask the user to enable edit mode (or approve the edit) and stop — do NOT retry the edit, and do NOT work around the block with shell redirections (sed -i, echo >, tee, etc.). When edit mode is ON, edits proceed with a .bak backup.
- Before reading or editing a file, run_shell 'ls' / read_file to see what exists. If an edit_file 'before' string is rejected, read the file to get exact content.
- run_shell output is returned to you verbatim (capped). Use exit codes to judge success. For long-running commands pass a generous timeout.
- When a command fails, inspect the error and try to fix it yourself (check logs, configs, dependencies) before giving up.
- Commands flagged as destructive (rm -rf, mkfs, dd, git push -f, etc.) or any command using sudo will prompt the user for y/N approval. If approval is declined, do NOT retry the same command; explain and offer a safer alternative.
- run_interactive starts a program on the user's terminal (editors, top, etc.); the user operates it directly. Prefer run_shell for anything that can run non-interactively.
- Ask before making large or irreversible changes you cannot verify (e.g. deleting data, wiping disks, changing firewall rules). For routine config edits and service restarts, just proceed.
- Keep responses concise. Summarize what you changed and the resulting state.

Available Tools (schemas are sent with each request):
{tool_names}
"""
        # The system prompt must lead the conversation. A resumed session has
        # restored messages already queued, so appending would put it last —
        # which also breaks _trim_messages' "messages[0] is the system prompt"
        # assumption when the window overflows.
        self.messages.insert(0, Message(role="system", content=system_prompt))

    def _get_session_info(self) -> str:
        if not self.session_context:
            return ""
        sc = self.session_context
        lines = ["Session Context:", f"  Session: {sc.name}", f"  ID: {sc.session_id}",
                 f"  Created: {sc.created_at.split('T')[0]}"]
        if self.conversation_summary:
            lines.append(f"  Recent: {self.conversation_summary}")
        return "\n".join(lines)

    def _get_project_info(self) -> str:
        info = []
        try:
            entries = sorted(self.project_root.iterdir(),
                             key=lambda p: (not p.is_dir(), p.name.lower()))
            if entries:
                info.append("Working directory contents (top-level):")
                info += [f"  - {i.name}/" if i.is_dir() else f"  - {i.name}"
                         for i in entries[:self.config.max_context_files]]
            else:
                info.append("Working directory is empty.")
        except Exception as exc:
            info.append(f"Could not read working directory: {exc}")
        if (self.project_root / ".git").exists():
            info.append("\nThis is a git repository.")
        return "\n".join(info)


# --------------------------------------------------------------------------
# DemoAgent — fake tool calls + a destructive one that needs confirm(), used
# behind --demo for endpoint-free TUI/stdio smoke tests.
# --------------------------------------------------------------------------

class DemoAgent(threading.Thread):
    def __init__(self, app: "WheeljackApp"):
        super().__init__(daemon=True)
        self.app = app
        self.last_answer = "Done."
        self.last_streamed = False

    def _wait(self, seconds: float) -> bool:
        """Cancel-aware sleep; returns False if cancelled."""
        end = time.time() + seconds
        while time.time() < end:
            if self.app.cancel.is_set():
                return False
            time.sleep(0.02)
        return True

    def _flush_steers(self) -> None:
        for msg in self.app.drain_steers():
            self.app.emit(StreamDelta(f"(steered) noted: {msg}\n"))

    def run(self) -> None:
        app = self.app
        app.turn_active = True
        app.emit(LogMessage("demo agent running (no LLM endpoint used)"))
        try:
            app.emit(TurnStarted())
            app.emit(ReasoningDelta("looking at the request"))
            app.emit(StreamDelta(
                "Checking disk usage, then a (fake) destructive step.\n"))

            app.emit(ToolCallStarted("1", "run_shell", {"cmd": "df -h"}))
            if self._wait(0.6):
                app.emit(ToolCallFinished("1", "run_shell", "42% used on /", True))
            self._flush_steers()

            if not app.cancel.is_set():
                ok = app.confirm(
                    "Run 'rm -rf /tmp/demo-scratch'?", default="n", timeout=1.0) == "y"
                app.emit(ToolCallStarted(
                    "2", "run_shell", {"cmd": "rm -rf /tmp/demo-scratch"}))
                if self._wait(0.3):
                    app.emit(ToolCallFinished(
                        "2", "run_shell",
                        "done" if ok else "blocked by user", ok))
            self._flush_steers()

            if app.cancel.is_set():
                app.emit(StreamDelta("(cancelled)\n"))
                self.last_answer = "(cancelled)"
            else:
                app.emit(StreamDelta("Done.\n"))
                self.last_answer = "Done."
        finally:
            app.emit(TurnEnded())
            app.turn_active = False


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def _install_sigint(app: "WheeljackApp") -> None:
    """First Ctrl+C cancels the running turn; a second one exits."""
    state = {"count": 0}

    def handler(signum, frame):
        state["count"] += 1
        if state["count"] == 1:
            app.request_cancel()
        else:
            raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, handler)
    except Exception:
        pass


def _render_answer(answer: str, config: Config) -> None:
    """Render a non-streamed final answer (glow markdown when available)."""
    if config.format_markdown and _USE_COLOR and _GLOW_PATH:
        _render_markdown(f"\n{answer}")
    else:
        print(f"\n🤖 {_c(answer, 'green')}")


def _load_config(defaults: Dict[str, Any], home: Path) -> Config:
    """Override Config() with ~/.wheeljack/config.json + the CLI map."""
    config = Config()
    cfg_file = Path(home) / "config.json"
    if cfg_file.is_file():
        try:
            data = json.loads(cfg_file.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        for key in ("model", "api_base", "api_key", "plugin_repo",
                    "max_iterations", "request_timeout"):
            if data.get(key) is not None:
                setattr(config, key, data[key])
    for key, value in defaults.items():
        if value is not None:
            setattr(config, key, value)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wheeljack — server ops & coding agent (OpenAI-compatible LLM)")
    parser.add_argument("--model", default=None, help="model name (default: env or qwen3.8:27b)")
    parser.add_argument("--api-base", dest="api_base_cli", default=None,
                        help="OpenAI-compatible API base, e.g. http://localhost:11434/v1")
    parser.add_argument("--session", nargs="?", const="__list__", default=None,
                        help="session name to load; bare --session lists saved sessions")
    parser.add_argument("--cwd", default=None, help="working directory for tools")
    parser.add_argument("-p", "--prompt", default=None,
                        help="non-interactive mode: run a single prompt and exit")
    parser.add_argument("-n", "--max-iteration", dest="max_iterations", type=int, default=None,
                        help="max tool rounds per turn (default: 50)")
    parser.add_argument("-x", "--no-format", dest="format_markdown", action="store_false",
                        help="disable glow markdown formatting of non-streamed answers")
    parser.add_argument("-s", "--no-session", dest="session_enabled", action="store_false",
                        help="disable sessions: do not load, create, or save any session")
    parser.add_argument("-w", "--write", dest="allow_edits", action="store_true",
                        help="enable edit mode: allow write_file/edit_file to modify files")
    parser.add_argument("-t", "--timeout", dest="request_timeout", type=int, default=None,
                        help="LLM request timeout in seconds (default: 300)")
    parser.add_argument("--agents-md", dest="agents_md", metavar="PATH", default=None,
                        help="AGENTS.md file to load ('none' disables; default: auto-discover)")
    parser.add_argument("--no-tui", action="store_true", help="force the stdio renderer")
    parser.add_argument("--plugins", default="./.wheeljack/plugins",
                        help="project plugin directory (default: ./.wheeljack/plugins)")
    parser.add_argument("--no-installed-plugins", action="store_true",
                        help="do not load ~/.wheeljack/plugins")
    parser.add_argument("--demo", action="store_true",
                        help="fake agent (no LLM endpoint) for smoke tests")
    parser.add_argument("--show-reasoning", action="store_true",
                        help="stream reasoning inline instead of a collapsed counter")
    parser.add_argument("--no-parallel", dest="no_parallel", action="store_true",
                        help="serialize read-only tools too")
    parser.add_argument("--install", metavar="NAME|URL", default=None,
                        help="install a plugin from the repo (checksum + confirm)")
    parser.add_argument("--uninstall", metavar="NAME", default=None,
                        help="remove an installed plugin")
    parser.add_argument("--list-plugins", action="store_true",
                        help="list installed, project, and available plugins")
    parser.add_argument("--plugin-repo", dest="plugin_repo", default=None,
                        help="plugin repo base URL (default: env or raw.githubusercontent.com/k4ml/blaster/main)")
    parser.add_argument("--force", action="store_true", help="overwrite an installed plugin")
    parser.add_argument("--sha256", dest="sha256_hex", default=None,
                        help="expected sha256 (used when no sidecar is present)")
    parser.add_argument("--insecure", action="store_true",
                        help="install without a checksum (loud warn)")
    args = parser.parse_args()

    # Let plugins `import wheeljack` even when we were started as a script.
    sys.modules.setdefault("wheeljack", sys.modules[__name__])

    home = ensure_home()

    # -- pre-agent shell commands (no renderer exists yet) ----------------
    repo = args.plugin_repo or os.getenv("WHEELJACK_PLUGIN_REPO") or DEFAULT_PLUGIN_REPO
    if args.session == "__list__":
        for line in _sessions_table(home / "sessions"):
            print(line)
        return
    if args.list_plugins:
        for line in list_plugins(Path(args.plugins), home=home, repo_base=repo):
            print(line)
        return
    if args.install:
        try:
            target = install_plugin(args.install, repo_base=repo,
                                    target_dir=home / "plugins",
                                    force=args.force, sha256_hex=args.sha256_hex,
                                    insecure=args.insecure)
        except InstallError as exc:
            print(f"error: {exc}")
            sys.exit(1)
        if target:
            print(f"installed: {target}")
            sys.exit(0)
        sys.exit(1)
    if args.uninstall:
        try:
            ok = uninstall_plugin(args.uninstall, target_dir=home / "plugins")
        except InstallError as exc:
            print(f"error: {exc}")
            sys.exit(1)
        if ok:
            print(f"uninstalled: {args.uninstall}")
        sys.exit(0 if ok else 1)

    # -- config: CLI flag > config.json > env > defaults -------------------
    if args.agents_md is not None and args.agents_md.strip().lower() == "none":
        agents_md_enabled = False
        agents_md_file = None
    elif args.agents_md is not None:
        agents_md_enabled = True
        agents_md_file = args.agents_md
    else:
        agents_md_enabled = None  # leave the dataclass default alone
        agents_md_file = None

    config = _load_config({
        "model": args.model,
        "api_base": args.api_base_cli,
        "max_iterations": args.max_iterations,
        "request_timeout": args.request_timeout,
        "format_markdown": args.format_markdown,
        "session_enabled": args.session_enabled,
        "allow_edits": args.allow_edits,
        "non_interactive": args.prompt is not None,
        "tool_parallel_reads": not args.no_parallel,
        "show_reasoning": args.show_reasoning,
        "plugin_repo": args.plugin_repo,
        "agents_md_enabled": agents_md_enabled,
        "agents_md_file": agents_md_file,
    }, home)
    config.sessions_dir = home / "sessions"

    project_root = Path(args.cwd).expanduser() if args.cwd else Path.cwd()
    project_root = project_root.resolve()

    app = WheeljackApp(config)
    register_core_tools(app, config, project_root)
    load_plugins(app, Path(args.plugins),
                 None if args.no_installed_plugins else home / "plugins")

    # -- non-interactive single shot --------------------------------------
    if args.prompt:
        renderer = NonInteractiveRenderer(app)
        app._renderer = renderer
        if args.demo:
            agent = DemoAgent(app)
            app.agent = agent
            agent.start()
            agent.join()
            return
        agent = Agent(app, config, project_root, session_name=args.session)
        app.agent = agent
        try:
            answer = agent.run_once(args.prompt)
        except Exception as exc:
            print(f"error: {exc}")
            sys.exit(1)
        if answer and answer != "(no response)" and not agent.last_streamed:
            _render_answer(answer, config)
        return

    # -- interactive -------------------------------------------------------
    renderer = app.make_renderer(prefer_tui=not args.no_tui)
    app.flush_logs()
    _install_sigint(app)
    renderer.start(app)

    agent = DemoAgent(app) if args.demo else Agent(
        app, config, project_root, session_name=args.session)
    app.agent = agent
    agent.start()

    if hasattr(renderer, "loop"):
        # The TUI plugin must run on the main thread; stop it once the turn
        # ends (or the user quits).
        def watch() -> None:
            while agent.is_alive() and not app.quit_requested:
                time.sleep(0.05)
            time.sleep(0.3)
            renderer.stop()

        threading.Thread(target=watch, daemon=True).start()
        try:
            renderer.loop(app)
        except KeyboardInterrupt:
            renderer.stop()
        except Exception as exc:
            renderer.stop()
            app._renderer = StdioRenderer(app)
            app.log(f"TUI failed ({exc}); falling back to stdio")
            app.flush_logs()
    else:
        try:
            agent.join()
        except KeyboardInterrupt:
            app.request_cancel()
            agent.join()
        renderer.stop()

    # Save on every exit path, including a SIGINT raised out of join(): the
    # per-turn autosave can miss the final turn, so persist the summary and
    # messages here too.
    if isinstance(agent, Agent):
        agent.conversation_summary = agent._generate_conversation_summary()
        agent._save_session()


if __name__ == "__main__":
    main()