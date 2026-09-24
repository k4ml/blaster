#!/usr/bin/env python3
"""
Wheeljack — the blaster successor (foundation).

This is the hardened architectural core, not the full agent yet. It is:

    - a small vocabulary of Events the agent loop emits instead of print()
    - a Renderer interface (BaseRenderer) with a stdio implementation in core;
      the curses TUI lives in wheeljack_plugins/wheeljack_tui.py and registers
      itself via app.set_renderer()
    - a modal queue so concurrent confirm() calls can never deadlock, with a
      timeout/cancel path
    - a steer_queue that lets the user type ahead while the agent works
    - a plugin loader (plugin_dir/wheeljack_*.py, a register(app) function) and
      a versioned app surface (API_VERSION)
    - a slash-command dispatcher (/help, /quit, /tools, /plugins, /model)
    - a DemoAgent that exercises all of the above with fake "tools"

Real LLM streaming + blaster's actual tool set (read_file/write_file/edit_file/
run_shell/list_files) still need porting in on top of this.

Usage:
    python wheeljack.py            # curses TUI if a plugin provides one, else stdio
    python wheeljack.py --no-tui   # force the plain stdio renderer
    python wheeljack.py -p "..."   # non-interactive: no renderer threads at all
"""
from __future__ import annotations

import argparse
import atexit
import collections
import fcntl
import importlib.util
import os
import queue
import select
import shutil
import signal
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# Events — the vocabulary the agent loop speaks. Core code calls app.emit(event)
# instead of print(); the renderer decides how (or whether) to draw it. All of
# the queueing, locking, and threading lives inside the renderers.
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
#
# Only one thread ever "owns" raw terminal input at a time (the InputThread for
# stdio mode, or the curses loop thread for the TUI). confirm() blocks the
# *calling* thread (usually a worker running tool execution) on an Event, while
# the input-owning thread keeps running and routes each keystroke to either
# "answer the modal" or "append to steering text".
#
# Modals are a QUEUE, not a single slot: if two workers raise a confirm() at
# once (parallel tools), both must be answerable. A single slot silently
# orphaned the first caller's Event and hung it forever.
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
#
# The old skeleton wrote stream deltas straight to sys.stdout while the input
# thread sat in input(), so tokens and the prompt interleaved. Every write now
# goes through here: it erases the in-progress input line before writing output
# and leaves it hidden until the input layer asks for it back (redrawing on
# every token would make the prompt bounce). Piped output never sets _visible,
# so no escape codes are ever emitted off a TTY.
# --------------------------------------------------------------------------

class ConsoleWriter:
    def __init__(self, stream=None) -> None:
        self._lock = threading.RLock()
        self._stream = stream if stream is not None else sys.stdout
        self._prompt = ""
        self._buf = ""
        self._visible = False
        self._line_start = True

    @property
    def is_tty(self) -> bool:
        try:
            return bool(self._stream.isatty())
        except Exception:
            return False

    def emit(self, text: str) -> None:
        with self._lock:
            self._erase()
            self._stream.write(text)
            self._stream.flush()
            self._line_start = text.endswith("\n") or text == ""

    def emit_line(self, text: str = "") -> None:
        self.emit(text + "\n")

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


# --------------------------------------------------------------------------
# EnhancedInput — raw-mode line editor with history (ported from blaster).
# Used for the idle stdio prompt; multiline paste and wrapped redraw included.
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

    def turn_started(self, e: TurnStarted) -> None:
        self.writer.emit_line("")

    def turn_ended(self, e: TurnEnded) -> None:
        pass  # the input layer owns the prompt; never print one here

    def tool_started(self, e: ToolCallStarted) -> None:
        self.writer.emit_line(f"  \U0001F527 {e.name}({e.args}) ...")

    def tool_finished(self, e: ToolCallFinished) -> None:
        mark = "\u2713" if e.ok else "\u2717"
        self.writer.emit_line(f"  {mark} {e.name}: {e.result}")

    def stream_delta(self, e: StreamDelta) -> None:
        self.writer.emit(e.text)

    def reasoning_delta(self, e: ReasoningDelta) -> None:
        pass  # collapsed by default in stdio mode, matches blaster's behavior

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


_BUILTIN_COMMANDS = ("help", "quit", "exit", "tools", "plugins", "model")


def _cmd_help(app: "WheeljackApp", args: str) -> None:
    lines = [
        "Wheeljack commands:",
        "  /help            show this help",
        "  /quit, /exit     exit",
        "  /tools           list registered tools",
        "  /plugins         list loaded plugins",
        "  /model [NAME]    show or set the model",
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


# --------------------------------------------------------------------------
# Plugin loader — a plugin is plugin_dir/wheeljack_*.py exposing register(app).
# The filename convention is deliberate: never glob every sibling .py, or a
# stray blaster.py / tests.py next to core would be executed at startup.
# --------------------------------------------------------------------------

PLUGIN_API_VERSION = 1


def _core_module():
    return sys.modules.get("wheeljack") or sys.modules.get(__name__)


def load_plugins(app: "WheeljackApp", plugin_dir: Path) -> None:
    if not plugin_dir.is_dir():
        return
    for path in sorted(plugin_dir.glob("wheeljack_*.py")):
        if path.name.startswith("_") or path.name == "wheeljack.py":
            continue
        modname = f"wheeljack_plugin_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(modname, path)
            if spec is None or spec.loader is None:
                app.log(f"plugin {path.name} could not be loaded; skipping")
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[modname] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(modname, None)
            app.log(f"plugin {path.name} failed to load ({exc}); skipping")
            continue

        declared = getattr(module, "API_VERSION", PLUGIN_API_VERSION)
        if declared != PLUGIN_API_VERSION:
            app.log(f"plugin {path.name} targets API v{declared}, "
                    f"core is v{PLUGIN_API_VERSION}; loading anyway")

        register = getattr(module, "register", None)
        if not callable(register):
            app.log(f"plugin {path.name} has no register(app); skipping")
            continue
        try:
            register(app)
        except Exception as exc:
            app.log(f"plugin {path.name} register() failed ({exc}); skipping")
            continue
        app.plugin_names.append(path.stem)
        app.log(f"loaded plugin: {path.name}")


# --------------------------------------------------------------------------
# The app: registry + event bus + renderer selection + input routing.
# --------------------------------------------------------------------------

class WheeljackApp:
    """Surface a plugin's register(app) can call into."""

    API_VERSION = PLUGIN_API_VERSION

    def __init__(self, config=None) -> None:
        self.core = _core_module()
        self.config = config if config is not None else SimpleNamespace(
            model="demo-model", api_base="local")
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
        for handler in self.observers.get(type(event), ()):
            try:
                handler(event)
            except Exception as exc:
                self.log(f"observer error: {exc}")

    def on(self, event_type: type, handler: Callable) -> None:
        self.observers.setdefault(event_type, []).append(handler)

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
# DemoAgent — fake tool calls + a destructive one that needs confirm(), on a
# worker thread, checking steer_queue between rounds. Replaced by the real
# LLM/tool loop later.
# --------------------------------------------------------------------------

class DemoAgent(threading.Thread):
    def __init__(self, app: "WheeljackApp"):
        super().__init__(daemon=True)
        self.app = app

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
                    "Run 'rm -rf /tmp/demo-scratch'?", default="n") == "y"
                app.emit(ToolCallStarted(
                    "2", "run_shell", {"cmd": "rm -rf /tmp/demo-scratch"}))
                if self._wait(0.3):
                    app.emit(ToolCallFinished(
                        "2", "run_shell",
                        "done" if ok else "blocked by user", ok))
            self._flush_steers()

            if app.cancel.is_set():
                app.emit(StreamDelta("(cancelled)\n"))
            else:
                app.emit(StreamDelta("Done.\n"))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Wheeljack foundation demo")
    parser.add_argument("--no-tui", action="store_true",
                        help="force the stdio renderer")
    parser.add_argument("-p", "--prompt",
                        help="non-interactive: skip renderer threading entirely")
    parser.add_argument("--plugins", default="./wheeljack_plugins",
                        help="plugin directory (wheeljack_*.py files)")
    parser.add_argument("--model", default=None, help="model name (demo only)")
    args = parser.parse_args()

    # Let plugins `import wheeljack` even when we were started as a script.
    sys.modules.setdefault("wheeljack", sys.modules[__name__])

    config = SimpleNamespace(model=args.model or "demo-model", api_base="local")
    app = WheeljackApp(config)
    load_plugins(app, Path(args.plugins))

    if args.prompt:
        renderer = NonInteractiveRenderer(app)
        app._renderer = renderer
        agent = DemoAgent(app)
        app.agent = agent
        agent.start()
        agent.join()
        return

    renderer = app.make_renderer(prefer_tui=not args.no_tui)
    app.flush_logs()
    _install_sigint(app)
    renderer.start(app)

    agent = DemoAgent(app)
    app.agent = agent
    agent.start()

    if hasattr(renderer, "loop"):
        # The TUI plugin must run on the main thread; stop it once the demo
        # turn ends (or the user quits).
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


if __name__ == "__main__":
    main()
