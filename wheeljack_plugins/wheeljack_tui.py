#!/usr/bin/env python3
"""
Wheeljack TUI plugin — a curses renderer, loaded as a sibling plugin.

Drop this file in the plugin directory (default ./wheeljack_plugins) named
`wheeljack_tui.py`; `register(app)` installs a renderer factory. Core falls
back to the stdio renderer whenever this returns None (no TTY, TERM=dumb, or
curses unavailable), so piping/redirecting still works exactly as before.

This is the "first plugin" the foundation was built to prove: it only needs
the public surface (app.set_renderer, BaseRenderer, app.submit_line,
app.quit_requested) and the shared modal queue — no core changes.

Design notes (vs the old in-core CursesRenderer):
    - a growable scrollback of logical lines that WRAPS, instead of a fixed
      2000-row pad that raised curses.error once it filled
    - redraw only when dirty, not every 50 ms unconditionally
    - keypad(True) + get_wch() so arrows/unicode/KEY_RESIZE work
    - a real line editor in the input box (history, cursor movement)
    - the modal queue lives in BaseRenderer, so concurrent confirms can't hang
"""
from __future__ import annotations

import os
import sys
import threading
from typing import List, Optional

import wheeljack as core  # resolved via the alias core installs in main()


def _wrap(text: str, width: int) -> List[str]:
    """Split a logical line into display rows of at most `width` columns."""
    width = max(1, width)
    if text == "":
        return [""]
    out: List[str] = []
    for logical in text.split("\n"):
        while len(logical) > width:
            out.append(logical[:width])
            logical = logical[width:]
        out.append(logical)
    return out


class TuiRenderer(core.BaseRenderer):
    def __init__(self, app) -> None:
        super().__init__(app)
        import curses  # imported lazily; stdio-only installs never need it
        self.curses = curses
        self._lock = threading.Lock()
        self._lines: List[str] = []
        self._status = ""
        self._input = ""
        self._cursor = 0
        self._history: List[str] = []
        self._hist_i = 0
        self._saved_input = ""
        self._modal: Optional[core.ModalRequest] = None
        self._running = True
        self._dirty = True

    # -- event API: mutate the buffer under lock, the UI thread draws it --
    def _append_line(self, text: str) -> None:
        with self._lock:
            self._lines.append(text)
            self._dirty = True

    def _append_text(self, text: str) -> None:
        with self._lock:
            for i, part in enumerate(text.split("\n")):
                if i == 0:
                    if self._lines:
                        self._lines[-1] += part
                    else:
                        self._lines.append(part)
                else:
                    self._lines.append(part)
            self._dirty = True

    def turn_started(self, e) -> None:
        self._append_line("")

    def turn_ended(self, e) -> None:
        pass

    def tool_started(self, e) -> None:
        self._append_line(f"  [tool] {e.name}({e.args}) ...")

    def tool_finished(self, e) -> None:
        mark = "OK" if e.ok else "FAIL"
        self._append_line(f"  [{mark}] {e.name}: {e.result}")

    def stream_delta(self, e) -> None:
        self._append_text(e.text)

    def reasoning_delta(self, e) -> None:
        with self._lock:
            self._status = "thinking: " + e.text[-40:]
            self._dirty = True

    def steer_queued(self, e) -> None:
        with self._lock:
            self._status = f"{e.pending_count} steer message(s) queued"
            self._dirty = True

    def log_message(self, e) -> None:
        self._append_line(f"[wheeljack] {e.text}")

    def _show_modal(self, req) -> None:
        with self._lock:
            self._modal = req
            self._dirty = True

    def _hide_modal(self) -> None:
        with self._lock:
            self._modal = None
            self._dirty = True

    def stop(self) -> None:
        self._running = False

    # -- the UI thread: drawing + input, all in one place --
    def loop(self, app) -> None:
        self.curses.wrapper(self._run, app)

    def _run(self, stdscr, app) -> None:
        curses = self.curses
        curses.curs_set(1)
        stdscr.keypad(True)
        stdscr.timeout(50)

        while self._running:
            try:
                ch = stdscr.get_wch()
            except curses.error:
                ch = None  # timeout: no key this tick
            except (KeyboardInterrupt, EOFError):
                break

            if ch is not None:
                self._handle_key(stdscr, app, ch)

            if self._dirty:
                self._redraw(stdscr, app)

    def _handle_key(self, stdscr, app, ch) -> None:
        curses = self.curses
        modal = self.current_modal()

        if modal is not None:
            key = ch.lower() if isinstance(ch, str) and ch.isalpha() else ch
            if key in modal.choices:
                self.answer_modal(key)
            elif ch in ("\n", "\r", curses.KEY_ENTER, "\x1b"):
                self.answer_modal(modal.default)
            return

        if ch in ("\n", "\r", curses.KEY_ENTER):
            if self._input.strip():
                self._history.append(self._input)
                self._hist_i = len(self._history)
                app.submit_line(self._input)
            self._input, self._cursor = "", 0
            self._dirty = True
        elif ch in (curses.KEY_BACKSPACE, "\x7f", "\b"):
            if self._cursor > 0:
                self._input = self._input[:self._cursor - 1] + self._input[self._cursor:]
                self._cursor -= 1
                self._dirty = True
        elif ch == curses.KEY_DC:
            if self._cursor < len(self._input):
                self._input = self._input[:self._cursor] + self._input[self._cursor + 1:]
                self._dirty = True
        elif ch == curses.KEY_LEFT:
            self._cursor = max(0, self._cursor - 1)
            self._dirty = True
        elif ch == curses.KEY_RIGHT:
            self._cursor = min(len(self._input), self._cursor + 1)
            self._dirty = True
        elif ch == curses.KEY_UP:
            self._history_up()
        elif ch == curses.KEY_DOWN:
            self._history_down()
        elif ch == curses.KEY_RESIZE:
            self._dirty = True
        elif isinstance(ch, str) and ch.isprintable():
            self._input = self._input[:self._cursor] + ch + self._input[self._cursor:]
            self._cursor += 1
            self._dirty = True

    def _history_up(self) -> None:
        if not self._history:
            return
        if self._hist_i == len(self._history):
            self._saved_input = self._input
        if self._hist_i > 0:
            self._hist_i -= 1
            self._input = self._history[self._hist_i]
            self._cursor = len(self._input)
            self._dirty = True

    def _history_down(self) -> None:
        if self._hist_i < len(self._history):
            self._hist_i += 1
            self._input = (self._history[self._hist_i]
                           if self._hist_i < len(self._history) else self._saved_input)
            self._cursor = len(self._input)
            self._dirty = True

    def _redraw(self, stdscr, app) -> None:
        curses = self.curses
        height, width = stdscr.getmaxyx()
        with self._lock:
            lines = list(self._lines)
            status = self._status
            inp = self._input
            cursor = self._cursor
            modal = self._modal
            self._dirty = False

        rows: List[str] = []
        for ln in lines:
            rows.extend(_wrap(ln, width - 1))
        body_h = max(1, height - 2)
        view = rows[-body_h:] if len(rows) > body_h else rows

        stdscr.erase()
        for i, row in enumerate(view):
            try:
                stdscr.addnstr(i, 0, row, width - 1)
            except curses.error:
                pass

        if modal is not None:
            stext = f"\u26A0 {modal.prompt} [{'/'.join(modal.choices)}]"
        else:
            stext = status
        try:
            stdscr.addnstr(height - 2, 0, stext, width - 1)
            stdscr.addnstr(height - 1, 0, "> " + inp, width - 1)
            stdscr.move(height - 1, min(2 + cursor, width - 1))
        except curses.error:
            pass
        stdscr.refresh()


def _make(app):
    """Renderer factory: None means 'not applicable, use stdio'."""
    if os.environ.get("TERM", "") in ("", "dumb"):
        return None
    if not (sys.stdout.isatty() and sys.stdin.isatty()):
        return None
    try:
        import curses  # noqa: F401
    except Exception:
        return None
    return TuiRenderer(app)


def register(app) -> None:
    app.set_renderer(_make)
