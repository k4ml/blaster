#!/usr/bin/env python3
"""
Wheeljack TUI plugin — a curses renderer, loaded as a sibling plugin.

Drop this file in the plugin directory (default ./.wheeljack/plugins) named
`tui.py`; `register(app)` installs a renderer factory. Core falls
back to the stdio renderer whenever this returns None (no TTY, TERM=dumb, or
curses unavailable), so piping/redirecting still works exactly as before.

This is the "first plugin" the foundation was built to prove: it only needs
the public surface (app.set_renderer, BaseRenderer, app.submit_line,
app.quit_requested) and the shared modal queue — no core changes.

Design notes (vs the old in-core CursesRenderer):
    - a growable scrollback of logical lines that WRAPS, instead of a fixed
      2000-row pad that raised curses.error once it filled
    - real scrollback: PgUp/PgDn scroll by a page, Shift+Up/Down by a row,
      Home/End jump to the oldest line and back to the tail, and the mouse
      wheel scrolls a few rows. Scrolling up detaches from the tail so
      incoming output does not move the text you are reading; End (or PgDn
      past the bottom) re-attaches
    - mouse wheel reporting is enabled, so drag-select to copy text needs
      Shift held down (the usual terminal convention for this tradeoff)
    - redraw only when dirty, not every 50 ms unconditionally
    - keypad(True) + get_wch() so arrows/unicode/KEY_RESIZE work
    - a real line editor in the input box (history, cursor movement)
    - the modal queue lives in BaseRenderer, so concurrent confirms can't hang
"""
from __future__ import annotations

import os
import sys
import threading
from typing import List, Optional, Tuple

import wheeljack as core  # resolved via the alias core installs in main()

# Oldest lines are dropped past this many, so a long session cannot grow the
# buffer without bound. Generous enough that scrolling back stays useful.
SCROLLBACK_MAX_LINES = 5000

# Rows moved per mouse-wheel notch. Terminals send one event per notch, so this
# is a deliberate step rather than a line-by-line crawl.
WHEEL_ROWS = 3


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
        self._lines: List[Tuple[str, bool]] = []
        self._status = ""
        self._input = ""
        self._cursor = 0
        self._history: List[str] = []
        self._hist_i = 0
        self._saved_input = ""
        self._modal: Optional[core.ModalRequest] = None
        self._running = True
        self._dirty = True
        # Scrollback: None = follow the tail. An int pins the top-most visible
        # row, so output arriving while you read does not shift the viewport.
        self._scroll: Optional[int] = None
        self._last_body_h = 0        # body height from the last frame
        self._last_total_rows = 0    # wrapped row count from the last frame

    # -- event API: mutate the buffer under lock, the UI thread draws it --
    def _append_line(self, text: str, dim: bool = False) -> None:
        with self._lock:
            self._lines.append((text, dim))
            if len(self._lines) > SCROLLBACK_MAX_LINES:
                # Drop the oldest lines so a long session cannot grow without
                # bound; the pinned offset is adjusted so the viewport does not
                # jump while scrolled back.
                del self._lines[:len(self._lines) - SCROLLBACK_MAX_LINES]
                if self._scroll is not None:
                    self._scroll = max(0, self._scroll - 1)
            self._dirty = True

    def _append_text(self, text: str) -> None:
        with self._lock:
            dropped = 0
            for i, part in enumerate(text.split("\n")):
                if i == 0:
                    if self._lines:
                        last_text, last_dim = self._lines[-1]
                        self._lines[-1] = (last_text + part, last_dim)
                    else:
                        self._lines.append((part, False))
                else:
                    self._lines.append((part, False))
            if len(self._lines) > SCROLLBACK_MAX_LINES:
                dropped = len(self._lines) - SCROLLBACK_MAX_LINES
                del self._lines[:dropped]
                if self._scroll is not None:
                    self._scroll = max(0, self._scroll - dropped)
            self._dirty = True

    def turn_started(self, e) -> None:
        with self._lock:
            self._status = ""
            self._dirty = True
        self._append_line("")

    def turn_ended(self, e) -> None:
        with self._lock:
            if self._status.startswith("thinking:"):
                self._status = ""
            self._dirty = True
        agent = getattr(self.app, "agent", None)
        if agent is None:
            return
        answer = getattr(agent, "last_answer", "") or ""
        if answer == "(no response)":
            self._append_line("(no response from model)")
            return
        if answer and not getattr(agent, "last_streamed", False):
            self._append_line(answer)

    def tool_started(self, e) -> None:
        self._append_line(f"  [tool] {e.name}({e.args}) ...", dim=True)

    def tool_finished(self, e) -> None:
        mark = "OK" if e.ok else "FAIL"
        self._append_line(f"  [{mark}] {e.name}: {e.result}", dim=True)

    def stream_delta(self, e) -> None:
        with self._lock:
            if self._status.startswith("thinking:"):
                self._status = ""
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
        # Enable wheel reporting so the scrollback is reachable by mouse. This
        # takes over drag-to-select, which is why the docs note Shift+drag.
        try:
            # This build's mousemask() takes a single mask argument, and it
            # raises TypeError (not curses.error) when called otherwise, so the
            # except must be broad - a narrow one lets the whole TUI fall over.
            curses.mousemask(curses.ALL_MOUSE_EVENTS)
            # Short interval: wheel notches arrive as separate events rather
            # than coalescing into one click.
            curses.mouseinterval(50)
            curses.ESCDELAY = 25
        except Exception:
            pass   # a terminal without mouse support just uses the keys

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
        # Mouse first: the wheel can scroll while a modal is showing (it cannot
        # answer the modal), and decoding it needs getmouse() which may raise.
        if ch == curses.KEY_MOUSE:
            self._handle_mouse(stdscr)
            return

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
            self._scroll = None          # sending re-attaches to the tail
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
            # KEY_UP is input history; scrollback uses PgUp / Shift+Up.
            self._history_up()
        elif ch == curses.KEY_DOWN:
            self._history_down()
        elif ch == curses.KEY_PPAGE:
            self._scroll_up(self._last_body_h or 1)
        elif ch == curses.KEY_NPAGE:
            self._scroll_down(self._last_body_h or 1)
        elif ch == curses.KEY_SR:            # Shift+Up
            self._scroll_up(1)
        elif ch == curses.KEY_SF:            # Shift+Down
            self._scroll_down(1)
        elif ch == curses.KEY_HOME:
            self._scroll_to_top()
        elif ch == curses.KEY_END:
            self._scroll_to_tail()
        elif ch == curses.KEY_RESIZE:
            self._dirty = True
        elif isinstance(ch, str) and ch.isprintable():
            self._input = self._input[:self._cursor] + ch + self._input[self._cursor:]
            self._cursor += 1
            self._dirty = True

    # -- mouse ------------------------------------------------------------
    def _wheel_direction(self, bstate: int) -> int:
        """+1 for wheel-down (toward live), -1 for wheel-up, 0 if not a wheel.

        Shift is respected: it is the usual modifier for selecting text in a
        terminal, so a Shift+wheel must not steal the drag.
        """
        curses = self.curses
        if getattr(curses, "BUTTON_SHIFT", 0) and bstate & curses.BUTTON_SHIFT:
            return 0
        up = (getattr(curses, "BUTTON4_PRESSED", 0)
              | getattr(curses, "BUTTON4_CLICKED", 0)
              | getattr(curses, "BUTTON4_RELEASED", 0))
        down = (getattr(curses, "BUTTON5_PRESSED", 0)
                | getattr(curses, "BUTTON5_CLICKED", 0)
                | getattr(curses, "BUTTON5_RELEASED", 0))
        if bstate & up:
            return -1
        if bstate & down:
            return 1
        return 0

    def _handle_mouse(self, stdscr) -> None:
        curses = self.curses
        try:
            _, _, _, _, bstate = curses.getmouse()
        except curses.error:
            return   # event already consumed; nothing we can do
        self._handle_mouse_event(bstate)

    def _handle_mouse_event(self, bstate: int) -> None:
        direction = self._wheel_direction(bstate)
        if direction < 0:
            self._scroll_up(WHEEL_ROWS)
        elif direction > 0:
            self._scroll_down(WHEEL_ROWS)

    # -- scrollback -------------------------------------------------------
    def _clamp_scroll(self, start: int) -> int:
        """Keep the pinned top row inside the buffer."""
        max_top = max(0, self._last_total_rows - self._last_body_h)
        return max(0, min(start, max_top))

    def _scroll_up(self, rows: int) -> None:
        """Move the viewport up (toward older output) and detach from the tail."""
        with self._lock:
            total = self._last_total_rows
            body = self._last_body_h
            if total <= body:
                return  # everything already fits; nothing to scroll
            # Detaching pins the current top row, which is total - body.
            current = self._scroll if self._scroll is not None else total - body
            new = self._clamp_scroll(current - rows)
            if new != self._scroll:
                self._scroll = new
                self._dirty = True

    def _scroll_down(self, rows: int) -> None:
        """Move the viewport down; reaching the bottom re-attaches to the tail."""
        with self._lock:
            if self._scroll is None:
                return  # already following the tail
            total = self._last_total_rows
            body = self._last_body_h
            new = self._scroll + rows
            if new >= max(0, total - body):
                self._scroll = None      # back to following live output
            else:
                self._scroll = self._clamp_scroll(new)
            self._dirty = True

    def _scroll_to_top(self) -> None:
        with self._lock:
            if self._last_total_rows <= self._last_body_h:
                return
            self._scroll = 0
            self._dirty = True

    def _scroll_to_tail(self) -> None:
        with self._lock:
            if self._scroll is not None:
                self._scroll = None
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
            scroll = self._scroll
            self._dirty = False

        rows: List[Tuple[str, bool]] = []
        for ln, dim in lines:
            for wrapped in _wrap(ln, width - 1):
                rows.append((wrapped, dim))
        body_h = max(1, height - 2)
        total = len(rows)

        # Publish the geometry so the key handlers can clamp against it.
        with self._lock:
            self._last_body_h = body_h
            self._last_total_rows = total

        if scroll is None or total <= body_h:
            top = max(0, total - body_h)      # follow the tail
            following = True
        else:
            top = max(0, min(scroll, total - body_h))
            following = False
            with self._lock:
                # Re-clamp if the buffer grew past the pinned position.
                if self._scroll != top:
                    self._scroll = top
        view = rows[top:top + body_h]

        stdscr.erase()
        for i, (row, dim) in enumerate(view):
            try:
                attr = curses.A_DIM if dim else 0
                stdscr.addnstr(i, 0, row, width - 1, attr)
            except curses.error:
                pass

        if following:
            indicator = ""
        else:
            hidden = max(0, total - (top + body_h))
            # The way back to live output matters more than the row counts, so
            # degrade by dropping the counts first on a narrow terminal.
            indicator = (f"[SCROLLED \u2191{top} \u2193{hidden} "
                         f"\u00b7 End=live]")
            if len(indicator) > width - 1:
                indicator = "[SCROLLED \u00b7 End=live]"
            if len(indicator) > width - 1:
                indicator = "[SCROLLED]"
        if modal is not None:
            stext = f"\u26A0 {modal.prompt} [{'/'.join(modal.choices)}]"
        elif indicator:
            stext = f"{status} {indicator}" if status else indicator
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
