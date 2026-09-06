#!/usr/bin/env python3
"""Blaster: single-file server-ops & coding agent (stdlib only).

Named after the Transformers Autobot Blaster — the agent that gets things
done on the host it runs on. Uses an OpenAI-compatible chat endpoint
(Ollama/llama.cpp/OpenRouter/...) as a natural-language shell: inspect/edit
files, run shell commands, configure and maintain services. Tools:
read_file, write_file, edit_file, list_files, run_shell, run_interactive.
Destructive or sudo commands require y/N approval.

Usage:
    python blaster.py [--model qwen3.8:27b] [--api-base http://localhost:11434/v1]
                      [--cwd DIR] [--session NAME] [-n MAX_ITERATION]
Env: BLASTER_MODEL / BLASTER_API_BASE / OPENAI_API_KEY (for remote endpoints)
"""
import argparse, json, os, random, re, shutil, subprocess, sys, termios, tty, urllib.error, urllib.request, uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Config ---------------------------------------------------------------
@dataclass
class Config:
    api_base: str = os.getenv("BLASTER_API_BASE", "http://localhost:11434/v1")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("BLASTER_MODEL", "qwen3.8:27b")
    max_tokens: int = 2000
    temperature: float = 0.2
    request_timeout: int = 120
    max_context_files: int = 20
    max_file_size: int = 100_000        # read_file cap (100KB)
    max_output_chars: int = 40_000      # run_shell output cap
    max_iterations: int = 50            # tool round limit per turn
    sessions_dir: Path = Path.home() / ".blaster" / "sessions"
    cwd: Path = Path.cwd()
    # Render non-interactive markdown answers with glow when available and
    # stdout is a TTY; -x/--no-format disables.
    format_markdown: bool = True


# Command patterns needing y/N approval (whole-word; harmless uses like
# `grep rm` are not flagged). Declined commands are remembered for the session.
DESTRUCTIVE_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+", r"\bmkfs(?:\.[a-z0-9]+)?\b",
    r"\bdd\b", r"\bmkswap\b|\bswapoff\b", r"\bparted\b|\bfdisk\b|\bsfdisk\b",
    r"\bkill\s+-9\b|\bpkill\s+-9\b", r"\bgit\s+push\s+(-f|--force)\b", r"\bmv\s+/\s+",
]
SUDO_PATTERN = re.compile(r"(^|[;&|]\s*)sudo\s+")


# Data structures ------------------------------------------------------
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


class EnhancedInput:
    """Raw-mode line editor with history (arrow keys, backspace, Ctrl+A/E/U)."""

    def __init__(self):
        self.history: List[str] = []
        self.hist_i = 0
        self.line = ""
        self.pos = 0
        self._saved = ""

    # --- line editing ---------------------------------------------------
    def _redraw(self):
        # Repaint prompt+line and park the cursor at `pos`.
        sys.stdout.write("\r\x1b[K>> " + self.line + "\r>> " + self.line[:self.pos])
        sys.stdout.flush()

    def _key(self) -> str:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch

    def _insert(self, ch: str):
        self.line = self.line[:self.pos] + ch + self.line[self.pos:]
        self.pos += 1
        self._redraw()

    def _delete(self):
        if self.pos == 0:
            return
        self.line = self.line[:self.pos - 1] + self.line[self.pos:]
        self.pos -= 1
        self._redraw()

    def _hist(self, direction: int):
        if not self.history:
            return
        if direction < 0:  # up: older
            if self.hist_i == 0:
                self._saved = self.line
            if self.hist_i < len(self.history):
                self.hist_i += 1
                self.line = self.history[-self.hist_i]
        else:              # down: newer
            if self.hist_i > 0:
                self.hist_i -= 1
                self.line = self.history[-self.hist_i] if self.hist_i else self._saved
        self.pos = len(self.line)
        self._redraw()

    def readline(self, prompt: str = ">> ") -> str:
        # Non-TTY (piped/scripted) fallback.
        if not sys.stdin.isatty():
            try:
                line = input(prompt).strip()
            except EOFError:
                raise KeyboardInterrupt
            if line and (not self.history or self.history[-1] != line):
                self.history.append(line)
            return line

        sys.stdout.write(prompt)
        sys.stdout.flush()
        self.line, self.pos, self.hist_i = "", 0, 0
        while True:
            try:
                ch = self._key()
                if ch in ("\r", "\n"):
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    line = self.line.strip()
                    if line and (not self.history or self.history[-1] != line):
                        self.history.append(line)
                    return line
                if ch == "\x03":
                    raise KeyboardInterrupt
                if ch in ("\x7f", "\x08"):        # backspace
                    self._delete()
                elif ch == "\x15":                 # Ctrl+U clear
                    self.line, self.pos = "", 0
                    self._redraw()
                elif ch == "\x01":                 # Ctrl+A home
                    self.pos = 0
                    self._redraw()
                elif ch == "\x05":                 # Ctrl+E end
                    self.pos = len(self.line)
                    self._redraw()
                elif ch == "\x1b":                 # escape sequence
                    seq = self._key() + self._key()
                    if seq == "[A":
                        self._hist(-1)
                    elif seq == "[B":
                        self._hist(1)
                    elif seq == "[C" and self.pos < len(self.line):
                        self.pos += 1
                        self._redraw()
                    elif seq == "[D" and self.pos > 0:
                        self.pos -= 1
                        self._redraw()
                elif ch >= " ":
                    self._insert(ch)
            except KeyboardInterrupt:
                sys.stdout.write("^C\n")
                sys.stdout.flush()
                raise


def _truncate(text: str, limit: int) -> str:
    """Truncate text to limit chars, keeping whole lines, with a marker."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if "\n" in cut:
        cut = cut.rsplit("\n", 1)[0]
    return f"{cut}\n... [output truncated, {len(text) - len(cut)} more chars]"


# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------
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
    """Render markdown via glow when it exists and stdout is a TTY.

    Glow reads the markdown on stdin and writes styled output straight to the
    terminal (its own stdout must be a TTY to emit ANSI). Falls back to
    printing the plain text when glow is missing or output is piped.
    """
    if not _USE_COLOR or not _GLOW_PATH or not text.strip():
        print(text)
        return
    try:
        proc = subprocess.Popen([_GLOW_PATH], stdin=subprocess.PIPE)
        proc.communicate(text.encode(), timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        print(text)


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI-style function calling)
# ---------------------------------------------------------------------------
# (name, description, props, required) — props maps arg name -> (type, desc).
_TOOL_SPECS = [
    ("read_file", "Read a file's content. Use when the user asks about a file, config, log, or file content.",
     {"path": ("string", "File path, absolute or relative to cwd")}, ["path"]),
    ("write_file", "Create a new file or overwrite an existing one.",
     {"path": ("string", "File path, absolute or relative to cwd"),
      "content": ("string", "Full new content of the file")}, ["path", "content"]),
    ("edit_file", "Edit an existing file by replacing a unique substring (before) with new text (after). A .bak backup is made.",
     {"path": ("string", "File path, absolute or relative to cwd"),
      "before": ("string", "Exact existing text to find (must be unique)"),
      "after": ("string", "Replacement text")}, ["path", "before", "after"]),
    ("list_files", "List files and directories under a path (recursive=true for a tree).",
     {"path": ("string", "Directory to list, default '.'"),
      "recursive": ("boolean", "List recursively as a tree")}, []),
    ("run_shell", "Run a shell command (bash -c) on this machine, returning stdout/stderr and exit code. Use for server ops: service status, logs, package installs, process checks, systemctl, docker, disk/network, etc. Destructive or sudo commands require the user's y/N approval.",
     {"command": ("string", "The shell command to run"),
      "timeout": ("integer", "Seconds to wait before killing (default 30)")}, ["command"]),
    ("run_interactive", "Run an interactive terminal program (editor, pager, dialog, top). Output is not captured; the user operates it directly.",
     {"command": ("string", "Command to run interactively")}, ["command"]),
]


def _get_tools() -> List[Dict]:
    """Return the OpenAI-style JSON schema for every tool."""
    tools = []
    for name, desc, props, required in _TOOL_SPECS:
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {p: {"type": t, "description": d}
                                    for p, (t, d) in props.items()},
                    "required": required,
                },
            },
        })
    return tools


class BasicCodingAgent:
    def __init__(self, config: Config = Config(), session_name: Optional[str] = None):
        self.config = config
        self.messages: List[Message] = []
        # Once a dangerous/sudo command is declined, block further attempts in
        # this session so the model cannot silently retry it.
        self._declined_commands: List[str] = []
        # Working directory for all tools (files + shell). Defaults to the
        # launch directory; --cwd overrides it.
        self.project_root = config.cwd.resolve()
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.session_context: Optional[SessionContext] = None
        self.conversation_summary = ""
        self.total_tokens_used = 0
        self.input_handler = EnhancedInput()

        # Setup sessions directory
        self.config.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Sessions are identified by a unique id and a human-friendly name
        # describing the task. Load a session by name if one was given,
        # otherwise start a fresh session with an auto-generated name.
        if session_name:
            self._load_or_create_session(session_name)
        else:
            self._create_new_session(self._generate_session_name())

        # Initialize with system prompt
        self._initialize_system_prompt()

    @staticmethod
    def _generate_session_name() -> str:
        """Generate a human-friendly session name (adjective-noun)."""
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
        """Generate a unique session id (used as the storage filename)."""
        return uuid.uuid4().hex[:12]

    def _load_or_create_session(self, name: str):
        """Load a session by name, or create a fresh one if none exists.

        Sessions identify tasks, not directories, so the name is matched
        across every saved session regardless of project.
        """
        target = self._find_session_by_name(name)
        if target is None:
            self._create_new_session(name)
            return

        try:
            session_file, session_data = target
            context_data = session_data.get('context', {})

            self.session_context = SessionContext(
                session_id=context_data.get('session_id', session_file.stem),
                name=context_data.get('name', name),
                created_at=context_data.get('created_at', datetime.now().isoformat()),
                last_accessed=datetime.now().isoformat(),
                message_count=context_data.get('message_count', 0),
            )

            # Load conversation summary
            self.conversation_summary = session_data.get('conversation_summary', '')
            self.total_tokens_used = session_data.get('total_tokens', 0)

            # Load previous messages (limited to recent ones for context)
            saved_messages = session_data.get('messages', [])
            if saved_messages:
                recent_messages = saved_messages[-100:]  # Load last 100
                for msg_data in recent_messages:
                    role = msg_data.get('role', 'user')
                    content = msg_data.get('content', '')
                    if role == 'system':
                        # The saved file carries the old system prompt;
                        # a fresh one is added when the agent starts.
                        continue
                    if role == 'tool':
                        # Persisted tool messages carry the call id in
                        # the timestamp slot of older sessions; just
                        # synthesize a placeholder name so reloading
                        # produces a well-formed tool result.
                        self.messages.append(Message(
                            role='tool',
                            content=content,
                            tool_call_id=msg_data.get('tool_call_id', 'legacy'),
                            name=msg_data.get('name', 'tool'),
                            timestamp=msg_data.get('timestamp')
                        ))
                    else:
                        self.messages.append(Message(
                            role=role,
                            content=content,
                            timestamp=msg_data.get('timestamp')
                        ))

            print(f"📂 Loaded session '{self.session_context.name}' "
                  f"({len(saved_messages)} previous messages, will be saved on quit)")

        except Exception as e:
            print(f"⚠️  Could not load session: {e}. Starting fresh session.")
            self._create_new_session(name)

    def _find_session_by_name(self, name: str):
        """Return (session_file, session_data) for a saved session with this name."""
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

    def _create_new_session(self, name: Optional[str] = None):
        """Create a new session, generating a unique id and a name if needed."""
        if not name:
            name = self._generate_session_name()
        self.session_context = SessionContext(
            session_id=self._generate_session_id(),
            name=name,
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            message_count=0,
        )
        print(f"📝 Created new session '{name}' (will be saved on quit)")

    def _save_session(self):
        """Save current session to disk"""
        if not self.session_context:
            return

        try:
            session_file = self.config.sessions_dir / f"{self.session_context.session_id}.json"

            # Update session context
            self.session_context.last_accessed = datetime.now().isoformat()
            self.session_context.message_count = len(self.messages)

            # Prepare session data
            session_data = {
                'context': {
                    'session_id': self.session_context.session_id,
                    'name': self.session_context.name,
                    'created_at': self.session_context.created_at,
                    'last_accessed': self.session_context.last_accessed,
                    'message_count': self.session_context.message_count,
                    'project_root': str(self.project_root),
                },
                'conversation_summary': self.conversation_summary,
                'total_tokens': self.total_tokens_used,
                'messages': [
                    {
                        'role': msg.role,
                        'content': msg.content,
                        'timestamp': msg.timestamp or datetime.now().isoformat(),
                        'tool_call_id': msg.tool_call_id,
                        'name': msg.name,
                    }
                    for msg in self.messages
                ]
            }

            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"⚠️  Could not save session: {e}")

    def _generate_conversation_summary(self) -> str:
        """Generate a brief summary of recent conversation for context"""
        if not self.messages:
            return ""

        # Take last few user-assistant exchanges
        recent_exchanges = []
        for i in range(max(0, len(self.messages) - 6), len(self.messages)):
            msg = self.messages[i]
            if msg.role in ['user', 'assistant'] and len(msg.content) < 200:
                role = "User" if msg.role == "user" else "Assistant"
                recent_exchanges.append(f"{role}: {msg.content[:100]}...")

        return " | ".join(recent_exchanges[-3:])  # Last 3 exchanges

    def _initialize_system_prompt(self):
        """Create system prompt with tool rules and session context"""
        project_info = self._get_project_info()
        session_info = self._get_session_info()

        system_prompt = f"""You are Blaster, a server operations and coding assistant running directly on this machine. You help set up, maintain, configure, and debug servers and code, and you execute actions via tools.

You operate in the working directory: {self.project_root}
The current date is {datetime.now().strftime('%Y-%m-%d')}.

{project_info}

{session_info}

TOOL USE RULES (critical):
- Prefer using tools over merely describing what to do. You can actually change the system.
- Before reading or editing a file, run_shell 'ls' / read_file to see what exists. If an edit_file 'before' string is rejected, read the file to get exact content.
- run_shell output is returned to you verbatim (capped). Use exit codes to judge success. For long-running commands pass a generous timeout.
- When a command fails, inspect the error and try to fix it yourself (check logs, configs, dependencies) before giving up.
- Commands flagged as destructive (rm -rf, mkfs, dd, git push -f, etc.) or any command using sudo will prompt the user for y/N approval. If approval is declined, do NOT retry the same command; explain and offer a safer alternative.
- run_interactive starts a program on the user's terminal (editors, top, etc.); the user operates it directly. Prefer run_shell for anything that can run non-interactively.
- Ask before making large or irreversible changes you cannot verify (e.g. deleting data, wiping disks, changing firewall rules). For routine config edits and service restarts, just proceed.
- Keep responses concise. Summarize what you changed and the resulting state.

You can call the following tools (schemas are sent with each request):
read_file, write_file, edit_file, list_files, run_shell, run_interactive
"""
        self.messages.append(Message(role="system", content=system_prompt))

    def _get_session_info(self) -> str:
        """Get session-specific information for context"""
        if not self.session_context:
            return ""

        session_info = []
        session_info.append("Session Context:")
        session_info.append(f"  Session: {self.session_context.name}")
        session_info.append(f"  ID: {self.session_context.session_id}")
        session_info.append(f"  Created: {self.session_context.created_at.split('T')[0]}")

        if self.conversation_summary:
            session_info.append(f"  Recent: {self.conversation_summary}")

        return "\n".join(session_info)

    def _get_project_info(self) -> str:
        """Get basic information about the working directory"""
        info = []

        # Working directory listing
        try:
            entries = sorted(self.project_root.iterdir(),
                             key=lambda p: (not p.is_dir(), p.name.lower()))
            files = []
            for item in entries[:self.config.max_context_files]:
                if item.is_file():
                    files.append(f"  - {item.name}")
                elif item.is_dir():
                    files.append(f"  - {item.name}/")
            if entries:
                info.append("Working directory contents (top-level):")
                info.extend(files)
            else:
                info.append("Working directory is empty.")
        except Exception as e:
            info.append(f"Could not read working directory: {e}")

        # Git info if available
        try:
            if (self.project_root / ".git").exists():
                info.append("\nThis is a git repository.")
        except Exception:
            pass

        return "\n".join(info)

    def _call_llm(self, messages: List[Message]) -> LLMResponse:
        """Call an OpenAI-compatible chat completions endpoint.

        Uses the local Ollama-style /chat/completions JSON API. Sends the
        `tools` array only when the tail of the conversation actually contains
        tool calls (or a system prompt that references them), which keeps the
        prompt smaller and lets local models behave. Parses a non-streamed JSON
        response; if the server returns NDJSON streaming deltas it reads them
        and reassembles a normal-looking response object.
        """
        if not self.config.api_key:
            # Local endpoints (Ollama) accept an empty key; remote providers
            # require one. Only raise if the endpoint does not look local.
            if "localhost" not in self.config.api_base and "127.0.0.1" not in self.config.api_base:
                raise ValueError(
                    "API key required for non-local endpoint. Set OPENAI_API_KEY.")

        # Trim the oldest messages so a long session cannot overflow the
        # model's context window.
        history = self._trim_messages(messages)

        need_tools = self._conversation_needs_tools(history)
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [self._message_to_dict(msg) for msg in history],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if need_tools:
            payload["tools"] = _get_tools()
            payload["tool_choice"] = "auto"

        # Make HTTP request
        url = f"{self.config.api_base.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.config.request_timeout) as response:
                ctype = response.headers.get("Content-Type", "")
                raw = response.read()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", "replace") if e.fp else str(e)
            raise Exception(f"API Error {e.code}: {error_body}")
        except Exception as e:
            raise Exception(f"Request failed: {e}")

        try:
            text = raw.decode("utf-8", "replace")
        except Exception:
            text = ""

        if "application/x-ndjson" in ctype or text.lstrip().startswith("{"):
            # Non-streamed: parse the single JSON object (or an NDJSON doc).
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Streaming fallback: server returned NDJSON lines.
                data = self._parse_streaming_ndjson(text)
            if data.get("error"):
                raise Exception(f"API error: {data['error']}")
            return LLMResponse(
                choices=data.get("choices", []),
                usage=data.get("usage",
                               {"prompt_tokens": 0, "completion_tokens": 0})
            )

        # Otherwise treat the body as NDJSON streaming chunks.
        data = self._parse_streaming_ndjson(text)
        if not data.get("choices"):
            raise Exception(f"Empty/unexpected API response: {text[:500]}")
        return LLMResponse(choices=data["choices"], usage=data.get("usage", {
            "prompt_tokens": 0, "completion_tokens": 0}))

    def _parse_streaming_ndjson(self, text: str) -> Dict:
        """Reassemble an OpenAI-style response from NDJSON stream chunks."""
        content_parts: List[str] = []
        tool_calls_acc: Dict[str, Dict[str, Any]] = {}
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
            delta = chunk["choices"][0].get("delta", {})
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
        # Filter out chunks that were only usage / keep a stable shape.
        return {
            "choices": [{
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }

    def _trim_messages(self, messages: List[Message]) -> List[Message]:
        """Drop the oldest messages once the conversation grows too long.

        Keeps the system prompt, the most recent N messages, and never splits
        an assistant tool_call block away from the tool results that follow it.
        """
        # Conversation window sent to the model each turn. 100 matches the
        # number of messages restored from a saved session, so a resumed
        # conversation keeps its full history in context.
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
        """True if the request should advertise the tool schema.

        We advertise tools when the last user/assistant turn happened, i.e.
        whenever the tail of the conversation might still trigger a tool call.
        Returning False keeps the prompt small when only a plain answer is
        expected (e.g. a summary after a long command), but risks a local
        model answering instead of calling a tool it was never told about.
        """
        if len(history) >= 2:
            return True
        # Fresh session with just the system prompt: tools should be available.
        return any("Available Tools" in (m.content or "") for m in history[:1])

    def _message_to_dict(self, message: Message) -> Dict:
        """Convert Message to dict for API"""
        result: Dict[str, Any] = {
            "role": message.role,
            "content": message.content
        }

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": tc.function
                }
                for tc in message.tool_calls
            ]

        if message.tool_call_id:
            result["tool_call_id"] = message.tool_call_id

        if message.name:
            result["name"] = message.name

        return result

    def _parse_tool_calls(self, response: LLMResponse) -> List[ToolCall]:
        """Parse tool calls from LLM response"""
        tool_calls = []

        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if choice.get('finish_reason') == 'tool_calls':
                for tc in choice.get('message', {}).get('tool_calls', []):
                    tool_calls.append(ToolCall(
                        id=tc['id'],
                        function=tc['function']
                    ))

        return tool_calls

    def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return result"""
        tool_name = tool_call.function['name']
        args = tool_call.function.get('arguments', '{}')

        try:
            args_dict = json.loads(args)
        except json.JSONDecodeError:
            return f"Error: Invalid tool arguments: {args}"

        try:
            if tool_name == "read_file":
                path = args_dict.get('path', '')
                return self._read_file(path)

            elif tool_name == "write_file":
                path = args_dict.get('path', '')
                content = args_dict.get('content', '')
                return self._write_file(path, content)

            elif tool_name == "edit_file":
                path = args_dict.get('path', '')
                before = args_dict.get('before', '')
                after = args_dict.get('after', '')
                return self._edit_file(path, before, after)

            elif tool_name == "list_files":
                path = args_dict.get('path', '.')
                recursive = args_dict.get('recursive', False)
                return self._list_files(path, recursive)

            elif tool_name == "run_shell":
                command = args_dict.get('command', '')
                timeout = int(args_dict.get('timeout', 30))
                return self._run_shell(command, timeout=timeout)

            elif tool_name == "run_interactive":
                command = args_dict.get('command', '')
                return self._run_interactive(command)

            else:
                return f"Error: Unknown tool {tool_name}"

        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    def _read_file(self, path: str) -> str:
        """Read file content (truncated to max_file_size)"""
        file_path = self._resolve_path(path)

        if not file_path.exists():
            return f"Error: File not found: {file_path}"
        if file_path.is_dir():
            return f"Error: {file_path} is a directory, not a file"

        file_size = file_path.stat().st_size
        if file_size > self.config.max_file_size:
            return (f"Error: File too large ({file_size} bytes, "
                    f"max {self.config.max_file_size})")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            return (f"Error: {file_path} is not valid UTF-8 text "
                    f"({file_size} bytes); use run_shell to inspect it")
        except Exception as e:
            return f"Error reading file: {e}"

        return f"--- {file_path} ---\n{content}\n--- end {file_path} ---"

    def _write_file(self, path: str, content: str) -> str:
        """Write content to file, creating a .bak backup if the file exists"""
        file_path = self._resolve_path(path)

        # Create backup if the file already exists
        backup_path = file_path.with_suffix(file_path.suffix + '.bak')
        if file_path.exists():
            try:
                shutil.copy2(file_path, backup_path)
            except Exception as e:
                return f"Error backing up existing file: {e}"

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            return f"Error writing file: {e}"

        msg = f"Successfully wrote {len(content)} bytes to {file_path}"
        if backup_path.exists():
            msg += f" (backup: {backup_path.name})"
        return msg

    def _edit_file(self, path: str, before: str, after: str) -> str:
        """Edit a file by replacing a unique substring."""
        file_path = self._resolve_path(path)
        if not file_path.exists():
            return f"Error: File not found: {file_path}"
        if file_path.is_dir():
            return f"Error: {file_path} is a directory"

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return f"Error reading file: {e}"

        if before not in content:
            return (f"Error: 'before' text not found in {file_path}. "
                    f"Use read_file to get the exact current content first.")
        if content.count(before) > 1:
            return (f"Error: 'before' text is not unique in {file_path} "
                    f"({content.count(before)} matches). Include more context.")

        new_content = content.replace(before, after, 1)
        backup_path = file_path.with_suffix(file_path.suffix + '.bak')
        try:
            shutil.copy2(file_path, backup_path)
        except Exception as e:
            return f"Error backing up existing file: {e}"

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        except Exception as e:
            shutil.copy2(backup_path, file_path)
            return f"Error writing file: {e}"

        return (f"Successfully edited {file_path} "
                f"({len(before)} chars replaced, backup: {backup_path.name})")

    def _list_files(self, path: str = ".", recursive: bool = False) -> str:
        """List files in a directory, optionally as a recursive tree"""
        dir_path = self._resolve_path(path)

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
        except Exception as e:
            return f"Error listing files: {e}"

    def _resolve_path(self, path: str) -> Path:
        """Resolve file path relative to project root (cwd)"""
        if path.startswith('/'):
            return Path(path)
        return (self.project_root / path).resolve()

    def _add_tool_response(self, tool_call: ToolCall, result: str):
        """Add tool response to messages"""
        self.messages.append(Message(
            role="tool",
            content=result,
            tool_call_id=tool_call.id,
            name=tool_call.function['name'],
            timestamp=datetime.now().isoformat()
        ))

    def _run_shell(self, command: str, timeout: int = 30) -> str:
        """Run a shell command, capturing output. Returns text for the LLM."""
        if not command.strip():
            return "Error: empty command"

        # Safety: destructive or sudo commands need explicit y/N approval.
        if not self._confirm_dangerous_command(command):
            return ("[command blocked: user declined approval. "
                    "Abort or rephrase without destructive/sudo operations.]")

        timeout = max(1, min(int(timeout), 600))

        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                executable="/bin/bash",
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            return f"Error starting command: {e}"

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
        return _truncate(result.format(), self.config.max_output_chars)

    def _confirm_dangerous_command(self, command: str) -> bool:
        """Return True if a command may run without (further) approval."""
        dangerous = False
        reasons = []
        for pat in DESTRUCTIVE_PATTERNS:
            if re.search(pat, command):
                dangerous = True
                reasons.append("destructive pattern")
                break
        if SUDO_PATTERN.search(command):
            dangerous = True
            reasons.append("uses sudo")

        if not dangerous:
            return True

        # A previously-declined command stays blocked for this session, so the
        # model cannot keep retrying the same destructive/sudo operation.
        if command in self._declined_commands:
            return False

        try:
            sys.stdout.write(
                f"\n⚠️  This command is flagged as {', '.join(reasons)}:\n"
                f"    {command}\n"
                f"Run it? [y/N] ")
            sys.stdout.flush()

            # If stdin is not a TTY (piped/scripted), read a plain line; the
            # whole program should only run destructively when the user is at
            # a real terminal anyway.
            if not sys.stdin.isatty():
                try:
                    answer = sys.stdin.readline().strip().lower()
                except Exception:
                    answer = ""
                sys.stdout.write("\n")
                sys.stdout.flush()
                approved = answer in ("y", "yes")
                if not approved:
                    self._declined_commands.append(command)
                return approved

            # Temporarily restore cooked mode so the user can type freely.
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                answer = ""
                while True:
                    ch = sys.stdin.read(1)
                    if ch in ("\r", "\n"):
                        break
                    if ch in ("y", "Y", "n", "N"):
                        answer = ch
                        break
                    if ch == "\x03":
                        raise KeyboardInterrupt
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            sys.stdout.write("\n")
            sys.stdout.flush()
            approved = answer.lower() == "y"
            if not approved:
                self._declined_commands.append(command)
            return approved
        except KeyboardInterrupt:
            sys.stdout.write("\n[interrupted]\n")
            return False

    def _run_interactive(self, command: str) -> str:
        """Run an interactive program using the user's own TTY."""
        if not command.strip():
            return "Error: empty command"
        try:
            sys.stdout.write(f"\n[Starting interactive: {command}]\n")
            sys.stdout.flush()
            # Run without pipes so the child gets the real terminal.
            subprocess.call(command, shell=True,
                            executable="/bin/bash",
                            cwd=str(self.project_root))
            return f"[interactive command finished: {command}]"
        except Exception as e:
            return f"Error running interactive command: {e}"

    def _show_past_messages(self):
        """Recap the restored conversation so the user sees prior context.

        Message sequence in self.messages: [system, ...past turns...].
        Tool messages are the model's internal read/write/shell results; skip
        them and the synthetic echo messages to keep the recap readable.
        """
        past = [m for m in self.messages if m.role in ("user", "assistant")
                and m.content]
        if len(past) <= 1:
            return  # nothing restored (just the system prompt)

        n = len(past) - 1  # exclude the (empty) initial context
        print(_c(f"\n— previous conversation ({n} messages) —", "dim"))
        for m in past:
            who = _c("you", "cyan") if m.role == "user" else _c("blaster", "green")
            text = m.content.strip().replace("\n", " ")
            if len(text) > 300:
                text = text[:300] + "…"
            print(f"  {who}: {text}")
        print(_c("— end of previous conversation —", "dim"))

    def _run_turn(self) -> Optional[str]:
        """Run one full user turn: LLM call plus tool execution loop.

        The user message has already been appended to self.messages. Returns
        the final assistant text ('' if the model produced none).
        """
        response = self._call_llm(self.messages)
        choice = response.choices[0]
        if not choice:
            raise RuntimeError("empty response from LLM")

        # Tool execution loop: keep feeding results back to the model until it
        # answers without another tool call.
        tool_rounds = 0
        while True:
            tool_calls = self._parse_tool_calls(response)
            if not tool_calls:
                break

            tool_rounds += 1
            if tool_rounds > self.config.max_iterations:
                print(_c(f"⚠️  Tool round limit reached ({self.config.max_iterations}) - stopping to avoid a loop.", "red"))
                self.messages.append(Message(
                    role="user",
                    content=f"(system) Tool round limit reached ({self.config.max_iterations}). Stop calling tools and answer now.",
                    timestamp=datetime.now().isoformat()
                ))
                response = self._call_llm(self.messages)
                break

            print(f"\n🔧 {_c(f'Executing {len(tool_calls)} tool(s)...', 'bold')}")

            # Echo the assistant tool-call message so the model sees what
            # arguments it used (required before tool results).
            self.messages.append(Message(
                role="assistant",
                content="",
                tool_calls=[tc for tc in tool_calls],
                timestamp=datetime.now().isoformat()
            ))

            for tool_call in tool_calls:
                tool_name = tool_call.function['name']
                result = self._execute_tool(tool_call)
                self._print_tool_result(tool_name, result)
                self._add_tool_response(tool_call, result)

            print("🤖 " + _c("Getting next response...", "dim"))
            response = self._call_llm(self.messages)
            if not response.choices or not response.choices[0]:
                raise RuntimeError("empty response from LLM")

        choice = response.choices[0]
        assistant_message = (choice.get('message') or {}).get('content') or ""
        if assistant_message:
            self.messages.append(Message(
                role="assistant",
                content=assistant_message,
                timestamp=datetime.now().isoformat()
            ))

        # Update token usage
        usage = response.usage
        self.total_tokens_used += usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)

        return assistant_message

    def run_once(self, prompt: str) -> str:
        """Non-interactive mode: run a single prompt and return the answer."""
        self.messages.append(Message(
            role="user",
            content=prompt,
            timestamp=datetime.now().isoformat()
        ))
        try:
            answer = self._run_turn()
        finally:
            if self.session_context:
                self._save_session()
        if self.config.format_markdown and _USE_COLOR:
            _render_markdown(f"\n{answer}")
        else:
            print(f"\n🤖 {_c(answer, 'green')}")
        return answer

    def run(self):
        """Main interaction loop"""
        print(_c("Blaster — server ops & coding agent", "bold"))
        print(f"  {_c('Project:', 'cyan')} {self.project_root}")
        print(f"  {_c('Model:', 'cyan')}   {self.config.model}")
        print(f"  {_c('API:', 'cyan')}     {self.config.api_base}")
        if self.session_context:
            print(f"  {_c('Session:', 'cyan')} {self.session_context.name} "
                  f"({self.session_context.session_id}, auto-saved)")
        print(_c("Type 'quit' to exit. Ctrl+C to interrupt.", "dim"))
        self._show_past_messages()
        print()

        while True:
            try:
                # Get user input with enhanced editing
                user_input = self.input_handler.readline().strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    if self.session_context:
                        self._save_session()
                        print(f"💾 {_c('Session saved', 'green')}: {self.session_context.name}")
                    break

                if not user_input:
                    continue

                # Add user message with timestamp
                self.messages.append(Message(
                    role="user",
                    content=user_input,
                    timestamp=datetime.now().isoformat()
                ))

                try:
                    self._run_turn()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(_c(f"Error: {e}", "red"))

                # Auto-save session after every interaction
                if self.session_context:
                    self._save_session()

            except KeyboardInterrupt:
                if self.session_context:
                    self._save_session()
                    print(f"\n💾 {_c('Session saved', 'green')}: {self.session_context.name}")
                print("Goodbye!")
                break
            except Exception as e:
                print(_c(f"Error: {e}", "red"))
                # Don't break on errors, continue the loop

    def _print_tool_result(self, tool_name: str, result: str):
        """Show a concise one-line summary of a tool result."""
        preview = result.replace('\n', ' ')[:120]
        # icon + colored tool name.
        if tool_name == "run_shell":
            head = f"💻 {_c(tool_name, 'cyan')}"
        elif tool_name == "run_interactive":
            head = f"🖥️  {_c(tool_name, 'cyan')}"
        elif tool_name in ("write_file", "edit_file"):
            head = f"✏️  {_c(tool_name, 'green')}"
        elif tool_name == "list_files":
            head = f"📁 {_c(tool_name, 'yellow')}"
        else:
            head = f"📄 {_c(tool_name, 'yellow')}"

        if tool_name in ("run_shell", "run_interactive"):
            body = _c(preview, "dim")
            print(f"  {head}: {body}")
        else:
            print(f"  {head}: {preview}")


def _show_sessions(sessions_dir: Path) -> None:
    """List saved sessions (most recently used first)."""
    files = sorted(sessions_dir.glob("*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print(_c("No saved sessions.", "yellow"))
        print(f"(sessions are stored in {sessions_dir})")
        return

    hdr = (f"{_c('NAME', 'bold'):<22} {_c('ID', 'bold'):<14} {_c('MSGS', 'bold'):>5}  "
           f"{_c('CREATED', 'bold'):<10}  {_c('LAST USED', 'bold'):<10}")
    print(hdr)
    print("-" * len(hdr))
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
        print(f"{name:<22} {sid:<14} {str(msgs):>5}  {created:<10}  {last:<10}")
    print()
    print(_c("Resume one with: python blaster.py --session NAME", "dim"))


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(
        description='Blaster - server ops & coding agent (OpenAI-compatible LLM)')
    parser.add_argument('--model', type=str,
                        help='Model name (default: BLASTER_MODEL or qwen3.8:27b)')
    parser.add_argument('--api-base', dest='api_base_cli', type=str,
                        help='OpenAI-compatible API base, e.g. http://localhost:11434/v1')
    parser.add_argument('--session', type=str, nargs='?', const='__list__', default=None,
                        help='Session name to load; bare --session lists saved sessions')
    parser.add_argument('--cwd', type=str, default=None,
                        help='Working directory for tools (default: current dir)')
    parser.add_argument('-p', '--prompt', type=str, default=None,
                        help='Non-interactive mode: run a single prompt and exit')
    parser.add_argument('-n', '--max-iteration', dest='max_iterations', type=int,
                        help='Max tool rounds per turn (default: 50)')
    parser.add_argument('-x', '--no-format', dest='format_markdown',
                        action='store_false',
                        help='Disable glow markdown formatting in non-interactive mode')
    args = parser.parse_args()

    if args.session == '__list__':
        _show_sessions(Config().sessions_dir)
        return

    # Config: CLI flag > BLASTER_API_BASE env > local Ollama default. API key is
    # optional because local endpoints (Ollama) usually need none.
    config = Config()
    if args.model:
        config.model = args.model
    if args.api_base_cli:
        config.api_base = args.api_base_cli
    if args.cwd:
        config.cwd = Path(args.cwd).expanduser()
    if args.max_iterations is not None:
        config.max_iterations = args.max_iterations
    config.format_markdown = args.format_markdown

    # Create and run agent
    agent = BasicCodingAgent(config, args.session)

    # Non-interactive mode: run a single prompt and exit.
    if args.prompt:
        agent.run_once(args.prompt)
        return

    agent.run()


if __name__ == "__main__":
    main()
