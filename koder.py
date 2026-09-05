#!/usr/bin/env python3
"""
Koder - Single-file server operations & coding agent using only stdlib.

Turns natural language into server-admin and coding actions on the host it
runs on: inspect files/dirs, run shell commands, install/configure services,
edit configs, and check service state. It uses any OpenAI-compatible chat
completions endpoint (Ollama / llama.cpp / OpenRouter / OpenAI / etc.).

Tools (used via OpenAI-style function calling):
  read_file      - read a file (truncated if large)
  write_file     - create/overwrite a file
  edit_file      - apply a before/after string replacement to a file
  list_files     - list a directory tree
  run_shell      - run a shell command, capture stdout/stderr/exit code
  run_interactive- run an interactive program (editor/pager/dialog)

Safety:
  - Destructive shell commands (rm -rf, mkfs, dd, ...) require y/N approval.
  - Commands containing "sudo" require y/N approval.
  - run_shell output is capped so a runaway command cannot flood the context.

Usage:
    python koder.py [--model qwen3.8:27b] [--session NAME]
                    [--api-base http://localhost:11434/v1] [--cwd DIR]

    # Local models via Ollama (no API key needed)
    python koder.py

    # Remote OpenAI-compatible provider
    OPENAI_API_KEY=sk-... python koder.py --api-base https://openrouter.ai/api/v1 \
        --model openai/gpt-4o

Environment:
    KODER_MODEL, KODER_API_BASE   overrides for model / endpoint
    OPENAI_API_KEY                used when the endpoint requires a key
"""

import os
import json
import re
import subprocess
import urllib.request
import urllib.error
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil
from dataclasses import dataclass
from datetime import datetime
import sys
import termios
import tty


# Configuration
@dataclass
class Config:
    # Default to a local Ollama endpoint; override with --api-base / KODER_API_BASE.
    api_base: str = os.getenv("KODER_API_BASE", "http://localhost:11434/v1")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("KODER_MODEL", "qwen3.8:27b")
    max_tokens: int = 2000
    temperature: float = 0.2
    request_timeout: int = 120
    max_context_files: int = 20
    max_file_size: int = 100000          # 100KB cap for read_file
    max_output_chars: int = 40000         # cap for run_shell stdout+stderr
    sessions_dir: Path = Path.home() / ".koder" / "sessions"
    cwd: Path = Path.cwd()


# Regexes for commands that need explicit approval. Operate on the raw command
# string (whole-word) so plain uses like `grep -r rm /etc` are not flagged.
DESTRUCTIVE_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+",   # rm -r / rm -f / rm -rf
    r"\bmkfs(?:\.[a-z0-9]+)?\b",
    r"\bdd\b",
    r"\bmkswap\b|\bswapoff\b",
    r"\bparted\b|\bfdisk\b|\bsfdisk\b",
    r"\bkill\s+-9\b|\bpkill\s+-9\b",
    r"\bgit\s+push\s+(-f|--force)\b",
    r"\bmv\s+/\s+",
]
SUDO_PATTERN = re.compile(r"(^|[;&|]\s*)sudo\s+")


# Data structures
@dataclass
class Message:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict]] = None
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
    created_at: str
    last_accessed: str
    message_count: int


@dataclass
class BashToolResult:
    """Result of a run_shell / run_interactive execution."""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False

    def format(self) -> str:
        """Format for the LLM: stdout then stderr, with exit code."""
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
    """Enhanced input function with line editing capabilities"""

    def __init__(self):
        self.history: List[str] = []
        self.history_index = 0
        self.current_line = ""
        self.cursor_pos = 0

    def _getch(self) -> str:
        """Get a single character from stdin"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def _move_cursor(self, pos: int):
        """Move cursor to specified position"""
        sys.stdout.write(f"\r>> {self.current_line}")
        sys.stdout.write(f"\r>> {self.current_line[:pos]}")
        sys.stdout.flush()
        self.cursor_pos = pos

    def _insert_char(self, char: str):
        """Insert character at current cursor position"""
        if self.cursor_pos == len(self.current_line):
            self.current_line += char
            sys.stdout.write(char)
        else:
            self.current_line = (self.current_line[:self.cursor_pos] + char +
                               self.current_line[self.cursor_pos:])
            sys.stdout.write(self.current_line[self.cursor_pos:])
            sys.stdout.write(f"\r>> {self.current_line}")
            sys.stdout.write(f"\r>> {self.current_line[:self.cursor_pos + 1]}")
        sys.stdout.flush()
        self.cursor_pos += 1

    def _delete_char(self):
        """Delete character at current cursor position"""
        if self.cursor_pos == 0:
            return
        if self.cursor_pos == len(self.current_line):
            self.current_line = self.current_line[:-1]
            sys.stdout.write("\b \b")
        else:
            self.current_line = (self.current_line[:self.cursor_pos - 1] +
                               self.current_line[self.cursor_pos:])
            sys.stdout.write("\b")
            sys.stdout.write(self.current_line[self.cursor_pos - 1:] + " ")
            sys.stdout.write(f"\r>> {self.current_line}")
            sys.stdout.write(f"\r>> {self.current_line[:self.cursor_pos - 1]}")
        sys.stdout.flush()
        self.cursor_pos -= 1

    def _clear_line(self):
        """Clear the current line"""
        sys.stdout.write("\r>> " + " " * len(self.current_line) + "\r>> ")
        sys.stdout.flush()
        self.current_line = ""
        self.cursor_pos = 0

    def _show_history(self, direction: int):
        """Show history item (1 for next, -1 for previous)"""
        if not self.history:
            return

        if direction == -1:  # Up arrow - previous history
            if self.history_index == 0:
                # Save current line when first going to history
                self.temp_line = self.current_line
            if self.history_index < len(self.history):
                self.history_index += 1
                self.current_line = self.history[-self.history_index]
        elif direction == 1:  # Down arrow - next history
            if self.history_index > 1:
                self.history_index -= 1
                self.current_line = self.history[-self.history_index]
            elif self.history_index == 1:
                self.history_index = 0
                self.current_line = getattr(self, 'temp_line', '')

        self._clear_line()
        sys.stdout.write(self.current_line)
        sys.stdout.flush()
        self.cursor_pos = len(self.current_line)

    def readline(self, prompt: str = ">> ") -> str:
        """Read a line with enhanced editing capabilities"""
        # Fall back to plain input() when stdin is not a TTY (piped/scripted),
        # since termios/tty raw mode needs an interactive terminal.
        if not sys.stdin.isatty():
            try:
                line = input(prompt)
            except EOFError:
                raise KeyboardInterrupt
            line = line.strip()
            if line and (not self.history or self.history[-1] != line):
                self.history.append(line)
            self.history_index = 0
            return line

        sys.stdout.write(prompt)
        sys.stdout.flush()

        self.current_line = ""
        self.cursor_pos = 0

        while True:
            try:
                char = self._getch()

                # Handle special characters
                if char == '\x03':  # Ctrl+C
                    raise KeyboardInterrupt
                elif char == '\r' or char == '\n':  # Enter
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    line = self.current_line.strip()
                    if line and (not self.history or self.history[-1] != line):
                        self.history.append(line)
                    self.history_index = 0
                    return line
                elif char == '\x7f' or char == '\x08':  # Backspace/Delete
                    self._delete_char()
                elif char == '\x15':  # Ctrl+U - clear line
                    self._clear_line()
                elif char == '\x01':  # Ctrl+A - move to beginning
                    self._move_cursor(0)
                elif char == '\x05':  # Ctrl+E - move to end
                    self._move_cursor(len(self.current_line))
                elif char == '\x1b':  # Escape sequence (arrows, etc.)
                    # Read the next two characters
                    try:
                        char2 = self._getch()
                        char3 = self._getch()
                        if char2 == '[':
                            if char3 == 'A':  # Up arrow
                                self._show_history(-1)
                            elif char3 == 'B':  # Down arrow
                                self._show_history(1)
                            elif char3 == 'C':  # Right arrow
                                if self.cursor_pos < len(self.current_line):
                                    self._move_cursor(self.cursor_pos + 1)
                            elif char3 == 'D':  # Left arrow
                                if self.cursor_pos > 0:
                                    self._move_cursor(self.cursor_pos - 1)
                    except:
                        pass  # Incomplete escape sequence
                elif char >= ' ':  # Printable character
                    self._insert_char(char)

            except KeyboardInterrupt:
                sys.stdout.write("^C\n")
                sys.stdout.flush()
                raise


# ---------------------------------------------------------------------------
# Tool schemas advertised to the model (OpenAI-style function calling)
# ---------------------------------------------------------------------------

def _get_tools() -> List[Dict]:
    """Return the JSON schema for every tool the model can call."""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "Read a file's content. Use this when the user asks about a "
                    "file, config, log, or any file content."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file, absolute or relative to cwd",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": (
                    "Create a new file or overwrite an existing file with content."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file, absolute or relative to cwd",
                        },
                        "content": {
                            "type": "string",
                            "description": "Full new content of the file",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": (
                    "Edit an existing file by replacing a unique substring (before) "
                    "with new text (after). A backup is made automatically."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file, absolute or relative to cwd",
                        },
                        "before": {
                            "type": "string",
                            "description": "Exact existing text to find (must be unique)",
                        },
                        "after": {
                            "type": "string",
                            "description": "Replacement text",
                        },
                    },
                    "required": ["path", "before", "after"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": (
                    "List files and directories under a path (non-recursive by "
                    "default; pass recursive=true for a tree)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory to list, default '.'",
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "List recursively as a tree",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": (
                    "Run a shell command (bash -c) on this machine and return its "
                    "stdout/stderr and exit code. Use for server ops: service "
                    "status, logs, package installs, process checks, systemctl, "
                    "docker, disk usage, network, etc. Destructive or sudo "
                    "commands require the user's y/N approval before running."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to run",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Seconds to wait before killing (default 30)",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_interactive",
                "description": (
                    "Run an interactive terminal program (editor, pager, dialog, "
                    "top, etc.). Output is not captured; use for things that need "
                    "a TTY and user interaction."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to run interactively",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
    ]


def _truncate(text: str, limit: int) -> str:
    """Truncate text to limit chars, keeping whole lines, with a marker."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # Drop the possibly-partial last line for cleanliness.
    if "\n" in cut:
        cut = cut.rsplit("\n", 1)[0]
    return f"{cut}\n... [output truncated, {len(text) - len(cut)} more chars]"


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

        # Only load session if explicitly requested, otherwise create new session
        if session_name:
            self._load_or_create_session(session_name)
        else:
            # Create new session without loading previous one
            session_name = self._generate_session_name()
            self._create_new_session(session_name)

        # Initialize with system prompt
        self._initialize_system_prompt()

    def _generate_session_name(self) -> str:
        """Generate a session name based on the current project directory"""
        # Use the directory name as the session name
        dir_name = self.project_root.name

        # If we're in the home directory or root, use a generic name
        if dir_name == "" or str(self.project_root) == str(Path.home()):
            dir_name = "default"

        # Clean the session name (remove special characters)
        session_name = re.sub(r'[^\w\-_]', '_', dir_name)

        return session_name

    def _auto_create_session(self):
        """Automatically create or load session based on project directory"""
        session_name = self._generate_session_name()
        self._load_or_create_session(session_name)

    def _load_or_create_session(self, session_name: str):
        """Load existing session or create new one"""
        session_file = self.config.sessions_dir / f"{session_name}.json"

        if session_file.exists():
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)

                # Check if this session is from the same project
                context_data = session_data.get('context', {})
                saved_project_root = context_data.get('project_root', '')

                if saved_project_root == str(self.project_root):
                    # Load session context
                    self.session_context = SessionContext(
                        session_id=context_data.get('session_id', session_name),
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
                        recent_messages = saved_messages[-10:]  # Load last 10
                        for msg_data in recent_messages:
                            role = msg_data.get('role', 'user')
                            content = msg_data.get('content', '')
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

                    print(f"📂 Loaded session '{session_name}' ({len(saved_messages)} previous messages, will be saved on quit)")
                else:
                    # Different project, create new session
                    print(f"📝 New project detected, creating fresh session '{session_name}'")
                    self._create_new_session(session_name)

            except Exception as e:
                print(f"⚠️  Could not load session: {e}. Starting fresh session.")
                self._create_new_session(session_name)
        else:
            self._create_new_session(session_name)

    def _create_new_session(self, session_name: str):
        """Create a new session"""
        self.session_context = SessionContext(
            session_id=session_name,
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            message_count=0,
        )
        print(f"📝 Created new session '{session_name}' (will be saved on quit)")

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

        system_prompt = f"""You are Koder, a server operations and coding assistant running directly on this machine. You help set up, maintain, configure, and debug servers and code, and you execute actions via tools.

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
        session_info.append(f"  Session: {self.session_context.session_id}")
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
        MAX_HISTORY = 40
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

    def run(self):
        """Main interaction loop"""
        print("Koder - server ops & coding agent")
        print(f"Project: {self.project_root}")
        print(f"Model: {self.config.model}")
        print(f"API: {self.config.api_base}")
        if self.session_context:
            print(f"Session: {self.session_context.session_id} (auto-saved)")
        print("Type 'quit' to exit. Ctrl+C to interrupt.")
        print()

        while True:
            try:
                # Get user input with enhanced editing
                user_input = self.input_handler.readline().strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    if self.session_context:
                        self._save_session()
                        print(f"💾 Session '{self.session_context.session_id}' saved")
                    break

                if not user_input:
                    continue

                # Add user message with timestamp
                self.messages.append(Message(
                    role="user",
                    content=user_input,
                    timestamp=datetime.now().isoformat()
                ))

                response = self._call_llm(self.messages)
                choice = response.choices[0]
                if not choice:
                    print("Error: empty response from LLM")
                    continue

                # Tool execution loop: keep feeding results back to the model
                # until it answers without another tool call.
                tool_rounds = 0
                while True:
                    tool_calls = self._parse_tool_calls(response)
                    if not tool_calls:
                        break

                    tool_rounds += 1
                    if tool_rounds > 12:
                        print("⚠️  Too many tool rounds - stopping to avoid a loop.")
                        self.messages.append(Message(
                            role="user",
                            content="(system) Tool round limit reached (12). Stop calling tools and answer now.",
                            timestamp=datetime.now().isoformat()
                        ))
                        response = self._call_llm(self.messages)
                        break

                    print(f"\n🔧 Executing {len(tool_calls)} tool(s)...")

                    # Echo the assistant tool-call message so the model sees
                    # what arguments it used (required before tool results).
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

                    print("🤖 Getting next response...")
                    response = self._call_llm(self.messages)
                    if not response.choices or not response.choices[0]:
                        print("Error: empty response from LLM")
                        break

                # Get final response
                choice = response.choices[0]
                assistant_message = (choice.get('message') or {}).get('content') or ""
                if assistant_message:
                    self.messages.append(Message(
                        role="assistant",
                        content=assistant_message,
                        timestamp=datetime.now().isoformat()
                    ))
                    print(f"\n🤖 {assistant_message}")

                # Update token usage
                usage = response.usage
                self.total_tokens_used += usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)
                print(f"\n💰 Tokens: {usage.get('prompt_tokens', 0)} + {usage.get('completion_tokens', 0)} = "
                      f"{usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)} "
                      f"(session: {self.total_tokens_used})")

                # Auto-save session after every interaction
                if self.session_context:
                    self._save_session()

            except KeyboardInterrupt:
                if self.session_context:
                    self._save_session()
                    print(f"\n💾 Session '{self.session_context.session_id}' saved")
                print("Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                # Don't break on errors, continue the loop

    def _print_tool_result(self, tool_name: str, result: str):
        """Show a concise one-line summary of a tool result."""
        n_lines = result.count('\n') + 1
        preview = result.replace('\n', ' ')[:120]
        if tool_name == "run_shell":
            print(f"  💻 {tool_name}: {preview}")
        elif tool_name == "run_interactive":
            print(f"  🖥️  {tool_name}: {preview}")
        elif tool_name == "write_file":
            print(f"  ✏️  {tool_name}: {preview}")
        elif tool_name == "edit_file":
            print(f"  ✏️  {tool_name}: {preview}")
        elif tool_name == "list_files":
            print(f"  📁 {tool_name}: {n_lines} lines")
        else:
            print(f"  📄 {tool_name}: {n_lines} lines, {len(result)} bytes")


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(
        description='Koder - server ops & coding agent (OpenAI-compatible LLM)')
    parser.add_argument('--model', type=str,
                        help='Model name (default: KODER_MODEL or qwen3.8:27b)')
    parser.add_argument('--api-base', dest='api_base_cli', type=str,
                        help='OpenAI-compatible API base, e.g. http://localhost:11434/v1')
    parser.add_argument('--session', type=str,
                        help='Session name to load (default: fresh session per directory)')
    parser.add_argument('--cwd', type=str, default=None,
                        help='Working directory for tools (default: current dir)')

    args = parser.parse_args()

    # Config: CLI flag > KODER_API_BASE env > local Ollama default. API key is
    # optional because local endpoints (Ollama) usually need none.
    config = Config()
    if args.model:
        config.model = args.model
    if args.api_base_cli:
        config.api_base = args.api_base_cli
    if args.cwd:
        config.cwd = Path(args.cwd).expanduser()

    # Create and run agent
    agent = BasicCodingAgent(config, args.session)
    agent.run()


if __name__ == "__main__":
    main()
