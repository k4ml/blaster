# Blaster

A single-file server operations & coding agent using **only Python's standard library**, named after the Transformers Autobot Blaster. Blaster turns natural language into real actions on the host it runs on — inspect and edit files, run shell commands, install and configure services, check logs and process state — through any OpenAI-compatible chat API.

```bash
python blaster.py          # talks to a local Ollama by default
```

## Overview

Blaster is a CLI agent for running a server or working in a codebase from the terminal. You describe what you want in plain language; it decides which tools to call, executes them, and iterates until the job is done. Everything — the LLM client, tool execution, safety prompts, session persistence, and the terminal UI — lives in one stdlib-only Python file.

It's optimized for **local models** (Ollama / llama.cpp): lightweight context, no mandatory API key, and tool schemas sent only when the conversation may need them.

## Features

- **Server ops**: run shell commands, check `systemctl`/`docker`/processes, tail logs, install packages, manage services
- **Coding**: read, write, and edit files with automatic `.bak` backups
- **Safe by default**: file edits (`write_file`/`edit_file`) are blocked unless edit mode is enabled with `-w`/`--write`; on an interactive terminal a single edit can be approved on the spot, while scripted/non-interactive runs are hard-blocked
- **Safety prompts**: destructive commands (`rm -rf`, `mkfs`, `dd`, ...) and anything using `sudo` ask for y/N approval before running; declined commands stay blocked for the session
- **Local-first**: defaults to `http://localhost:11434/v1` (Ollama) with no API key required
- **Any OpenAI-compatible backend**: Ollama, llama.cpp, OpenRouter, OpenAI, vLLM, etc.
- **Persistent sessions**: named conversations (e.g. `daring-horizon`) auto-saved and resumable across directories
- **Project instructions (`AGENTS.md`)**: the nearest `AGENTS.md` from the working directory upward is injected into the system prompt; `--agents-md PATH` overrides with a specific file and `--agents-md none` disables it
- **Non-interactive mode**: `-p "prompt"` runs a single prompt and exits — scriptable from cron, CI, or other tools
- **Markdown rendering**: non-interactive answers are rendered with `glow` when it's installed and stdout is a TTY (`-x`/`--no-format` disables)
- **Live "thinking" indicator**: responses are streamed (`stream: true`), so long generations don't time out the connection. A spinner animates during prefill, then for reasoning models (qwen3, etc.) the reasoning streams via `delta.reasoning` and is shown as a collapsed `Thinking ... (N lines) [Ctrl+O to expand]` line that updates live — press Ctrl+O to expand it inline. The answer streams inline after. (All this is TTY-only; when piped, reasoning stays silent and the answer streams plain text.)
- **Live feedback**: `run_shell` shows the exact command before it runs, `edit_file` shows a colored diff of what changes
- **Enhanced terminal**: arrow-key history, cursor movement, Ctrl+A/E/U shortcuts, and multiline paste — a pasted block (bracketed paste, or a burst of input) becomes one message; falls back to plain `input()` when piped

## Tools

| Tool | Purpose |
|------|---------|
| `read_file(path)` | Read a file's content (100KB cap) |
| `write_file(path, content)` | Create/overwrite a file (backs up existing) |
| `edit_file(path, before, after)` | Replace a unique substring in a file (backs up existing) |
| `list_files(path, recursive?)` | List a directory, optionally as a recursive tree |
| `run_shell(command, timeout?)` | Run a shell command; returns stdout/stderr + exit code (output capped) |
| `run_interactive(command)` | Run an interactive program (editor, top, dialog) on your terminal |

## Installation

### Prerequisites

- Python 3.7+
- An OpenAI-compatible endpoint — a local one such as **Ollama** (`ollama serve`) or any remote provider

### Setup

1. **Copy `blaster.py`** anywhere you want to work (a project dir, a server, `~/.local/bin`).

2. **For local models (default)**: nothing to set. Just make sure Ollama is running and has a model pulled, e.g.

   ```bash
   ollama pull qwen3.8:27b
   ```

3. **For a remote provider** that needs a key:

   ```bash
   export OPENAI_API_KEY='your-key-here'
   ```

4. Optional: `chmod +x blaster.py`.

## Usage

```bash
# Local Ollama (defaults)
python blaster.py

# Pick a specific model / endpoint
python blaster.py --model qwen3.8:27b
python blaster.py --model gpt-4o --api-base https://openrouter.ai/api/v1

# Operate in a different directory (for running on a server elsewhere)
python blaster.py --cwd /srv/myapp

# Resume a named session (bare --session lists saved sessions)
python blaster.py --session prod-setup
python blaster.py --session

# Non-interactive: run one prompt and exit (scriptable)
python blaster.py -p "check disk usage and report the top 5 largest dirs"

# Disable markdown rendering / sessions
python blaster.py -p "..." -x
python blaster.py -s

# Enable edit mode (otherwise write_file/edit_file are blocked)
python blaster.py -w
python blaster.py -w -p "add a docstring to utils.py and fix the typo in config.py"
```

### Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `BLASTER_MODEL` or `qwen3.8:27b` | Model name |
| `--api-base` | `BLASTER_API_BASE` or `http://localhost:11434/v1` | OpenAI-compatible API base URL |
| `--cwd` | current directory | Working directory for all tools |
| `-p`/`--prompt` | — | Non-interactive mode: run a single prompt and exit |
| `-n`/`--max-iteration` | 50 | Max tool rounds per turn (safety limit to prevent loops) |
| `-t`/`--timeout` | 300 | LLM request timeout in seconds (per socket read; streaming keeps it alive during generation) |
| `-w`/`--write` | off | Enable edit mode: allow `write_file`/`edit_file` to modify files (blocked by default for safety) |
| `--session` | auto (random name) | Session name to load/resume; bare `--session` lists sessions |
| `-x`/`--no-format` | off | Disable `glow` markdown rendering in non-interactive mode |
| `-s`/`--no-session` | off | Disable sessions entirely (no load, create, or save) |
| `--agents-md` | auto-discover | `AGENTS.md` file to load (`--agents-md PATH` for a specific file, `--agents-md none` to disable) |

### Environment variables

| Variable | Purpose |
|----------|---------|
| `BLASTER_MODEL` | Default model |
| `BLASTER_API_BASE` | Default API base URL |
| `OPENAI_API_KEY` | Used when the endpoint requires auth (non-local) |

### Example interactions

```
>> what services are running?
🔧 Executing 1 tool(s)...
  💻 run_shell: UNIT LOAD ACTIVE SUB DESCRIPTION ... [exit code: 0]
🤖 nginx, docker, containerd and sshd are running. Want me to check
   any of them or look at their logs?
```

```
>> add a swapfile of 2G and enable it
🔧 Executing 1 tool(s)...

⚠️  This command is flagged as destructive pattern:
    dd if=/dev/zero of=/swapfile bs=1M count=2048
Run it? [y/N] y
  💻 run_shell: ... [exit code: 0]
🤖 Created /swapfile, formatted it as swap, and added it to /etc/fstab.
```

```
>> what does the nginx config in /etc/nginx/sites-enabled do?
🔧 Executing 1 tool(s)...
  📄 read_file: 47 lines
🤖 It's a reverse proxy that forwards example.com to a local Node app
   on port 3000, with websocket support...
```

If the user answers `N` to a safety prompt, the agent reports the block and offers a safer alternative instead of retrying.

## Configuration

Defaults live in the `Config` dataclass at the top of `blaster.py`:

```python
@dataclass
class Config:
    api_base: str = os.getenv("BLASTER_API_BASE", "http://localhost:11434/v1")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("BLASTER_MODEL", "qwen3.8:27b")
    max_tokens: int = 2000          # response length cap
    temperature: float = 0.2
    request_timeout: int = 300      # LLM HTTP timeout (seconds); -t/--timeout
    max_context_files: int = 20     # entries shown in the startup context
    max_file_size: int = 100_000    # read_file cap (bytes)
    max_output_chars: int = 40_000  # run_shell output cap (chars)
    max_iterations: int = 50        # tool round limit per turn
    sessions_dir: Path = Path.home() / ".blaster" / "sessions"
    cwd: Path = Path.cwd()
    format_markdown: bool = True    # render non-interactive answers with glow
    session_enabled: bool = True    # -s/--no-session disables all session I/O
    allow_edits: bool = False      # -w/--write enables write_file/edit_file
    non_interactive: bool = False  # set in -p mode; never prompts for edits
    agents_md_enabled: bool = True        # --agents-md none disables AGENTS.md
    agents_md_file: Optional[str] = None  # --agents-md PATH overrides discovery
    max_agents_md_chars: int = 20_000     # cap on AGENTS.md text injected
```

### Edit mode (safe by default)

Blaster is meant to run on production servers, so it does **not** modify files unless you opt in. By default `write_file` and `edit_file` are blocked:

- **`-w`/`--write`** enables edit mode for the whole run — edits proceed (with `.bak` backups), no prompts. Use this for scripted or unattended work that needs to change files.
- **Without `--write`, on an interactive terminal**, each attempted edit prompts `Allow this edit? [y/N]` so you can approve a one-off change without restarting.
- **Without `--write`, in non-interactive/non-TTY runs** (e.g. `-p` from cron/CI, or piped input), edits are **hard-blocked** — a scripted invocation cannot change files unless `--write` is passed. The agent is told the edit was blocked and should report this back rather than retry or work around it with shell redirections (`sed -i`, `echo >`, `tee`).

`run_shell` destructive/sudo commands keep their own y/N approval regardless of edit mode.

### Safety rules

Commands are checked against `DESTRUCTIVE_PATTERNS` (whole-word, so harmless uses like `grep rm` are not flagged) and a sudo pattern. Matches prompt for y/N approval before running:

- `rm -r` / `rm -f` / `rm -rf`
- `mkfs*`, `dd`, `mkswap`/`swapoff`, `parted`/`fdisk`/`sfdisk`
- `kill -9` / `pkill -9`
- `git push -f` / `git push --force`
- moving from `/`
- any command invoking `sudo`

A declined command is remembered for the session so the model cannot silently retry it.

### Project instructions (`AGENTS.md`)

On startup Blaster looks for an `AGENTS.md` file — the convention for project-level agent instructions — starting from the working directory and walking up its parents (bounded to 10 levels). The nearest one found is injected into the system prompt under a clear heading, so the model follows your project's coding standards, commands, and conventions. The startup banner shows `Rules: …/AGENTS.md` when one is loaded.

- Size is capped at `max_agents_md_chars` (default 20 KB); larger files are truncated with a marker.
- `--agents-md PATH` loads a specific file instead of discovering one (handy for shared/standard rule files); a missing path warns and loads nothing.
- `--agents-md none` disables loading entirely — useful when you run Blaster on a host where you don't want a repo-controlled file steering the agent.

## Sessions

Conversations auto-save after every turn to `~/.blaster/sessions/<session_id>.json`. Each session gets a unique id (the filename) and a human-friendly name — auto-generated as an adjective-noun pair (e.g. `daring-horizon`) unless you pass `--session NAME`. Sessions identify tasks, not directories: `--session` matches by name across every saved session, so you can resume one from any working directory. Bare `python blaster.py --session` lists saved sessions (name, id, message count, created/last-used). On resume the last 100 messages are restored and shown as a recap, and the per-turn context window is capped at 100 messages to protect the model's context. Use `-s`/`--no-session` to skip all session loading and saving.

## Architecture

The file is organized as:

1. **`Config` + safety patterns** — endpoint/model defaults and the approval regexes
2. **Dataclasses** — `Message`, `ToolCall`, `LLMResponse`, `SessionContext`, `BashToolResult`
3. **`EnhancedInput` / `_Spinner` / `_ThinkingPanel`** — cbreak line editor (history, cursor keys, Ctrl+A/E/U, multiline paste) that repaints by tracking the cursor's screen row (wrapped lines counted), with plain `input()` fallback when piped; a prefill spinner; and a collapsed reasoning panel (`Thinking ... (N lines) [Ctrl+O to expand]`, TTY-only)
4. **`_TOOL_SPECS` / `_get_tools()`** — the tool schemas advertised to the model; add a tool by appending one tuple
5. **`BasicCodingAgent`** — orchestration: session load/save, `AGENTS.md` discovery + system-prompt assembly, streaming LLM calls (`stream: true`, with live reasoning/answer output via `_read_stream`), message trimming, the tool-execution loop (with a configurable round guard, default 50), and the file/shell tool implementations
6. **`main()`** — CLI parsing (`--model`, `--api-base`, `--cwd`, `--session`, `-p`, `-n`, `-t`, `-w`, `-x`, `-s`, `--agents-md`), interactive loop, and non-interactive single-prompt mode

### Data flow

```
User input → message history → LLM (chat/completions) → tool calls?
                                                        ↓ yes
                ◄── assistant tool_call echo + tool results
                                                        ↓ no
                          final answer → printed + session saved
```

### Design notes

- **Stdlib only** — HTTP via `urllib`, processes via `subprocess`, no dependencies.
- **Local-model friendly** — tool schemas are only attached when the tail of the conversation may need them; long tool output is truncated; history is capped.
- **Streaming-tolerant** — parses both plain JSON and NDJSON streaming responses, so it works with servers that stream by default.

## Troubleshooting

**Connection refused to `http://localhost:11434/v1`**
Make sure Ollama is running (`ollama serve`) and reachable.

**"API key required for non-local endpoint"**
Set `OPENAI_API_KEY` (or pass an `--api-base` pointing at a local server).

**"File too large"**
Adjust `max_file_size` in `Config`, or use `run_shell` (e.g. `tail`, `head`) for big files.

**Model isn't calling tools reliably**
Prefer a model with solid function calling (e.g. `qwen3.8:27b`, `devstral`). Check `ollama list`; some small models handle tool use poorly.

**Session not resuming**
Sessions are matched by name across all saved sessions (not tied to a directory). Check the exact name with bare `python blaster.py --session`, then pass it with `--session NAME`.

## Development

The project deliberately stays stdlib-only (blaster itself is a single file; `tests.py` is separate). To extend it:

1. **Add a tool**: append a `(name, description, props, required)` tuple to `_TOOL_SPECS`, add an `elif` branch in `_execute_tool`, and implement the handler method.
2. **Tune safety**: edit `DESTRUCTIVE_PATTERNS` / `SUDO_PATTERN`.
3. **Change behavior**: adjust the system prompt in `_initialize_system_prompt` or the `Config` defaults.

### Tests

Run the stdlib-only suite (spins up in-process mock LLM servers, no network):

```bash
python tests.py
```

It covers the safe-by-default edit gate, streamed responses (including the short-timeout case), the live reasoning panel, `AGENTS.md` discovery/injection, the multiline input redraw (including wrapped lines), and HTTP parsing/fallbacks. Exits non-zero on failure, so it's CI-friendly.

## License

Provided as-is for personal and educational use. Respect your API provider's terms of service.
