# Blaster Usage Guide

Blaster is a single-file server operations & coding agent (stdlib only), named after the Transformers Autobot Blaster. This guide covers day-to-day usage against an OpenAI-compatible chat API.

## Prerequisites

- Python 3.7+
- An OpenAI-compatible endpoint — by default a local Ollama (`http://localhost:11434/v1`, no key needed). For remote endpoints, set `OPENAI_API_KEY`.

## Running Blaster

```bash
# Local Ollama (defaults)
python blaster.py

# Specific model / endpoint
python blaster.py --model qwen3.8:27b
python blaster.py --model gpt-4o --api-base https://openrouter.ai/api/v1

# Work in another directory
python blaster.py --cwd /srv/myapp

# Resume a named session; bare --session lists saved sessions
python blaster.py --session prod-setup
python blaster.py --session
```

Environment overrides: `BLASTER_MODEL`, `BLASTER_API_BASE`, `OPENAI_API_KEY`.

## Tools

The agent calls these tools itself to act on the machine:

- `read_file(path)` / `write_file(path, content)` / `edit_file(path, before, after)` — file operations; edits and overwrites create a `.bak` backup
- `list_files(path, recursive?)` — list a directory, optionally as a tree
- `run_shell(command, timeout?)` — run a shell command and return stdout/stderr + exit code (output capped)
- `run_interactive(command)` — run an interactive program (editor, top, dialog) on your terminal

## Safety

Destructive commands (`rm -rf`, `mkfs`, `dd`, `fdisk`, `kill -9`, `git push -f`, ...) and anything using `sudo` require y/N approval. Declined commands stay blocked for the session so the agent won't retry them.

## Sessions

Conversations auto-save after every turn to `~/.blaster/sessions/<name>.json` (default name = working directory). On resume the last 100 messages are restored and shown as a recap in the chat. The per-turn context window is capped at 100 messages.

## Configuration

Defaults live in the `Config` dataclass at the top of `blaster.py` — model, api_base, timeouts, output caps, sessions dir, etc.

For more detail, see `README.md` and the inline comments in `blaster.py`.
