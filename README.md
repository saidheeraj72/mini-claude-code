# mini-claude-code

A local coding agent. Ollama for inference, Postgres for memory, a terminal REPL for the
interface. No API keys, no network calls — everything runs on your machine.

```
› add a docstring to the chunker in mcc/db/index.py

  ● read_file  mcc/db/index.py
    ✓ 220 lines  8ms
  ● edit_file  mcc/db/index.py  (edit)
    ✓ line 84  3ms

Added a docstring explaining the overlap and boundary snapping.
```

## What it does

- **Agent loop** with 8 tools: `read_file`, `write_file`, `edit_file`, `list_dir`,
  `grep`, `glob`, `semantic_search`, `run_bash`
- **Permission gating** on every mutating call — allow once, always for the tool,
  always for a matching pattern, or deny. Decisions persist in Postgres.
- **Session persistence** — every message and tool call is written to Postgres as it
  happens, so `--resume` picks up exactly where you left off, even after a crash.
- **Semantic code search** over pgvector, blending embedding similarity with literal
  term matching.
- **Automatic compaction** when history approaches the context limit. Folded messages
  are flagged, never deleted.

## Setup

Requires Postgres 17+ and Ollama.

```bash
brew install pgvector                # if not already present
brew services start postgresql@17

ollama serve &
ollama pull qwen3:4b
ollama pull nomic-embed-text

createdb mcc
uv sync
uv run python -m mcc.db.migrate
```

## Use

```bash
uv run mcc                       # start a session in the current directory
uv run mcc --continue            # resume the latest session here
uv run mcc --resume 3f9a1c2b     # resume a specific session (8-char prefix is enough)
uv run mcc "what does agent.py do"   # one-shot, then exit
uv run mcc --yolo                # skip permission prompts
```

Slash commands: `/help` `/sessions` `/resume` `/new` `/index` `/compact` `/model`
`/think` `/cost` `/tools` `/exit`

Run `/index` once per repo to enable `semantic_search`. Re-running it is cheap — files
whose content hash is unchanged are skipped.

## Configuration

Copy `.env.example` and export what you need. The one that matters most:

**`MCC_NUM_CTX` (default 8192).** Two traps in one setting. Ollama defaults `num_ctx` to
4096 and silently drops older messages when you exceed it — no error, the agent just
forgets — so this client always sends the value explicitly. But setting it *too high* is
worse: qwen3:4b costs roughly 147 KB of KV cache per token, so 16k context is ~2.4 GB on
top of 2.5 GB of weights. On an 8 GB machine that swaps, and measured throughput collapsed
from 11.9 tok/s to **0.10 tok/s** — a 100x cliff. Raise it only if you have the RAM.

## Design notes

Everything here is shaped by running a 4B model on 8 GB of RAM.

**Tool calls are validated, not trusted.** Small models produce malformed arguments
routinely. Each tool's pydantic model generates its JSON schema *and* validates incoming
calls; a validation failure is fed back to the model as the tool result so it can correct
itself, capped at 2 retries per tool before the agent is told to try something else.

**Tool calls emitted as prose are recovered.** When the model ignores the tools API and
writes a fenced JSON block instead — common below 7B — `extract_inline_calls` parses it
out rather than treating it as chat.

**Edits are exact string replacement, never diffs.** A 4B model cannot reliably produce a
valid unified diff. `edit_file` requires `old_string` to appear exactly once and returns a
specific, actionable error when it doesn't.

**Edits that break a file are rejected before they land.** The most common 4B failure is
not a malformed tool call — it is a perfectly-formed `edit_file` whose `new_string` has
the wrong indentation. The string replacement succeeds and the file stops compiling. So
`edit_file` and `write_file` parse the result first (Python via `ast`, JSON via
`json.loads`) and refuse the write if a previously-valid file would become invalid,
handing the syntax error back to the model instead. A file that was *already* broken is
never blocked, so the model can still repair it.

**Paths are pinned to the workspace.** `ToolContext.resolve` rejects any path that escapes
the working directory, so a confused model cannot wander into `/etc`.

**Thinking is requested, not suppressed.** On qwen3, `think=False` does not stop the
model reasoning — it just stops Ollama separating it, so the reasoning lands in `content`
and gets persisted as the assistant's answer. Thinking is always requested so `content`
stays clean; `MCC_THINK` only controls whether you see it.

**Dangerous commands always prompt**, even under `--yolo` and even when a standing rule
would allow them: `rm -rf`, force pushes, hard resets, `sudo`, pipes to shell, writes
outside the workspace.

## Layout

```
mcc/
├── cli.py          REPL, slash commands, session resume
├── agent.py        the loop: stream → validate → gate → execute → persist
├── context.py      token budgeting and compaction splitting
├── permissions.py  rule matching and pattern generalisation
├── prompts.py      system prompt (short by design)
├── ui.py           streaming output, tool cards, approval prompts
├── llm/ollama.py   streaming client + inline tool-call recovery
├── tools/          base contract, fs, search, shell, registry
└── db/             pool, session/message repos, pgvector index, migrations
```

## Tests

```bash
uv run pytest
```

Covers the logic that must hold regardless of what the model does: path escaping, edit
uniqueness, danger detection, inline call parsing, compaction splitting, and permission
pattern generalisation.
