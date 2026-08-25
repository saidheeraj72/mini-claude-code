# mini-claude-code — Build Plan

A local coding agent: Ollama for inference, Postgres for memory, Python REPL for the interface.

## Binding constraint: 8 GB RAM

Everything below is shaped by this. A 7B model at q4 is ~4.7 GB resident; add the embedding
model, Postgres, and an editor and you are swapping. Two consequences:

1. **Default to a 4B model.** Small models are bad at tool-calling, so the loop must be built
   to *expect* malformed output and correct it, not to assume compliance.
2. **The context window is a budget, not a given.** Ollama defaults `num_ctx` to 4096 and
   silently truncates older messages. This must be set explicitly or the agent will "forget"
   mid-task with no error.

### Models

| Role | Model | Size | Notes |
|---|---|---|---|
| Agent (default) | `qwen3:4b` | ~2.6 GB | Native tool support, comfortable headroom |
| Agent (stretch) | `qwen2.5-coder:7b-instruct-q4_K_M` | ~4.7 GB | Better code reasoning; expect swap pressure |
| Embeddings | `nomic-embed-text` | ~274 MB | 768-dim, fast, stays resident |

Make the agent model a config value so it is one env var to swap.

---

## Architecture

```
mcc/
├── cli.py            REPL entrypoint, slash commands, Ctrl-C handling
├── config.py         env-driven settings (model, num_ctx, db url, cwd)
├── agent.py          the loop: prompt -> stream -> tool calls -> results -> repeat
├── context.py        token budgeting + history compaction
├── permissions.py    allow/deny gating, session + global rules
├── ui.py             rich rendering: streaming text, tool cards, diffs
├── llm/
│   ├── ollama.py     streaming /api/chat client, tool spec marshalling
│   └── protocol.py   fallback tool protocol + tolerant parser
├── tools/
│   ├── base.py       Tool ABC; pydantic model -> JSON schema -> ollama tool spec
│   ├── fs.py         read_file, write_file, edit_file, list_dir
│   ├── search.py     grep, glob, semantic_search
│   ├── shell.py      run_bash (timeout, output cap, cwd pinning)
│   └── registry.py
└── db/
    ├── pool.py       psycopg3 connection pool
    ├── sessions.py   session / message / tool_call repositories
    └── index.py      file indexing + pgvector retrieval
```

**Dependencies:** `ollama`, `psycopg[binary,pool]`, `pydantic`, `rich`, `prompt_toolkit`, `tiktoken` (token counting only).

---

## Database schema

```sql
-- conversation state
sessions(id uuid pk, cwd text, model text, title text,
         created_at timestamptz, updated_at timestamptz, status text)

messages(id bigserial pk, session_id uuid fk cascade, seq int,
         role text,           -- user | assistant | tool | system | summary
         content text, tool_calls jsonb, token_count int, created_at timestamptz,
         unique(session_id, seq))

tool_calls(id bigserial pk, session_id uuid fk, message_id bigint fk,
           name text, args jsonb, result text, is_error bool,
           duration_ms int, approved_by text, created_at timestamptz)

-- persisted "always allow" decisions; session_id null = global
permissions(id bigserial pk, session_id uuid null, tool_name text,
            pattern text, decision text, created_at timestamptz)

-- codebase index
files(id bigserial pk, repo_root text, path text, sha256 text,
      mtime timestamptz, size int, lang text, indexed_at timestamptz,
      unique(repo_root, path))

chunks(id bigserial pk, file_id bigint fk cascade,
       start_line int, end_line int, content text, embedding vector(768))
```

Index `chunks` with HNSW (`vector_cosine_ops`). Reindex is content-hash gated: skip any file
whose `sha256` is unchanged, so re-indexing a repo is near-free.

---

## Phases

### Phase 0 — Environment
- `ollama serve` (`brew services start ollama`), pull `qwen3:4b` and `nomic-embed-text`
- `brew install pgvector`, `createdb mcc`, `CREATE EXTENSION vector`
- `uv init`, add deps, run `migrations/001_init.sql`

**Done when:** a one-off script round-trips a chat completion and a `SELECT 1` from Postgres.

### Phase 1 — Streaming REPL, no tools
Ollama client with streaming, `num_ctx` set explicitly, `rich` live rendering, Ctrl-C aborts
generation without killing the process.

**Done when:** you can hold a multi-turn conversation and interrupt mid-stream cleanly.

### Phase 2 — Tool framework + agent loop
The core. Each tool is a pydantic model + handler; schemas generate from the model so there
is one source of truth. The loop: send history → model emits tool calls → validate args →
execute → append results as `role: tool` → repeat until a text-only reply or a step cap (~15).

Tools: `read_file` (offset/limit), `write_file`, `edit_file` (exact string replace, must be
unique or it errors), `list_dir`, `grep`, `run_bash`.

**Small-model handling — the part that actually matters:**
- Validate every tool call against its pydantic schema. On failure, return the validation
  error *as the tool result* and let the model retry. Cap at 2 retries per call.
- If the model emits a tool call as prose instead of a structured call, `protocol.py` parses
  a fenced ```json block as a fallback.
- Keep the system prompt under ~600 tokens. Small models drown in long instructions.
- `edit_file` uses exact string matching, not diffs — 4B models cannot produce valid unified
  diffs reliably.

**Done when:** "read README.md and add a description section" works end to end.

### Phase 3 — Permissions
Classify tools read-only vs mutating. Mutating calls prompt: allow once / always allow this
tool / always allow this pattern / deny with feedback. Persist "always" choices to the
`permissions` table. `run_bash` gets an extra guard: a deny-list of destructive patterns
(`rm -rf`, `git push --force`, output redirection outside cwd) that always prompts.

Ship a `--yolo` flag that bypasses gating, since you will want it while testing.

### Phase 4 — Persistence + resume
Write each message and tool call to Postgres as it happens (not batched at session end, so a
crash keeps the history). Add `/sessions` to list, `--resume <id>` to rehydrate,
`--continue` for the most recent session in the current cwd. Auto-title sessions from the
first user message.

### Phase 5 — Semantic search
Index the repo: walk with gitignore respected, chunk by function/class where the language
allows and by line window otherwise (~40 lines, 10 overlap), embed with `nomic-embed-text`,
store in `chunks`. Expose `semantic_search(query, k)` as a tool.

Retrieval quality: combine vector similarity with a `grep` pass and merge the results.
Pure vector search on code underperforms; the hybrid is noticeably better.

### Phase 6 — Compaction + polish
When history exceeds ~70% of `num_ctx`, summarize the oldest turns into a single
`role: summary` message, persisted alongside the originals so nothing is lost. Add `/clear`,
`/compact`, `/model`, `/cost` (token counts), and tool-call cards in the UI.

---

## Risks

| Risk | Mitigation |
|---|---|
| 4B model loops or calls tools wrongly | Schema validation + error-feedback retry; hard step cap |
| Ollama silently truncates at 4096 ctx | Set `num_ctx` explicitly; assert it on startup |
| RAM exhaustion with 2 models loaded | `OLLAMA_MAX_LOADED_MODELS=1`, embed in a batch pass, not inline |
| `run_bash` damages the working tree | Permission gate + deny-list + cwd pinning; develop against a scratch repo |
| Embedding a large repo is slow | Content-hash gating; index in background on first run |

## Build order

Phases 1 → 2 give a genuinely usable agent. 3 → 4 make it safe and durable.
5 → 6 are the capability multipliers. Do not start Phase 5 before Phase 2 is solid —
retrieval on top of a loop that cannot reliably call tools just hides the real problem.
