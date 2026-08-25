"""REPL entrypoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.table import Table
from rich.text import Text

from mcc.agent import Agent
from mcc.config import CONFIG, Config
from mcc.context import total_tokens
from mcc.db import sessions as db
from mcc.db.pool import close_pool, init_pool
from mcc.llm.ollama import OllamaClient
from mcc.permissions import PermissionManager
from mcc.ui import ConsoleEvents, ask_permission, banner, console, DIM, print_markdown

HELP = """\
  /help              show this
  /sessions          list recent sessions in this directory
  /resume <id>       switch to another session
  /new               start a fresh session
  /index             (re)index this codebase for semantic search
  /compact           summarise history now
  /clear             start a fresh session, same directory
  /model [name]      show or switch the model
  /think             toggle model thinking output
  /cost              token usage for this session
  /tools             list available tools
  /exit              quit
"""


def preflight(cfg: Config) -> bool:
    """Verify Ollama is reachable and the configured models exist."""
    client = OllamaClient(cfg)
    try:
        models = client.available_models()
    except Exception as e:
        console.print(Text(f"Cannot reach Ollama at {cfg.ollama_host}: {e}", style="red"))
        console.print(Text("Start it with:  ollama serve", style=DIM))
        return False

    def have(name: str) -> bool:
        return any(m == name or m.split(":")[0] == name.split(":")[0] for m in models)

    ok = True
    if not have(cfg.model):
        console.print(Text(f"Model {cfg.model} not pulled.", style="red"))
        console.print(Text(f"  ollama pull {cfg.model}", style=DIM))
        ok = False
    if not have(cfg.embed_model):
        console.print(
            Text(f"Embedding model {cfg.embed_model} not pulled -- semantic "
                 f"search will be unavailable.", style="yellow")
        )
        console.print(Text(f"  ollama pull {cfg.embed_model}", style=DIM))
    return ok


def show_sessions(cfg: Config) -> None:
    rows = db.list_sessions(str(cfg.cwd))
    if not rows:
        console.print(Text("  no sessions here yet", style=DIM))
        return
    table = Table(box=None, pad_edge=False, header_style=DIM)
    table.add_column("id"); table.add_column("updated"); table.add_column("msgs")
    table.add_column("title")
    for r in rows:
        table.add_row(
            str(r["id"])[:8],
            r["updated_at"].strftime("%m-%d %H:%M"),
            str(r["n_messages"]),
            (r["title"] or "(untitled)")[:60],
        )
    console.print(table)


def resolve_session(prefix: str) -> str | None:
    """Accept an 8-char prefix as well as a full uuid."""
    for row in db.list_sessions(None, limit=200):
        if str(row["id"]).startswith(prefix):
            return str(row["id"])
    return None


def run_index(agent: Agent) -> None:
    console.print(Text("  indexing…", style=DIM))
    seen = {"n": 0}

    def progress(rel: str) -> None:
        seen["n"] += 1
        if seen["n"] % 5 == 0:
            console.print(Text(f"    {rel}", style=DIM))

    try:
        stats = agent.build_index(progress)
    except Exception as e:
        console.print(Text(f"  index failed: {e}", style="red"))
        return
    console.print(
        Text(f"  indexed {stats['indexed']} files "
             f"({stats['skipped']} unchanged, {stats['chunks']} chunks)", style="green")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mcc", description="mini claude code")
    parser.add_argument("prompt", nargs="*", help="run one turn and exit")
    parser.add_argument("--resume", metavar="ID", help="resume a session by id prefix")
    parser.add_argument("--continue", dest="cont", action="store_true",
                        help="resume the latest session in this directory")
    parser.add_argument("--model", help="override the model")
    parser.add_argument("--yolo", action="store_true", help="skip permission prompts")
    parser.add_argument("--think", action="store_true", help="show model thinking")
    parser.add_argument("--cwd", help="workspace directory (default: current)")
    args = parser.parse_args(argv)

    cfg = CONFIG
    if args.model:
        cfg.model = args.model
    if args.cwd:
        cfg.cwd = Path(args.cwd).resolve()
    cfg.yolo = args.yolo
    if args.think:
        cfg.think = True

    if not preflight(cfg):
        return 1

    try:
        init_pool(cfg.database_url)
    except Exception as e:
        console.print(Text(f"Cannot connect to Postgres ({cfg.database_url}): {e}",
                           style="red"))
        return 1

    try:
        session_id, resumed = _open_session(cfg, args)
        if session_id is None:
            return 1

        events = ConsoleEvents(show_thinking=cfg.think)
        perms = PermissionManager(session_id, yolo=cfg.yolo, asker=ask_permission)
        agent = Agent(cfg, session_id, perms, events)

        if args.prompt:
            agent.run_turn(" ".join(args.prompt))
            return 0

        banner(cfg, session_id, resumed)
        return repl(cfg, agent)
    finally:
        close_pool()


def _open_session(cfg: Config, args) -> tuple[str | None, bool]:
    if args.resume:
        sid = resolve_session(args.resume)
        if not sid:
            console.print(Text(f"No session matching {args.resume!r}", style="red"))
            return None, False
        return sid, True
    if args.cont:
        sid = db.latest_session(str(cfg.cwd))
        if sid:
            return sid, True
        console.print(Text("No previous session here; starting a new one.", style=DIM))
    return db.create_session(str(cfg.cwd), cfg.model), False


def repl(cfg: Config, agent: Agent) -> int:
    hist = Path.home() / ".mcc_history"
    session = PromptSession(history=FileHistory(str(hist)))

    while True:
        try:
            line = session.prompt("› ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        if not line:
            continue

        if line.startswith("/"):
            action = handle_command(line, cfg, agent)
            if action == "quit":
                return 0
            if action == "handled":
                continue

        try:
            agent.run_turn(line)
        except KeyboardInterrupt:
            console.print(Text("\n  interrupted", style=DIM))
        except Exception as e:
            console.print(Text(f"  error: {type(e).__name__}: {e}", style="red"))
        console.print()


def switch_session(agent: Agent, session_id: str) -> None:
    """Point the agent, its tool context, and its permission rules at a session."""
    agent.session_id = session_id
    agent.ctx.session_id = session_id
    agent.permissions.session_id = session_id
    agent.permissions.reload()


def handle_command(line: str, cfg: Config, agent: Agent) -> str:
    parts = line.split()
    cmd, rest = parts[0], parts[1:]

    if cmd in ("/exit", "/quit", "/q"):
        return "quit"

    if cmd == "/help":
        console.print(HELP)
    elif cmd == "/sessions":
        show_sessions(cfg)
    elif cmd in ("/new", "/clear"):
        sid = db.create_session(str(cfg.cwd), cfg.model)
        switch_session(agent, sid)
        console.print(Text(f"  new session {sid[:8]}", style=DIM))
    elif cmd == "/resume":
        if not rest:
            console.print(Text("  usage: /resume <id>", style="yellow"))
        else:
            sid = resolve_session(rest[0])
            if not sid:
                console.print(Text(f"  no session matching {rest[0]!r}", style="red"))
            else:
                switch_session(agent, sid)
                console.print(Text(f"  resumed {sid[:8]}", style=DIM))
    elif cmd == "/index":
        run_index(agent)
    elif cmd == "/compact":
        if agent.maybe_compact(force=True):
            console.print(Text("  compacted", style="green"))
        else:
            console.print(Text("  nothing to compact", style=DIM))
    elif cmd == "/model":
        if rest:
            cfg.model = rest[0]
            console.print(Text(f"  model → {cfg.model}", style=DIM))
        else:
            console.print(Text(f"  {cfg.model}", style=DIM))
    elif cmd == "/think":
        cfg.think = not cfg.think
        agent.events.show_thinking = cfg.think
        console.print(Text(f"  thinking {'on' if cfg.think else 'off'}", style=DIM))
    elif cmd == "/cost":
        msgs = db.load_messages(agent.session_id)
        est = total_tokens([{"role": m["role"], "content": m["content"]} for m in msgs])
        pct = 100 * est / cfg.num_ctx
        console.print(Text(
            f"  {len(msgs)} messages  ~{est} tokens  {pct:.0f}% of {cfg.num_ctx} ctx\n"
            f"  last call: {agent.last_usage['prompt']} in / "
            f"{agent.last_usage['completion']} out", style=DIM))
    elif cmd == "/tools":
        from mcc.tools.registry import ALL_TOOLS
        for t in ALL_TOOLS:
            flag = "!" if t.mutating else " "
            console.print(Text(f"  {flag} {t.name}", style=DIM))
    else:
        console.print(Text(f"  unknown command {cmd}; /help for the list", style="yellow"))
        return "handled"
    return "handled"


if __name__ == "__main__":
    sys.exit(main())
