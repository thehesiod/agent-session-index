#!/usr/bin/env python3
"""Unified local CLI for Agent Session Index."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

try:
    from . import config
    from . import semantic
    from .analyzer import (
        analytics,
        format_analytics,
        format_context,
        format_synthesis,
        get_context,
        synthesize,
    )
    from .search import SessionSearch, format_result
except ImportError:
    import config
    import semantic
    from analyzer import (
        analytics,
        format_analytics,
        format_context,
        format_synthesis,
        get_context,
        synthesize,
    )
    from search import SessionSearch, format_result

SOURCES = ("claude", "codex")
SUBCOMMANDS = {
    "context", "analytics", "synthesize", "recent", "find",
    "tools", "topics", "stats", "index", "search", "usage", "embed",
}


def _add_source(parser):
    parser.add_argument(
        "--source", choices=SOURCES,
        help="Limit the command to one transcript source",
    )


def _add_subagents(parser):
    parser.add_argument(
        "--subagents", choices=("include", "exclude", "only"),
        default="include",
        help="Whether subagent transcripts take part (default: include)",
    )


def _add_ranking(parser):
    parser.add_argument(
        "--no-semantic", action="store_true",
        help="Skip the vector layer and rank on FTS alone",
    )
    parser.add_argument(
        "--no-recency", action="store_true",
        help="Rank purely on relevance, ignoring session age",
    )
    parser.add_argument(
        "--half-life", type=float, metavar="DAYS",
        help="Recency half-life in days (default: 90)",
    )
    parser.add_argument(
        "--days", type=int, metavar="N",
        help="Only search sessions from the last N days",
    )


def run_embed(db_path: Path, rebuild: bool = False, limit: int | None = None,
              source: str | None = None):
    """Backfill the vector layer over already-indexed sessions."""
    reason = semantic.unavailable_reason()
    if reason:
        print(f"Semantic layer unavailable: {reason}", file=sys.stderr)
        raise SystemExit(1)
    conn, index = semantic.open_index(db_path)
    if not index.enabled:
        print(
            "sqlite-vec is not loadable - "
            "pip install 'agent-session-index[semantic]'",
            file=sys.stderr,
        )
        raise SystemExit(1)
    index.ensure_schema(semantic.model_dims())
    conditions = ["c.content IS NOT NULL", "s.prose_chars > 0"]
    params: list = []
    if source:
        conditions.append("s.source = ?")
        params.append(source)
    if not rebuild:
        conditions.append(
            "NOT EXISTS (SELECT 1 FROM session_chunks k "
            "WHERE k.source = s.source AND k.session_id = s.session_id)"
        )
    statement = (
        "SELECT s.source, s.session_id, s.prose_chars, c.content "
        "FROM sessions s JOIN session_content c "
        "ON c.source = s.source AND c.session_id = s.session_id "
        "WHERE " + " AND ".join(conditions) + " ORDER BY s.start_time DESC"
    )
    if limit:
        statement += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(statement, params).fetchall()
    total_chunks = 0
    for done, row in enumerate(rows, start=1):
        total_chunks += index.index_session(
            row["source"], row["session_id"],
            (row["content"] or "")[:row["prose_chars"]],
        )
        if done % 50 == 0:
            conn.commit()
            print(f"  {done}/{len(rows)} sessions, {total_chunks} chunks")
    conn.commit()
    stats = index.stats()
    print(
        f"Embedded {len(rows)} sessions ({total_chunks} chunks). "
        f"Index now holds {stats['chunks']} chunks "
        f"across {stats['sessions']} sessions."
    )
    conn.close()


def _print_inline_context(result: dict, query: str, db_path: Path):
    context = get_context(
        result["session_id"], query=query, limit=3, db_path=db_path,
        source=result["source"],
    )
    for block in context.get("excerpts") or []:
        print(f"    │ 📄 {block[:250]}")
    for exchange in context.get("exchanges", []):
        timestamp = exchange["timestamp"][:16] if exchange["timestamp"] else ""
        try:
            display = datetime.fromisoformat(timestamp).strftime("%b %d, %H:%M")
        except (ValueError, TypeError):
            display = timestamp
        print(f"    ┌─ {display} {'─' * max(1, 36 - len(display))}")
        user = exchange["user"][:250].replace("\n", " ")
        assistant = exchange["assistant"][:250].replace("\n", " ")
        print(f"    │ 🧑 {user}")
        print(f"    │ 🤖 {assistant}")
        print(f"    └{'─' * 42}")


def _tokens(value: int | None) -> str:
    value = value or 0
    for limit, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if value >= limit:
            return f"{value / limit:.1f}{suffix}"
    return str(value)


def _bytes(value: int | None) -> str:
    value = value or 0
    for limit, suffix in ((1 << 30, "GB"), (1 << 20, "MB"), (1 << 10, "KB")):
        if value >= limit:
            return f"{value / limit:.1f}{suffix}"
    return f"{value}B"


def _usage_label(row: dict, per_session: bool) -> str:
    label = str(row["grouping"] or "unknown")
    if per_session:
        short = label.split(":")[-1].removeprefix("agent-")[:8]
        return f"{short} {row.get('title') or ''}".strip()
    return label


def _format_usage(rows: list[dict], by: str) -> str:
    width = max((len(str(row["grouping"] or "")) for row in rows), default=8)
    width = min(max(width, 8), 46)
    lines = [
        f"\n💠 Token usage by {by}\n",
        f"  {'':{width}}  {'calls':>7}  {'in':>8}  {'out':>8}"
        f"  {'cache w':>8}  {'cache r':>9}",
    ]
    totals = dict.fromkeys(
        ("calls", "input_tokens", "output_tokens",
         "cache_write_tokens", "cache_read_tokens"), 0
    )
    for row in rows:
        label = _usage_label(row, by == "session" and bool(row.get("title")))
        for key in totals:
            totals[key] += row.get(key) or 0
        lines.append(
            f"  {label[:width]:{width}}  {row['calls'] or 0:>7,}"
            f"  {_tokens(row['input_tokens']):>8}"
            f"  {_tokens(row['output_tokens']):>8}"
            f"  {_tokens(row['cache_write_tokens']):>8}"
            f"  {_tokens(row['cache_read_tokens']):>9}"
        )
    lines.append(
        f"  {'TOTAL':{width}}  {totals['calls']:>7,}"
        f"  {_tokens(totals['input_tokens']):>8}"
        f"  {_tokens(totals['output_tokens']):>8}"
        f"  {_tokens(totals['cache_write_tokens']):>8}"
        f"  {_tokens(totals['cache_read_tokens']):>9}"
    )
    return "\n".join(lines)


def _format_tool_tokens(rows: list[dict], tool: str | None,
                        command_of: str | None = None) -> str:
    width = max((len(_usage_label(row, bool(tool))) for row in rows), default=8)
    width = min(max(width, 10), 46)
    if command_of:
        heading = f"{command_of} command"
    elif tool:
        heading = f"tool {tool}, by session"
    else:
        heading = "tool"
    lines = [
        f"\n💠 Token cost by {heading}\n",
        f"  {'':{width}}  {'calls':>7}  {'write':>8}  {'inject':>9}"
        f"  {'result':>9}",
    ]
    totals = dict.fromkeys(
        ("calls", "write_tokens", "inject_tokens", "result_bytes"), 0
    )
    for row in rows:
        for key in totals:
            totals[key] += row.get(key) or 0
        lines.append(
            f"  {_usage_label(row, bool(tool))[:width]:{width}}"
            f"  {row['calls'] or 0:>7,}"
            f"  {_tokens(row['write_tokens']):>8}"
            f"  {_tokens(row['inject_tokens']):>9}"
            f"  {_bytes(row['result_bytes']):>9}"
        )
    lines.append(
        f"  {'TOTAL':{width}}  {totals['calls']:>7,}"
        f"  {_tokens(totals['write_tokens']):>8}"
        f"  {_tokens(totals['inject_tokens']):>9}"
        f"  {_bytes(totals['result_bytes']):>9}"
    )
    return "\n".join(lines)


def _print_results(title: str, results: list[dict]):
    print(f"\n{title}\n")
    for result in results:
        print(format_result(result))
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="sessions",
        description="Search and analyze local Claude and Codex sessions",
        usage='sessions "query" | sessions <command> [options]',
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Path to sessions.db (overrides config)",
    )
    subparsers = parser.add_subparsers(dest="command")

    command = subparsers.add_parser("search", help="Hybrid keyword + vector search")
    command.add_argument("query")
    command.add_argument("-n", "--limit", type=int, default=20)
    command.add_argument("--context", action="store_true")
    _add_source(command)
    _add_subagents(command)
    _add_ranking(command)

    command = subparsers.add_parser("context", help="Conversation context")
    command.add_argument("session_id", help="Session ID, prefix, or source:id")
    command.add_argument("query", nargs="?", default=None)
    command.add_argument("-n", "--limit", type=int, default=10)
    _add_source(command)

    command = subparsers.add_parser("analytics", help="Session analytics")
    command.add_argument("--client")
    command.add_argument("--project")
    command.add_argument("--week", action="store_true")
    command.add_argument("--month", action="store_true")
    _add_source(command)

    command = subparsers.add_parser(
        "synthesize",
        help="Explain local-only synthesis (no transcript uploads)",
    )
    command.add_argument("query")
    command.add_argument("--limit", type=int, default=10)

    command = subparsers.add_parser("recent", help="Recent sessions")
    command.add_argument("n", nargs="?", type=int, default=10)
    _add_source(command)
    _add_subagents(command)

    command = subparsers.add_parser("find", help="Filter sessions")
    command.add_argument("--client")
    command.add_argument("--tag")
    command.add_argument("--tool")
    command.add_argument("--agent")
    command.add_argument("--date")
    command.add_argument("--week", action="store_true")
    command.add_argument("--days", type=int)
    command.add_argument("--project")
    command.add_argument("--exclude-project")
    command.add_argument("--compacted", action="store_true")
    command.add_argument("-n", "--limit", type=int, default=20)
    _add_source(command)
    _add_subagents(command)

    command = subparsers.add_parser("tools", help="Tool usage")
    command.add_argument("tool_name", nargs="?")
    _add_source(command)

    command = subparsers.add_parser("topics", help="Topic timeline")
    command.add_argument("session_id")
    _add_source(command)

    command = subparsers.add_parser("usage", help="Token usage")
    command.add_argument(
        "--by",
        choices=(
            "model", "source", "project", "session", "agent", "day", "tool",
            "command",
        ),
        default="model",
    )
    command.add_argument(
        "--tool", metavar="NAME",
        help="Break one tool down by session (implies --by tool)",
    )
    command.add_argument("--project")
    command.add_argument("--week", action="store_true")
    command.add_argument("--days", type=int)
    command.add_argument("-n", "--limit", type=int, default=25)
    _add_source(command)
    _add_subagents(command)

    command = subparsers.add_parser("stats", help="Database overview")
    _add_source(command)

    command = subparsers.add_parser("index", help="Index local sessions")
    command.add_argument("--backfill", action="store_true")
    command.add_argument("--session", metavar="ID", action="append")
    command.add_argument("--file", metavar="PATH", action="append")
    command.add_argument("--claude-root", metavar="PATH")
    command.add_argument("--codex-root", metavar="PATH")
    command.add_argument(
        "--no-embed", action="store_true",
        help="Skip the vector layer while indexing",
    )
    _add_source(command)

    command = subparsers.add_parser(
        "embed", help="Build or refresh the semantic vector layer"
    )
    command.add_argument(
        "--rebuild", action="store_true",
        help="Re-embed every session, not just those with no vectors",
    )
    command.add_argument("-n", "--limit", type=int, help="Stop after N sessions")
    _add_source(command)

    raw_args = sys.argv[1:]
    if (
        raw_args
        and raw_args[0] not in SUBCOMMANDS
        and not raw_args[0].startswith("-")
    ):
        query_parts = []
        flags = []
        in_flags = False
        for argument in raw_args:
            if argument.startswith("-"):
                in_flags = True
            (flags if in_flags else query_parts).append(argument)
        sys.argv = [
            sys.argv[0], "search", " ".join(query_parts), *flags
        ]

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    db_path = (
        Path(args.db_path).expanduser()
        if args.db_path else config.get_db_path()
    )
    if args.command == "embed":
        run_embed(db_path, rebuild=args.rebuild, limit=args.limit,
                  source=args.source)
        return

    if args.command == "index":
        try:
            from .indexer import SessionIndexer
        except ImportError:
            from indexer import SessionIndexer
        indexer = SessionIndexer(
            db_path=db_path,
            projects_dir=Path(args.claude_root).expanduser()
            if args.claude_root else None,
            codex_sessions_dir=Path(args.codex_root).expanduser()
            if args.codex_root else None,
            embed=not args.no_embed,
        )
        indexer.connect()
        try:
            if args.backfill:
                indexer.backfill_all(source=args.source)
            elif args.session or args.file:
                targets = [{"session_id": s} for s in args.session or []]
                targets += [{"file_path": f} for f in args.file or []]
                failed = sum(
                    not indexer.index_session(source=args.source, **target)
                    for target in targets
                )
                print(f"Reindexed {len(targets) - failed}/{len(targets)} sessions")
                if failed:
                    raise SystemExit(1)
            else:
                stats = indexer.index_incremental(source=args.source)
                print(
                    f"Incremental: {stats['indexed']} new/updated, "
                    f"{stats['unchanged']} unchanged, "
                    f"{stats['errors']} errors"
                )
        finally:
            indexer.close()
        return

    config.ensure_indexed(db_path)

    if args.command == "context":
        print(format_context(get_context(
            args.session_id, query=args.query, limit=args.limit,
            db_path=db_path, source=args.source,
        )))
        return

    if args.command == "analytics":
        print(format_analytics(analytics(
            client=args.client, project=args.project,
            week=args.week, month=args.month,
            db_path=db_path, source=args.source,
        )))
        return

    if args.command == "synthesize":
        print(format_synthesis(synthesize(
            args.query, limit=args.limit, db_path=db_path
        )))
        return

    searcher = SessionSearch(db_path)
    searcher.connect()
    try:
        if args.command == "search":
            results = searcher.search(
                args.query, limit=args.limit, source=args.source,
                subagents=args.subagents,
                semantic_search=not args.no_semantic,
                recency=not args.no_recency,
                days=args.days,
                half_life=args.half_life,
            )
            if not results:
                print(f"No results for: {args.query}")
                return
            print(f'\n🔍 {len(results)} results for "{args.query}"\n')
            for result in results:
                print(format_result(result))
                if args.context:
                    _print_inline_context(result, args.query, db_path)
                print()

        elif args.command == "recent":
            results = searcher.recent(
                args.n, source=args.source, subagents=args.subagents
            )
            _print_results(f"📋 Last {len(results)} sessions", results)

        elif args.command == "find":
            results = searcher.find(
                client=args.client, tag=args.tag, tool=args.tool,
                agent=args.agent, date=args.date, week=args.week,
                days=args.days, project=args.project,
                exclude_project=args.exclude_project,
                has_compaction=True if args.compacted else None,
                limit=args.limit, source=args.source,
                subagents=args.subagents,
            )
            if not results:
                print("No sessions match those filters.")
                return
            _print_results(f"📋 {len(results)} sessions", results)

        elif args.command == "tools":
            results = searcher.tools_usage(
                args.tool_name, source=args.source
            )
            label = args.source or "all sources"
            if args.tool_name:
                print(f"\n🔧 [{label}] sessions using '{args.tool_name}'\n")
                for result in results:
                    title = (
                        result.get("title_display")
                        or result.get("title")
                        or "(unnamed)"
                    )
                    print(
                        f"  ◆ [{result['source']}] "
                        f"{result['session_id'][:8]} · "
                        f"{result['tool_name']} ×{result['use_count']}  {title}"
                    )
            else:
                print(f"\n🔧 Top tools — {label}\n")
                for result in results:
                    print(
                        f"  {result['tool_name']:25s} "
                        f"{result['total']:>6d} uses "
                        f"({result['session_count']} sessions)"
                    )

        elif args.command == "topics":
            session = searcher.resolve_session(
                args.session_id, args.source
            )
            if not session:
                print(
                    "No unique session found. Use a longer ID or --source."
                )
                return
            topics = searcher.topics(
                session["session_id"], session["source"]
            )
            if not topics:
                print(
                    f"No topics recorded for "
                    f"{session['source']}:{session['session_id'][:8]}"
                )
                return
            print(
                f"\n💬 [{session['source']}] topic timeline "
                f"({len(topics)} entries)\n"
            )
            for topic in topics:
                timestamp = (
                    topic["captured_at"][:16] if topic["captured_at"] else ""
                )
                exchange = (
                    f" (exchange {topic['exchange_number']})"
                    if topic["exchange_number"] else ""
                )
                print(f"  [{topic['source']:20s}] {timestamp}{exchange}")
                print(f"                       {topic['topic']}\n")

        elif args.command == "usage" and args.by == "command":
            rows = searcher.tool_detail(
                tool=args.tool or "Bash", source=args.source,
                project=args.project, days=args.days, week=args.week,
                subagents=args.subagents, limit=args.limit,
            )
            if not rows:
                print("No command detail recorded. Run: sessions index --backfill")
                return
            print(_format_tool_tokens(rows, None, args.tool or "Bash"))

        elif args.command == "usage" and (args.tool or args.by == "tool"):
            rows = searcher.tool_tokens(
                tool=args.tool, source=args.source, project=args.project,
                days=args.days, week=args.week, subagents=args.subagents,
                limit=args.limit,
            )
            if not rows:
                print("No tool usage recorded. Run: sessions index --backfill")
                return
            print(_format_tool_tokens(rows, args.tool))

        elif args.command == "usage":
            rows = searcher.usage(
                by=args.by, source=args.source, project=args.project,
                days=args.days, week=args.week, subagents=args.subagents,
                limit=args.limit,
            )
            if not rows:
                print("No usage recorded. Run: sessions index --backfill")
                return
            print(_format_usage(rows, args.by))

        elif args.command == "stats":
            stats = searcher.stats(args.source)
            print("\n📊 Database overview")
            print("═" * 40)
            print(f"  Source:    {args.source or 'all'}")
            print(f"  Sessions:  {stats.get('total_sessions', 0)}"
                  f" ({stats.get('total_subagents', 0)} subagent)")
            print(f"  Topics:    {stats.get('total_topics', 0)}")
            print(f"  Tools:     {stats.get('total_tools', 0)} distinct")
            print(f"  Agents:    {stats.get('total_agents', 0)} distinct")
            if stats.get("by_source"):
                print("\n  By source")
                for source, count in stats["by_source"].items():
                    print(f"  {source:25s}  {count:>5d}")
            date_range = stats.get("date_range", {})
            if date_range.get("earliest"):
                print(
                    f"  Range:     {date_range['earliest']} → "
                    f"{date_range['latest']}"
                )
            if stats.get("by_project"):
                print("\n  📁 By project")
                print(f"  {'─' * 36}")
                for name, count in list(stats["by_project"].items())[:10]:
                    print(f"  {name:25s}  {count:>5d}")
            if stats.get("top_tools"):
                print("\n  🔧 Top tools")
                print(f"  {'─' * 36}")
                for name, count in list(stats["top_tools"].items())[:10]:
                    print(f"  {name:25s}  {count:>5d}")
            print()
    finally:
        searcher.close()


if __name__ == "__main__":
    main()
