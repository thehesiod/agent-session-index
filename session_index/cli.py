#!/usr/bin/env python3
"""Unified local CLI for Agent Session Index."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

try:
    from . import config
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
    "tools", "topics", "stats", "index", "search",
}


def _add_source(parser):
    parser.add_argument(
        "--source", choices=SOURCES,
        help="Limit the command to one transcript source",
    )


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

    command = subparsers.add_parser("search", help="Full-text search")
    command.add_argument("query")
    command.add_argument("-n", "--limit", type=int, default=20)
    command.add_argument("--context", action="store_true")
    _add_source(command)

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

    command = subparsers.add_parser("tools", help="Tool usage")
    command.add_argument("tool_name", nargs="?")
    _add_source(command)

    command = subparsers.add_parser("topics", help="Topic timeline")
    command.add_argument("session_id")
    _add_source(command)

    command = subparsers.add_parser("stats", help="Database overview")
    _add_source(command)

    command = subparsers.add_parser("index", help="Index local sessions")
    command.add_argument("--backfill", action="store_true")
    command.add_argument("--session", metavar="ID")
    command.add_argument("--claude-root", metavar="PATH")
    command.add_argument("--codex-root", metavar="PATH")
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
        )
        indexer.connect()
        try:
            if args.backfill:
                indexer.backfill_all(source=args.source)
            elif args.session:
                if not indexer.index_session(
                    session_id=args.session, source=args.source
                ):
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
                args.query, limit=args.limit, source=args.source
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
            results = searcher.recent(args.n, source=args.source)
            _print_results(f"📋 Last {len(results)} sessions", results)

        elif args.command == "find":
            results = searcher.find(
                client=args.client, tag=args.tag, tool=args.tool,
                agent=args.agent, date=args.date, week=args.week,
                days=args.days, project=args.project,
                exclude_project=args.exclude_project,
                has_compaction=True if args.compacted else None,
                limit=args.limit, source=args.source,
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

        elif args.command == "stats":
            stats = searcher.stats(args.source)
            print("\n📊 Database overview")
            print("═" * 40)
            print(f"  Source:    {args.source or 'all'}")
            print(f"  Sessions:  {stats.get('total_sessions', 0)}")
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
    finally:
        searcher.close()


if __name__ == "__main__":
    main()
