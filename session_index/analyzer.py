#!/usr/bin/env python3
"""
Agent session analyzer — local context retrieval and analytics.

Three capabilities on top of the session index:
  A. context   — read JSONL, extract full conversation exchanges around a match
  B. analytics — pure SQL aggregations (time per client, tool trends, etc.)
  C. synthesize — disabled so transcripts never leave the machine

Usage:
    python3 -m session_index.analyzer context <session_id> "search term"
    python3 -m session_index.analyzer analytics [--client X] [--week] [--month]
    python3 -m session_index.analyzer synthesize "query" [--limit 10]
"""

import re
import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

try:
    from . import config
    from .sources import extract_exchanges_for_source
    from .search import resume_command
except ImportError:
    import config
    from sources import extract_exchanges_for_source
    from search import resume_command

# ---------------------------------------------------------------------------
# A. Context retrieval — JSONL parsing
# ---------------------------------------------------------------------------

def extract_exchanges(session_path: str | Path, query: str = None,
                      limit: int = 10, max_chars: int = 1000,
                      source: str = "claude") -> list[dict]:
    """Extract exchanges using the parser for the indexed source."""
    return extract_exchanges_for_source(
        source, session_path, query=query, limit=limit, max_chars=max_chars
    )


def indexed_excerpts(conn: sqlite3.Connection, source: str, session_id: str,
                     query: str = None, limit: int = 10,
                     max_chars: int = 1000) -> list[str]:
    """Recover readable text for a deleted transcript from the index itself."""
    row = conn.execute(
        "SELECT content FROM session_content WHERE source=? AND session_id=?",
        (source, session_id),
    ).fetchone()
    if not row or not row["content"]:
        return []

    blocks = [block.strip() for block in row["content"].split("\n")]
    blocks = [block for block in blocks if block]
    if query:
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            pattern = None
        if pattern:
            blocks = [block for block in blocks if pattern.search(block)]
    return [block[:max_chars] for block in blocks[:limit]]


def get_context(session_id: str, query: str = None, limit: int = 10,
                db_path: Path = None, source: str = None) -> dict:
    """Get conversation context for a session.

    Returns dict with session info + matching exchanges.
    """
    if db_path is None:
        db_path = config.get_db_path()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    if ":" in session_id and not source:
        prefix, value = session_id.split(":", 1)
        if prefix in ("claude", "codex"):
            source, session_id = prefix, value
    clauses = ["session_id LIKE ?"]
    params = [f"{session_id}%"]
    if source:
        clauses.append("source=?")
        params.append(source)
    try:
        rows = conn.execute(
            "SELECT source, session_id, file_path, title_display, project_name, "
            "client, start_time, exchange_count, duration_minutes "
            f"FROM sessions WHERE {' AND '.join(clauses)} "
            "ORDER BY CASE WHEN session_id=? THEN 0 ELSE 1 END LIMIT 2",
            [*params, session_id],
        ).fetchall()

        if not rows:
            return {"error": f"Session not found: {session_id}"}
        if len(rows) > 1:
            return {
                "error": (
                    f"Session prefix is ambiguous: {session_id}. "
                    "Pass --source claude or --source codex."
                )
            }

        session_info = dict(rows[0])
        transcript_missing = not Path(session_info["file_path"]).exists()
        exchanges: list[dict] = []
        excerpts: list[str] = []
        if transcript_missing:
            excerpts = indexed_excerpts(
                conn, session_info["source"], session_info["session_id"],
                query=query, limit=limit,
            )
        else:
            exchanges = extract_exchanges(
                session_info["file_path"], query=query, limit=limit,
                source=session_info["source"],
            )
    finally:
        conn.close()

    return {
        "session": session_info,
        "query": query,
        "exchanges": exchanges,
        "excerpts": excerpts,
        "transcript_missing": transcript_missing,
        "total_matches": len(exchanges) + len(excerpts),
    }


def format_context(result: dict) -> str:
    """Format context result for CLI output."""
    if "error" in result:
        return result["error"]

    lines = []
    s = result["session"]
    title = s.get("title_display") or "(unnamed)"

    # Session header card
    lines.append(f"\n╭─── {title} {'─' * max(1, 44 - len(title))}")
    meta = []
    if s.get('start_time'):
        meta.append(s['start_time'][:10])
    if s.get('project_name'):
        meta.append(s['project_name'])
    if s.get('exchange_count'):
        meta.append(f"{s['exchange_count']} exchanges")
    if s.get('duration_minutes'):
        meta.append(f"{s['duration_minutes']}min")
    lines.append(f"│ {' · '.join(meta)}")
    lines.append(f"│ source: {s['source']}")
    lines.append(f"│ → {resume_command(s['source'], s['session_id'])}")
    lines.append(f"╰{'─' * 48}")

    if result.get("transcript_missing"):
        lines.append(
            "\n⚠ transcript file is gone — showing text recovered from the index"
        )
        if result["query"]:
            lines.append(f"\nMatching text for \"{result['query']}\":\n")
        else:
            lines.append(f"\nIndexed text ({result['total_matches']} shown):\n")
        if not result.get("excerpts"):
            lines.append("  (no text was stored for this session)")
        for block in result.get("excerpts") or []:
            lines.append(f"  │ {block}")
        return "\n".join(lines)

    if result["query"]:
        lines.append(f"\nMatching exchanges for \"{result['query']}\":\n")
    else:
        lines.append(f"\nAll exchanges ({result['total_matches']} shown):\n")

    for i, ex in enumerate(result["exchanges"], 1):
        ts = ex["timestamp"][:16] if ex["timestamp"] else ""
        try:
            dt = datetime.fromisoformat(ts)
            ts_display = dt.strftime("%b %d, %H:%M")
        except (ValueError, TypeError):
            ts_display = ts
        lines.append(f"  ┌─ {ts_display} {'─' * max(1, 40 - len(ts_display))}")
        lines.append(f"  │")

        # User message — first line gets emoji, rest indented
        user_lines = ex['user'].split('\n')
        for j, ul in enumerate(user_lines):
            if j == 0:
                lines.append(f"  │  🧑 {ul}")
            else:
                lines.append(f"  │     {ul}")

        lines.append(f"  │")

        # Assistant message
        asst_lines = ex['assistant'].split('\n')
        for j, al in enumerate(asst_lines):
            if j == 0:
                lines.append(f"  │  🤖 {al}")
            else:
                lines.append(f"  │     {al}")

        lines.append(f"  │")
        lines.append(f"  └{'─' * 44}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# B. Analytics — pure SQL
# ---------------------------------------------------------------------------

def analytics(client: str = None, project: str = None,
              week: bool = False, month: bool = False,
              db_path: Path = None, source: str = None) -> dict:
    """Run analytics queries against sessions.db. Returns structured dict."""
    if db_path is None:
        db_path = config.get_db_path()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    results = {"source": source or "all"}

    # Period filter
    period_clause = ""
    period_params = []
    period_label = "all time"
    if week:
        period_clause = "AND s.start_time >= ?"
        period_params = [(datetime.now() - timedelta(days=7)).isoformat()]
        period_label = "this week"
    elif month:
        period_clause = "AND s.start_time >= ?"
        period_params = [(datetime.now() - timedelta(days=30)).isoformat()]
        period_label = "this month"

    results["period"] = period_label

    # Client filter
    client_clause = ""
    client_params = []
    if client:
        client_clause = "AND s.client LIKE ?"
        client_params = [f"%{client}%"]

    # Project filter
    project_clause = ""
    project_params = []
    if project:
        project_clause = "AND (s.project_name LIKE ? OR s.project LIKE ?)"
        project_params = [f"%{project}%", f"%{project}%"]

    source_clause = "AND s.source = ?" if source else ""
    source_params = [source] if source else []
    base_where = (
        f"WHERE 1=1 {period_clause} {client_clause} "
        f"{project_clause} {source_clause}"
    )
    base_params = (
        period_params + client_params + project_params + source_params
    )

    # 1. Time per client
    rows = conn.execute(f"""
        SELECT s.client, COUNT(*) as sessions,
               SUM(s.duration_minutes) as total_minutes,
               ROUND(AVG(s.exchange_count), 1) as avg_exchanges
        FROM sessions s
        {base_where} AND s.client IS NOT NULL
        GROUP BY s.client ORDER BY total_minutes DESC
    """, base_params).fetchall()
    results["time_per_client"] = [dict(r) for r in rows]

    # 2. Session frequency (last 14 days, ignoring period filter)
    rows = conn.execute(f"""
        SELECT date(s.start_time) as day, COUNT(*) as sessions,
               SUM(s.duration_minutes) as minutes
        FROM sessions s
        WHERE s.start_time >= date('now', '-14 days')
        {"AND s.source=?" if source else ""}
        GROUP BY day ORDER BY day
    """, source_params).fetchall()
    results["daily_trend"] = [dict(r) for r in rows]

    # 3. Overall stats for period
    row = conn.execute(f"""
        SELECT COUNT(*) as total_sessions,
               SUM(duration_minutes) as total_minutes,
               ROUND(AVG(duration_minutes), 1) as avg_duration,
               ROUND(AVG(exchange_count), 1) as avg_exchanges,
               SUM(CASE WHEN has_compaction = 1 THEN 1 ELSE 0 END) as compacted_sessions
        FROM sessions s
        {base_where}
    """, base_params).fetchone()
    results["overview"] = dict(row)

    # 4. Top tools (period-aware)
    rows = conn.execute(f"""
        SELECT st.tool_name, SUM(st.use_count) as total,
               COUNT(DISTINCT st.session_source || ':' || st.session_id)
                   as session_count
        FROM session_tools st
        JOIN sessions s
          ON s.source = st.session_source AND s.session_id = st.session_id
        {base_where}
        GROUP BY st.tool_name ORDER BY total DESC LIMIT 15
    """, base_params).fetchall()
    results["top_tools"] = [dict(r) for r in rows]

    # 5. Tool trends: this week vs last week
    this_week_start = (datetime.now() - timedelta(days=7)).isoformat()
    last_week_start = (datetime.now() - timedelta(days=14)).isoformat()
    rows = conn.execute(f"""
        SELECT tool_name,
               SUM(CASE WHEN s.start_time >= ? THEN use_count ELSE 0 END) as this_week,
               SUM(CASE WHEN s.start_time >= ? AND s.start_time < ? THEN use_count ELSE 0 END) as last_week
        FROM session_tools st
        JOIN sessions s
          ON s.source = st.session_source AND s.session_id = st.session_id
        WHERE s.start_time >= ?
          {"AND s.source=?" if source else ""}
        GROUP BY tool_name
        HAVING this_week > 0 OR last_week > 0
        ORDER BY this_week DESC
        LIMIT 15
    """, (
        this_week_start, last_week_start, this_week_start, last_week_start,
        *source_params,
    )).fetchall()
    results["tool_trends"] = [dict(r) for r in rows]

    # 6. Most-discussed topics
    rows = conn.execute(f"""
        SELECT st.topic, COUNT(*) as mentions, st.source
        FROM session_topics st
        JOIN sessions s
          ON s.source = st.session_source AND s.session_id = st.session_id
        {base_where}
        GROUP BY st.topic
        ORDER BY mentions DESC
        LIMIT 20
    """, base_params).fetchall()
    results["top_topics"] = [dict(r) for r in rows]

    # 7. Sessions per project
    rows = conn.execute(f"""
        SELECT s.project_name, COUNT(*) as sessions,
               SUM(s.duration_minutes) as total_minutes
        FROM sessions s
        {base_where}
        GROUP BY s.project_name ORDER BY sessions DESC
    """, base_params).fetchall()
    results["by_project"] = [dict(r) for r in rows]

    conn.close()
    return results


def format_analytics(data: dict) -> str:
    """Format analytics dict as readable CLI output."""
    lines = []

    period = data.get("period", "all time")
    lines.append(
        f"\nSession analytics — {period} · source: {data.get('source', 'all')}"
    )
    lines.append("═" * 50)

    # Overview
    ov = data.get("overview", {})
    total = ov.get("total_sessions", 0)
    mins = ov.get("total_minutes") or 0
    hours = round(mins / 60, 1) if mins else 0
    lines.append(f"\n  📊 {total} sessions · {hours}h total · "
                  f"avg {ov.get('avg_duration') or 0}min/session · "
                  f"avg {ov.get('avg_exchanges') or 0} exchanges")

    # Time per client
    tpc = data.get("time_per_client", [])
    if tpc:
        lines.append(f"\n  ⏱  Time per client")
        lines.append(f"  {'─' * 46}")
        for r in tpc:
            hrs = round((r["total_minutes"] or 0) / 60, 1)
            lines.append(f"  {r['client']:25s}  {r['sessions']:>4d} sessions  "
                          f"{hrs:>6.1f}h  avg {r['avg_exchanges']} exchanges")

    # By project
    bp = data.get("by_project", [])
    if bp:
        lines.append(f"\n  📁 By project")
        lines.append(f"  {'─' * 46}")
        for r in bp:
            hrs = round((r["total_minutes"] or 0) / 60, 1)
            lines.append(f"  {r['project_name']:25s}  {r['sessions']:>4d} sessions  {hrs:>6.1f}h")

    # Daily trend
    dt = data.get("daily_trend", [])
    if dt:
        lines.append(f"\n  📈 Daily trend (last 14 days)")
        lines.append(f"  {'─' * 46}")
        for r in dt:
            mins = r["minutes"] or 0
            bar = "█" * min(int(mins / 15), 40)
            lines.append(f"  {r['day']}  {r['sessions']:>3d} sessions  "
                          f"{round(mins / 60, 1):>5.1f}h  {bar}")

    # Top tools
    tt = data.get("top_tools", [])
    if tt:
        lines.append(f"\n  🔧 Top tools")
        lines.append(f"  {'─' * 46}")
        for r in tt:
            lines.append(f"  {r['tool_name']:25s}  {r['total']:>6d} uses  "
                          f"({r['session_count']} sessions)")

    # Tool trends
    trends = data.get("tool_trends", [])
    if trends:
        lines.append(f"\n  📊 Tool trends (this week vs last)")
        lines.append(f"  {'─' * 46}")
        for r in trends:
            tw = r["this_week"] or 0
            lw = r["last_week"] or 0
            if lw > 0:
                pct = round((tw - lw) / lw * 100)
                arrow = "↑" if pct > 0 else ("↓" if pct < 0 else "→")
                change = f"{arrow} {abs(pct)}%"
            elif tw > 0:
                change = "NEW"
            else:
                change = ""
            lines.append(f"  {r['tool_name']:25s}  {tw:>5d} (was {lw:>5d})  {change}")

    # Top topics
    topics = data.get("top_topics", [])
    if topics:
        lines.append(f"\n  💬 Most discussed topics")
        lines.append(f"  {'─' * 46}")
        for r in topics[:10]:
            lines.append(f"  {r['mentions']:>3d}×  {r['topic']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# C. Cross-session synthesis policy
# ---------------------------------------------------------------------------

def synthesize(query: str, limit: int = 10, max_excerpt_chars: int = 2000,
               db_path: Path = None) -> dict:
    """Refuse network synthesis so local transcripts never leave the machine."""
    return {
        "error": (
            "Network synthesis is disabled: Agent Session Index never uploads "
            "session content. Search and retrieve local context, then synthesize "
            "within your current agent conversation."
        ),
        "sessions": [],
        "synthesis": None,
    }

def format_synthesis(result: dict) -> str:
    """Format synthesis result for CLI output."""
    if "error" in result and result.get("synthesis") is None:
        return result["error"]

    lines = []
    lines.append(f"\nCross-session synthesis — \"{result.get('query', '')}\"")
    lines.append("═" * 50)

    # Sources
    sources = result.get("sessions", [])
    if sources:
        lines.append(f"\n  📚 Sources ({len(sources)} sessions, "
                      f"{result.get('excerpt_count', 0)} with matching exchanges)\n")
        for s in sources:
            title = s.get("title") or "(unnamed)"
            if len(title) > 55:
                title = title[:52] + "..."
            lines.append(f"    {s['date']}  {title}")
            lines.append(
                f"             → {resume_command(s.get('source', 'claude'), s['session_id'])}"
            )

    # Synthesis
    if result.get("synthesis"):
        lines.append(f"\n  {'─' * 48}\n")
        lines.append(result["synthesis"])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Session analyzer")
    parser.add_argument("--db-path", help="Path to sessions.db (overrides config)")
    subparsers = parser.add_subparsers(dest="command")

    # context
    sp = subparsers.add_parser("context", help="Show conversation context for a session")
    sp.add_argument("session_id", help="Session ID (full or prefix)")
    sp.add_argument("query", nargs="?", default=None, help="Filter to matching exchanges")
    sp.add_argument("-n", "--limit", type=int, default=10, help="Max exchanges to show")
    sp.add_argument("--source", choices=("claude", "codex"))

    # analytics
    sp = subparsers.add_parser("analytics", help="Session analytics")
    sp.add_argument("--client", help="Filter by client")
    sp.add_argument("--project", help="Filter by project")
    sp.add_argument("--week", action="store_true", help="This week only")
    sp.add_argument("--month", action="store_true", help="This month only")
    sp.add_argument("--source", choices=("claude", "codex"))

    # synthesize
    sp = subparsers.add_parser(
        "synthesize", help="Explain the local-only synthesis policy"
    )
    sp.add_argument("query", help="Topic to synthesize across sessions")
    sp.add_argument("--limit", type=int, default=10, help="Max sessions to analyze")

    args = parser.parse_args()

    # Resolve db_path from CLI flag
    db_path = None
    if args.db_path:
        db_path = Path(args.db_path).expanduser()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    config.ensure_indexed(db_path)

    if args.command == "context":
        result = get_context(args.session_id, query=args.query, limit=args.limit,
                             db_path=db_path, source=args.source)
        print(format_context(result))

    elif args.command == "analytics":
        result = analytics(
            client=args.client, project=args.project,
            week=args.week, month=args.month,
            db_path=db_path, source=args.source,
        )
        print(format_analytics(result))

    elif args.command == "synthesize":
        result = synthesize(args.query, limit=args.limit, db_path=db_path)
        print(format_synthesis(result))


if __name__ == "__main__":
    main()
