#!/usr/bin/env python3
"""Search and filtering APIs for the unified local session index."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

try:
    from . import config
except ImportError:
    import config

VALID_SOURCES = ("claude", "codex")
SUBAGENT_MODES = ("include", "exclude", "only")


def _escape_fts_query(query: str) -> str:
    """Quote FTS tokens while preserving explicitly quoted phrases."""
    if not query or not query.strip():
        return ""
    tokens = []
    index = 0
    chars = query.strip()
    while index < len(chars):
        if chars[index].isspace():
            index += 1
            continue
        if chars[index] == '"':
            end = chars.find('"', index + 1)
            if end == -1:
                tokens.append(f'"{chars[index + 1:]}"')
                break
            tokens.append(chars[index:end + 1])
            index = end + 1
        else:
            start = index
            while (
                index < len(chars)
                and not chars[index].isspace()
                and chars[index] != '"'
            ):
                index += 1
            tokens.append(f'"{chars[start:index]}"')
    return " ".join(tokens)


def resume_command(source: str, session_id: str) -> str:
    if source == "codex":
        return f"codex resume {session_id}"
    return f"claude --resume {session_id}"


class SessionSearch:
    def __init__(self, db_path: Path = None):
        self.db_path = Path(db_path) if db_path else config.get_db_path()
        self.conn: sqlite3.Connection | None = None

    def connect(self):
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        columns = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(sessions)")
        }
        if columns and "source" not in columns:
            self.conn.close()
            try:
                from .indexer import SessionIndexer
            except ImportError:
                from indexer import SessionIndexer
            indexer = SessionIndexer(self.db_path)
            indexer.connect()
            indexer.close()
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    @staticmethod
    def _source_condition(source: str | None, alias: str = "s"):
        if source and source not in VALID_SOURCES:
            raise ValueError(f"Unknown source: {source}")
        return (f"{alias}.source = ?", [source]) if source else ("1=1", [])

    @staticmethod
    def _subagent_condition(subagents: str = "include", alias: str = "s") -> str:
        if subagents not in SUBAGENT_MODES:
            raise ValueError(f"Unknown subagent mode: {subagents}")
        if subagents == "exclude":
            return f"{alias}.parent_session_id IS NULL"
        if subagents == "only":
            return f"{alias}.parent_session_id IS NOT NULL"
        return "1=1"

    def search(self, query: str, limit: int = 20,
               source: str | None = None,
               subagents: str = "include") -> list[dict]:
        source_clause, source_params = self._source_condition(source)
        subagent_clause = self._subagent_condition(subagents)
        rows = self.conn.execute(f"""
            SELECT s.source, s.session_id, s.parent_session_id, s.agent_name,
                   s.project_name, s.title,
                   s.title_display, s.client, s.tags, s.exchange_count,
                   s.start_time, s.duration_minutes, s.has_compaction,
                   snippet(session_content, 2, '>>>', '<<<', '...', 40)
                       AS snippet
            FROM session_content
            JOIN sessions s
              ON s.source = session_content.source
             AND s.session_id = session_content.session_id
            WHERE session_content MATCH ? AND {source_clause}
              AND {subagent_clause}
            ORDER BY rank
            LIMIT ?
        """, [_escape_fts_query(query), *source_params, limit]).fetchall()
        return [self._with_topics(row) for row in rows]

    def find(
        self,
        client: str = None,
        tag: str = None,
        tool: str = None,
        agent: str = None,
        date: str = None,
        week: bool = False,
        days: int = None,
        project: str = None,
        exclude_project: str = None,
        has_compaction: bool = None,
        limit: int = 20,
        source: str | None = None,
        subagents: str = "include",
    ) -> list[dict]:
        conditions = []
        params = []
        source_clause, source_params = self._source_condition(source)
        conditions.append(source_clause)
        conditions.append(self._subagent_condition(subagents))
        params.extend(source_params)
        if client:
            conditions.append("s.client LIKE ?")
            params.append(f"%{client}%")
        if tag:
            conditions.append("s.tags LIKE ?")
            params.append(f"%{tag}%")
        if tool:
            conditions.append("""
                EXISTS (
                    SELECT 1 FROM session_tools st
                    WHERE st.session_source=s.source
                      AND st.session_id=s.session_id
                      AND st.tool_name LIKE ?
                )
            """)
            params.append(f"%{tool}%")
        if agent:
            conditions.append("""
                EXISTS (
                    SELECT 1 FROM session_agents sa
                    WHERE sa.session_source=s.source
                      AND sa.session_id=s.session_id
                      AND sa.agent_name LIKE ?
                )
            """)
            params.append(f"%{agent}%")
        if project:
            conditions.append("(s.project_name LIKE ? OR s.project LIKE ?)")
            params.extend([f"%{project}%", f"%{project}%"])
        if date:
            conditions.append("s.start_time LIKE ?")
            params.append(f"{date}%")
        if week:
            conditions.append("s.start_time >= ?")
            params.append((datetime.now() - timedelta(days=7)).isoformat())
        if days:
            conditions.append("s.start_time >= ?")
            params.append((datetime.now() - timedelta(days=days)).isoformat())
        if exclude_project:
            conditions.append(
                "NOT (s.project_name LIKE ? OR s.project LIKE ?)"
            )
            params.extend([
                f"%{exclude_project}%", f"%{exclude_project}%"
            ])
        if has_compaction is not None:
            conditions.append("s.has_compaction = ?")
            params.append(int(has_compaction))

        params.append(limit)
        rows = self.conn.execute(f"""
            SELECT s.source, s.session_id, s.parent_session_id, s.agent_name,
                   s.project_name, s.title,
                   s.title_display, s.client, s.tags, s.exchange_count,
                   s.start_time, s.duration_minutes, s.has_compaction
            FROM sessions s
            WHERE {' AND '.join(conditions)}
            ORDER BY s.start_time DESC
            LIMIT ?
        """, params).fetchall()
        return [self._with_topics(row) for row in rows]

    def resolve_session(self, session_id: str,
                        source: str | None = None) -> dict | None:
        if ":" in session_id and not source:
            prefix, value = session_id.split(":", 1)
            if prefix in VALID_SOURCES:
                source, session_id = prefix, value
        source_clause, params = self._source_condition(source)
        rows = self.conn.execute(f"""
            SELECT * FROM sessions s
            WHERE {source_clause} AND s.session_id LIKE ?
            ORDER BY CASE WHEN s.session_id=? THEN 0 ELSE 1 END
            LIMIT 2
        """, [*params, f"{session_id}%", session_id]).fetchall()
        if len(rows) != 1:
            return None
        return dict(rows[0])

    def topics(self, session_id: str,
               source: str | None = None) -> list[dict]:
        resolved = self.resolve_session(session_id, source)
        if not resolved:
            return []
        rows = self.conn.execute("""
            SELECT topic, captured_at, exchange_number, source
            FROM session_topics
            WHERE session_source=? AND session_id=?
            ORDER BY captured_at
        """, (resolved["source"], resolved["session_id"])).fetchall()
        return [dict(row) for row in rows]

    def recent(self, n: int = 10, source: str | None = None,
               subagents: str = "include") -> list[dict]:
        return self.find(limit=n, source=source, subagents=subagents)

    def stats(self, source: str | None = None) -> dict:
        try:
            from .indexer import SessionIndexer
        except ImportError:
            from indexer import SessionIndexer
        indexer = SessionIndexer(self.db_path)
        indexer.conn = self.conn
        return indexer.get_stats(source)

    def tools_usage(self, tool_name: str = None, limit: int = 20,
                    source: str | None = None) -> list[dict]:
        conditions = []
        params = []
        if source:
            conditions.append("st.session_source=?")
            params.append(source)
        if tool_name:
            conditions.append("st.tool_name LIKE ?")
            params.append(f"%{tool_name}%")
            where = "WHERE " + " AND ".join(conditions)
            params.append(limit)
            rows = self.conn.execute(f"""
                SELECT s.source, s.session_id, s.title, s.title_display,
                       s.start_time, st.tool_name, st.use_count
                FROM session_tools st
                JOIN sessions s
                  ON s.source=st.session_source AND s.session_id=st.session_id
                {where}
                ORDER BY st.use_count DESC
                LIMIT ?
            """, params).fetchall()
        else:
            where = (
                "WHERE " + " AND ".join(conditions) if conditions else ""
            )
            params.append(limit)
            rows = self.conn.execute(f"""
                SELECT st.tool_name, SUM(st.use_count) AS total,
                       COUNT(DISTINCT st.session_source || ':' || st.session_id)
                           AS session_count
                FROM session_tools st
                {where}
                GROUP BY st.tool_name
                ORDER BY total DESC
                LIMIT ?
            """, params).fetchall()
        return [dict(row) for row in rows]

    def _with_topics(self, row: sqlite3.Row) -> dict:
        result = dict(row)
        result["topics"] = self._get_topics(
            result["source"], result["session_id"]
        )
        return result

    def _get_topics(self, source: str, session_id: str) -> list[dict]:
        rows = self.conn.execute("""
            SELECT topic, source
            FROM session_topics
            WHERE session_source=? AND session_id=?
            ORDER BY captured_at
            LIMIT 10
        """, (source, session_id)).fetchall()
        return [dict(row) for row in rows]


def format_result(result: dict, show_topics: bool = True) -> str:
    lines = []
    source = result.get("source", "claude")
    # Every subagent id starts with "agent-", leaving only 2 distinguishing chars
    short_id = result["session_id"].removeprefix("agent-")[:8]
    title = (
        result.get("title_display") or result.get("title") or "(unnamed)"
    )
    if len(title) > 70:
        title = title[:67] + "..."
    label = f"{source}/subagent" if result.get("parent_session_id") else source
    lines.append(f"  ◆ [{label}] {short_id} · {title}")
    meta = []
    if result.get("agent_name"):
        meta.append(f"agent {result['agent_name']}")
    if result.get("start_time"):
        meta.append(result["start_time"][:10])
    if result.get("project_name"):
        meta.append(result["project_name"])
    if result.get("client"):
        meta.append(result["client"])
    if result.get("exchange_count"):
        meta.append(f"{result['exchange_count']} exchanges")
    if result.get("duration_minutes"):
        meta.append(f"{result['duration_minutes']}min")
    if result.get("has_compaction"):
        meta.append("compacted")
    if meta:
        lines.append(f"    {' · '.join(meta)}")
    if result.get("snippet"):
        snippet = (
            result["snippet"].replace(">>>", "").replace("<<<", "")
            .replace("\n", " ")[:120]
        )
        lines.append(f'    "{snippet}"')
    if show_topics and result.get("topics"):
        topics = [topic["topic"] for topic in result["topics"][:5]]
        if topics:
            lines.append(f"    topics: {' → '.join(topics)}")
    if result.get("tags"):
        lines.append(f"    [{result['tags']}]")
    # A subagent transcript is not resumable; its parent conversation is
    resume_id = result.get("parent_session_id") or result["session_id"]
    lines.append(f"    → {resume_command(source, resume_id)}")
    return "\n".join(lines)


def main():
    # Keep the legacy session-search entrypoint while sharing the unified CLI.
    try:
        from .cli import main as cli_main
    except ImportError:
        from cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
