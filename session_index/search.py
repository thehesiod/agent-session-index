#!/usr/bin/env python3
"""Search and filtering APIs for the unified local session index."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

try:
    from . import config
    from . import semantic
except ImportError:
    import config
    import semantic

RRF_K = 60
# Static embeddings collapse rare identifiers onto subwords, so narrower queries stay FTS-only
MIN_SEMANTIC_QUERY_TOKENS = 3
SNIPPET_MAX_DOC_CHARS = 200_000
SNIPPET_SCAN_CHARS = 2_000_000
EXCERPT_RADIUS = 60
VALID_SOURCES = ("claude", "codex")
SUBAGENT_MODES = ("include", "exclude", "only")
USAGE_GROUPINGS = {
    "model": "u.model",
    "source": "s.source",
    "project": "s.project_name",
    "session": "s.source || ':' || s.session_id",
    "agent": "s.agent_name",
    "day": "substr(s.start_time, 1, 10)",
}


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


def _excerpt(content: str, terms: list[str]) -> str:
    """Cheap stand-in for snippet() on documents too large to rescan."""
    lowered = content.lower()
    for term in terms:
        found = lowered.find(term)
        if found != -1:
            start = max(found - EXCERPT_RADIUS, 0)
            return content[start:found + len(term) + EXCERPT_RADIUS]
    return content[:EXCERPT_RADIUS * 2]


def _recency_factor(start_time: str | None, half_life: float,
                    weight: float) -> float:
    """Half-life decay applied as a bounded multiplier on the fused score.

    Multiplicative and capped so it reorders near-ties without burying a
    strongly-relevant old session under a weakly-relevant fresh one.
    """
    if not start_time or weight <= 0 or half_life <= 0:
        return 1.0
    try:
        started = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    except ValueError:
        return 1.0
    now = datetime.now(started.tzinfo) if started.tzinfo else datetime.now()
    age_days = max((now - started).total_seconds() / 86400.0, 0.0)
    return 1.0 + weight * (0.5 ** (age_days / half_life))


def _mark_superseded(results: list[dict]) -> list[dict]:
    """Flag older results whose project and topic a newer result repeats.

    Exact normalized topic match only - a topic string is already a model-written
    summary of what the session was about, so repeating one in the same project
    is a real signal. No fuzzy matching, so this never silently drops a result.
    """
    seen: dict[tuple[str, str], str] = {}
    ordered = sorted(
        results,
        key=lambda item: item.get("start_time") or "",
        reverse=True,
    )
    for result in ordered:
        project = (result.get("project_name") or "").strip().lower()
        for topic in result.get("topics") or []:
            key = (project, (topic.get("topic") or "").strip().lower())
            if not key[1]:
                continue
            if key in seen and seen[key] != result["session_id"]:
                result["superseded_by"] = seen[key]
                break
            seen.setdefault(key, result["session_id"])
    return results


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

    def _lexical(self, query: str, limit: int, source: str | None,
                 subagents: str, days: int | None) -> list[dict]:
        source_clause, source_params = self._source_condition(source)
        conditions = [
            "session_content MATCH ?", source_clause,
            self._subagent_condition(subagents),
        ]
        params: list = [_escape_fts_query(query), *source_params]
        if days:
            conditions.append("s.start_time >= ?")
            params.append((datetime.now() - timedelta(days=days)).isoformat())
        params.append(limit)
        statement = (
            "SELECT s.source, s.session_id, s.parent_session_id, s.agent_name, "
            "s.project_name, s.title, s.title_display, s.client, s.tags, "
            "s.exchange_count, s.start_time, s.duration_minutes, "
            "s.has_compaction, s.content_chars "
            "FROM session_content "
            "JOIN sessions s ON s.source = session_content.source "
            "AND s.session_id = session_content.session_id "
            "WHERE " + " AND ".join(conditions) + " ORDER BY rank LIMIT ?"
        )
        return [dict(row) for row in self.conn.execute(statement, params)]

    def _attach_snippets(self, query: str, results: list[dict]) -> list[dict]:
        """Snippet only what will be displayed, and only where it is cheap.

        snippet() rescans the whole document to locate the match, so its cost
        tracks document size - seconds each on the few multi-megabyte sessions.
        Those fall back to a literal scan of a bounded prefix instead.
        """
        if not results:
            return results
        match = _escape_fts_query(query)
        terms = [term.strip('"').lower() for term in match.split() if term]
        for result in results:
            if (result.get("content_chars") or 0) <= SNIPPET_MAX_DOC_CHARS:
                row = self.conn.execute(
                    "SELECT snippet(session_content, 2, '>>>', '<<<', '...', 40) "
                    "FROM session_content "
                    "WHERE session_content MATCH ? "
                    "AND source = ? AND session_id = ?",
                    (match, result["source"], result["session_id"]),
                ).fetchone()
                if row:
                    result["snippet"] = row[0]
                continue
            row = self.conn.execute(
                "SELECT substr(content, 1, ?) FROM session_content "
                "WHERE source = ? AND session_id = ?",
                (SNIPPET_SCAN_CHARS, result["source"], result["session_id"]),
            ).fetchone()
            if row:
                result["snippet"] = _excerpt(row[0] or "", terms)
        return results

    def _hydrate(self, source: str, session_id: str) -> dict | None:
        row = self.conn.execute("""
            SELECT s.source, s.session_id, s.parent_session_id, s.agent_name,
                   s.project_name, s.title, s.title_display, s.client, s.tags,
                   s.exchange_count, s.start_time, s.duration_minutes,
                   s.has_compaction, s.content_chars
            FROM sessions s WHERE s.source=? AND s.session_id=?
        """, (source, session_id)).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _passes_filters(row: dict, source: str | None, subagents: str,
                        days: int | None) -> bool:
        if source and row.get("source") != source:
            return False
        if subagents == "exclude" and row.get("parent_session_id"):
            return False
        if subagents == "only" and not row.get("parent_session_id"):
            return False
        if days:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            if (row.get("start_time") or "") < cutoff:
                return False
        return True

    def search(self, query: str, limit: int = 20,
               source: str | None = None,
               subagents: str = "include",
               semantic_search: bool = True,
               recency: bool = True,
               days: int | None = None,
               half_life: float | None = None) -> list[dict]:
        """FTS and vector hits fused by reciprocal rank, then recency-decayed."""
        pool = max(limit * 5, 50)
        lexical = self._lexical(query, pool, source, subagents, days)
        dense: list[tuple[str, str, float]] = []
        if semantic_search and len(query.split()) >= MIN_SEMANTIC_QUERY_TOKENS:
            index = semantic.SemanticIndex(self.conn)
            if index.enabled and semantic.available():
                try:
                    dense = index.search(query, pool)
                except sqlite3.Error:
                    dense = []

        scores: dict[tuple[str, str], float] = {}
        rows: dict[tuple[str, str], dict] = {}
        for rank, row in enumerate(lexical):
            key = (row["source"], row["session_id"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            rows[key] = row
        for rank, (hit_source, session_id, _) in enumerate(dense):
            key = (hit_source, session_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)

        weight = config.get_recency_weight() if recency else 0.0
        decay = (
            half_life if half_life is not None
            else config.get_recency_half_life()
        )
        fused = []
        for key, score in scores.items():
            row = rows.get(key) or self._hydrate(*key)
            if not row or not self._passes_filters(row, source, subagents, days):
                continue
            row["score"] = score * _recency_factor(
                row.get("start_time"), decay, weight
            )
            fused.append(row)
        fused.sort(key=lambda item: item["score"], reverse=True)
        return _mark_superseded(self._attach_snippets(query, [
            self._with_topics_dict(row) for row in fused[:limit]
        ]))

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

    def tool_detail(self, tool: str = "Bash", source: str | None = None,
                    project: str | None = None, days: int | None = None,
                    week: bool = False, subagents: str = "include",
                    limit: int = 25) -> list[dict]:
        """Token cost per command within one tool, e.g. bash by command."""
        conditions = ["d.tool_name = ?"]
        params: list = [tool]
        source_clause, source_params = self._source_condition(source)
        conditions.append(source_clause)
        conditions.append(self._subagent_condition(subagents))
        params.extend(source_params)
        if project:
            conditions.append("(s.project_name LIKE ? OR s.project LIKE ?)")
            params.extend([f"%{project}%", f"%{project}%"])
        if week or days:
            conditions.append("s.start_time >= ?")
            params.append(
                (datetime.now() - timedelta(days=days or 7)).isoformat()
            )
        params.append(limit)
        statement = (
            "SELECT d.detail AS grouping, NULL AS title, "
            "COUNT(DISTINCT s.source || ':' || s.session_id) AS sessions, "
            "SUM(d.use_count) AS calls, "
            "SUM(d.write_tokens) AS write_tokens, "
            "SUM(d.inject_tokens) AS inject_tokens, "
            "SUM(d.result_bytes) AS result_bytes "
            "FROM session_tool_detail d "
            "JOIN sessions s ON s.source = d.session_source "
            "AND s.session_id = d.session_id "
            "WHERE " + " AND ".join(conditions) + " "
            "GROUP BY grouping HAVING grouping IS NOT NULL AND grouping != '' "
            "ORDER BY inject_tokens DESC, calls DESC LIMIT ?"
        )
        return [dict(row) for row in self.conn.execute(statement, params)]

    def tool_tokens(self, tool: str | None = None, source: str | None = None,
                    project: str | None = None, days: int | None = None,
                    week: bool = False, subagents: str = "include",
                    limit: int = 25) -> list[dict]:
        """Token cost per tool, or per session when one tool is named."""
        conditions = []
        params: list = []
        source_clause, source_params = self._source_condition(source)
        conditions.append(source_clause)
        conditions.append(self._subagent_condition(subagents))
        params.extend(source_params)
        if tool:
            conditions.append("st.tool_name LIKE ?")
            params.append(f"%{tool}%")
        if project:
            conditions.append("(s.project_name LIKE ? OR s.project LIKE ?)")
            params.extend([f"%{project}%", f"%{project}%"])
        if week or days:
            conditions.append("s.start_time >= ?")
            params.append(
                (datetime.now() - timedelta(days=days or 7)).isoformat()
            )
        params.append(limit)
        grouping = "s.source || ':' || s.session_id" if tool else "st.tool_name"
        statement = (
            "SELECT " + grouping + " AS grouping, "
            "MAX(s.title_display) AS title, "
            "COUNT(DISTINCT s.source || ':' || s.session_id) AS sessions, "
            "SUM(st.use_count) AS calls, "
            "SUM(st.write_tokens) AS write_tokens, "
            "SUM(st.inject_tokens) AS inject_tokens, "
            "SUM(st.result_bytes) AS result_bytes "
            "FROM session_tools st "
            "JOIN sessions s ON s.source = st.session_source "
            "AND s.session_id = st.session_id "
            "WHERE " + " AND ".join(conditions) + " "
            "GROUP BY grouping HAVING grouping IS NOT NULL "
            "ORDER BY inject_tokens DESC, calls DESC LIMIT ?"
        )
        return [dict(row) for row in self.conn.execute(statement, params)]

    def usage(self, by: str = "model", source: str | None = None,
              project: str | None = None, days: int | None = None,
              week: bool = False, subagents: str = "include",
              limit: int = 25) -> list[dict]:
        if by not in USAGE_GROUPINGS:
            raise ValueError(f"Unknown grouping: {by}")
        conditions = []
        params: list = []
        source_clause, source_params = self._source_condition(source)
        conditions.append(source_clause)
        conditions.append(self._subagent_condition(subagents))
        params.extend(source_params)
        if project:
            conditions.append("(s.project_name LIKE ? OR s.project LIKE ?)")
            params.extend([f"%{project}%", f"%{project}%"])
        if week or days:
            conditions.append("s.start_time >= ?")
            params.append(
                (datetime.now() - timedelta(days=days or 7)).isoformat()
            )
        params.append(limit)
        statement = (
            "SELECT " + USAGE_GROUPINGS[by] + " AS grouping, "
            "MAX(s.title_display) AS title, "
            "COUNT(DISTINCT s.source || ':' || s.session_id) AS sessions, "
            "SUM(u.calls) AS calls, "
            "SUM(u.input_tokens) AS input_tokens, "
            "SUM(u.output_tokens) AS output_tokens, "
            "SUM(u.cache_write_tokens) AS cache_write_tokens, "
            "SUM(u.cache_write_1h_tokens) AS cache_write_1h_tokens, "
            "SUM(u.cache_write_5m_tokens) AS cache_write_5m_tokens, "
            "SUM(u.cache_read_tokens) AS cache_read_tokens, "
            "SUM(u.reasoning_tokens) AS reasoning_tokens "
            "FROM session_usage u "
            "JOIN sessions s ON s.source = u.session_source "
            "AND s.session_id = u.session_id "
            "WHERE " + " AND ".join(conditions) + " "
            "GROUP BY grouping HAVING grouping IS NOT NULL "
            "ORDER BY output_tokens DESC LIMIT ?"
        )
        return [dict(row) for row in self.conn.execute(statement, params)]

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
        return self._with_topics_dict(dict(row))

    def _with_topics_dict(self, result: dict) -> dict:
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
    if result.get("superseded_by"):
        meta.append(
            f"superseded by {result['superseded_by'].removeprefix('agent-')[:8]}"
        )
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
