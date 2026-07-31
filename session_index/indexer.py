#!/usr/bin/env python3
"""Core SQLite/FTS indexer for local Claude and Codex sessions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from . import config
    from . import semantic
    from .sources import build_adapters
except ImportError:
    import config
    import semantic
    from sources import build_adapters

SCHEMA_VERSION = 3
VALID_SOURCES = ("claude", "codex")


class SessionIndexer:
    def __init__(
        self,
        db_path: Path = None,
        projects_dir: Path = None,
        codex_sessions_dir: Path = None,
        source_configs: dict | None = None,
        embed: bool = True,
    ):
        self.db_path = Path(db_path) if db_path else config.get_db_path()
        self.embed_enabled = embed
        # False means "not yet resolved"; None means "resolved to unavailable"
        self._semantic: object = False
        overrides = dict(source_configs or {})
        if projects_dir:
            overrides["claude"] = {
                **(overrides.get("claude") or {}),
                "root": str(projects_dir),
                "enabled": True,
            }
        if codex_sessions_dir:
            overrides["codex"] = {
                **(overrides.get("codex") or {}),
                "root": str(codex_sessions_dir),
                "enabled": True,
            }
        self.source_configs = config.get_source_configs(overrides)
        self.project_name_map = config.get_project_names()
        self.clients = config.get_clients()
        self.adapters = build_adapters(
            self.source_configs, self.project_name_map, self.clients
        )
        # Compatibility for integrations that still inspect this attribute.
        self.projects_dir = Path(
            self.source_configs["claude"]["root"]
        ).expanduser()
        self.conn: sqlite3.Connection | None = None

    def connect(self):
        """Open the database, migrate older schemas, and ensure FTS exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def _table_exists(self, name: str) -> bool:
        return bool(self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name=?", (name,)
        ).fetchone())

    def _columns(self, table: str) -> set[str]:
        if not self._table_exists(table):
            return set()
        return {
            row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")
        }

    def _create_schema(self):
        """Create the current schema or migrate a Claude-only database."""
        columns = self._columns("sessions")
        if columns and "source" not in columns:
            self._migrate_v1()
        else:
            self._create_v2_tables()
            # Support databases created by early multi-source development
            # snapshots without requiring a destructive rebuild.
            columns = self._columns("sessions")
            if "cwd" not in columns:
                self.conn.execute("ALTER TABLE sessions ADD COLUMN cwd TEXT")
            if "metadata_json" not in columns:
                self.conn.execute(
                    "ALTER TABLE sessions ADD COLUMN metadata_json TEXT"
                )
            if "parent_session_id" not in columns:
                self.conn.execute(
                    "ALTER TABLE sessions ADD COLUMN parent_session_id TEXT"
                )
            if "agent_name" not in columns:
                self.conn.execute(
                    "ALTER TABLE sessions ADD COLUMN agent_name TEXT"
                )
            if "prose_chars" not in columns:
                self.conn.execute(
                    "ALTER TABLE sessions ADD COLUMN prose_chars INTEGER DEFAULT 0"
                )
            if "content_chars" not in columns:
                self.conn.execute(
                    "ALTER TABLE sessions ADD COLUMN content_chars INTEGER DEFAULT 0"
                )
            if "cache_write_5m_tokens" not in self._columns("session_usage"):
                self.conn.execute(
                    "ALTER TABLE session_usage "
                    "ADD COLUMN cache_write_5m_tokens INTEGER DEFAULT 0"
                )
            tool_columns = self._columns("session_tools")
            if "write_tokens" not in tool_columns:
                self.conn.execute(
                    "ALTER TABLE session_tools "
                    "ADD COLUMN write_tokens INTEGER DEFAULT 0"
                )
            if "inject_tokens" not in tool_columns:
                self.conn.execute(
                    "ALTER TABLE session_tools "
                    "ADD COLUMN inject_tokens INTEGER DEFAULT 0"
                )
            if "result_bytes" not in tool_columns:
                self.conn.execute(
                    "ALTER TABLE session_tools "
                    "ADD COLUMN result_bytes INTEGER DEFAULT 0"
                )
            self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self.conn.commit()
        # After both paths: the column may have just been added above
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_parent "
            "ON sessions(source, parent_session_id)"
        )
        self._sweep_orphans()
        self.conn.commit()

    def _sweep_orphans(self):
        """Drop derived rows whose session row is gone.

        The v1 migration ran with foreign_keys=OFF, which left child rows behind
        and inflated any aggregate that does not join sessions. Only rebuildable
        counts are swept; session_content is left alone because its text may be
        the last copy of a reaped transcript.
        """
        for table in ("session_tools", "session_tool_detail", "session_agents",
                      "session_topics", "session_usage"):
            self.conn.execute(
                "DELETE FROM " + table + " AS c WHERE NOT EXISTS ("
                "SELECT 1 FROM sessions s WHERE s.source = c.session_source "
                "AND s.session_id = c.session_id)"
            )

    def _create_v2_tables(self):
        statements = (
            """CREATE TABLE IF NOT EXISTS sessions (
                source TEXT NOT NULL,
                session_id TEXT NOT NULL,
                parent_session_id TEXT,
                agent_name TEXT,
                project TEXT,
                project_name TEXT,
                cwd TEXT,
                title TEXT,
                title_display TEXT,
                tags TEXT,
                client TEXT,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                exchange_count INTEGER DEFAULT 0,
                start_time TEXT,
                end_time TEXT,
                duration_minutes INTEGER,
                model TEXT,
                has_compaction INTEGER DEFAULT 0,
                metadata_json TEXT,
                indexed_at TEXT NOT NULL,
                last_modified TEXT,
                file_hash TEXT,
                PRIMARY KEY (source, session_id)
            )""",
            """CREATE TABLE IF NOT EXISTS session_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_source TEXT NOT NULL DEFAULT 'claude',
                session_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                exchange_number INTEGER,
                source TEXT NOT NULL,
                FOREIGN KEY (session_source, session_id)
                    REFERENCES sessions(source, session_id) ON DELETE CASCADE
            )""",
            """CREATE TABLE IF NOT EXISTS session_tools (
                session_source TEXT NOT NULL DEFAULT 'claude',
                session_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                use_count INTEGER DEFAULT 0,
                write_tokens INTEGER DEFAULT 0,
                inject_tokens INTEGER DEFAULT 0,
                result_bytes INTEGER DEFAULT 0,
                PRIMARY KEY (session_source, session_id, tool_name),
                FOREIGN KEY (session_source, session_id)
                    REFERENCES sessions(source, session_id) ON DELETE CASCADE
            )""",
            """CREATE TABLE IF NOT EXISTS session_agents (
                session_source TEXT NOT NULL DEFAULT 'claude',
                session_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                invocation_count INTEGER DEFAULT 0,
                PRIMARY KEY (session_source, session_id, agent_name),
                FOREIGN KEY (session_source, session_id)
                    REFERENCES sessions(source, session_id) ON DELETE CASCADE
            )""",
            """CREATE TABLE IF NOT EXISTS session_tool_detail (
                session_source TEXT NOT NULL,
                session_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                detail TEXT NOT NULL,
                use_count INTEGER DEFAULT 0,
                write_tokens INTEGER DEFAULT 0,
                inject_tokens INTEGER DEFAULT 0,
                result_bytes INTEGER DEFAULT 0,
                PRIMARY KEY (session_source, session_id, tool_name, detail),
                FOREIGN KEY (session_source, session_id)
                    REFERENCES sessions(source, session_id) ON DELETE CASCADE
            )""",
            """CREATE TABLE IF NOT EXISTS session_usage (
                session_source TEXT NOT NULL,
                session_id TEXT NOT NULL,
                model TEXT NOT NULL,
                calls INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                cache_write_1h_tokens INTEGER DEFAULT 0,
                cache_write_5m_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                PRIMARY KEY (session_source, session_id, model),
                FOREIGN KEY (session_source, session_id)
                    REFERENCES sessions(source, session_id) ON DELETE CASCADE
            )""",
            """CREATE TABLE IF NOT EXISTS index_state (
                source TEXT PRIMARY KEY,
                initialized_at TEXT NOT NULL
            )""",
            "CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_client ON sessions(client)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_start ON sessions(start_time)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source)",
            "CREATE INDEX IF NOT EXISTS idx_usage_model ON session_usage(model)",
            "CREATE INDEX IF NOT EXISTS idx_detail_tool "
            "ON session_tool_detail(tool_name, detail)",
            "CREATE INDEX IF NOT EXISTS idx_topics_session "
            "ON session_topics(session_source, session_id)",
            "CREATE INDEX IF NOT EXISTS idx_topics_source ON session_topics(source)",
        )
        for statement in statements:
            self.conn.execute(statement)
        if not self._table_exists("session_content"):
            self.conn.execute("""
                CREATE VIRTUAL TABLE session_content USING fts5(
                    source UNINDEXED,
                    session_id UNINDEXED,
                    content,
                    tokenize='porter unicode61'
                )
            """)

    def _migrate_v1(self):
        """Backfill a Claude-only schema into source-qualified schema v2."""
        self.conn.execute("PRAGMA foreign_keys=OFF")
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            had_content = self._table_exists("session_content")
            if had_content:
                self.conn.execute("""
                    CREATE TEMP TABLE session_content_migration AS
                    SELECT session_id, content FROM session_content
                """)
                self.conn.execute("DROP TABLE session_content")

            for index_name in (
                "idx_sessions_project", "idx_sessions_client",
                "idx_sessions_start", "idx_topics_session",
                "idx_topics_source",
            ):
                self.conn.execute(f"DROP INDEX IF EXISTS {index_name}")

            for table in (
                "sessions", "session_topics", "session_tools", "session_agents"
            ):
                if self._table_exists(table):
                    self.conn.execute(
                        f"ALTER TABLE {table} RENAME TO {table}_v1"
                    )

            self._create_v2_tables()
            self.conn.execute("""
                INSERT INTO sessions (
                    source, session_id, project, project_name, title,
                    title_display, tags, client, file_path, file_size,
                    exchange_count, start_time, end_time, duration_minutes,
                    model, has_compaction, indexed_at, last_modified, file_hash
                )
                SELECT
                    'claude', session_id, project, project_name, title,
                    title_display, tags, client, file_path, file_size,
                    exchange_count, start_time, end_time, duration_minutes,
                    model, has_compaction, indexed_at, last_modified, file_hash
                FROM sessions_v1
            """)

            if self._table_exists("session_topics_v1"):
                self.conn.execute("""
                    INSERT INTO session_topics (
                        session_source, session_id, topic, captured_at,
                        exchange_number, source
                    )
                    SELECT 'claude', session_id, topic, captured_at,
                           exchange_number, source
                    FROM session_topics_v1
                """)
            if self._table_exists("session_tools_v1"):
                self.conn.execute("""
                    INSERT INTO session_tools (
                        session_source, session_id, tool_name, use_count
                    )
                    SELECT 'claude', session_id, tool_name, use_count
                    FROM session_tools_v1
                """)
            if self._table_exists("session_agents_v1"):
                self.conn.execute("""
                    INSERT INTO session_agents (
                        session_source, session_id, agent_name, invocation_count
                    )
                    SELECT 'claude', session_id, agent_name, invocation_count
                    FROM session_agents_v1
                """)
            if had_content:
                self.conn.execute("""
                    INSERT INTO session_content (source, session_id, content)
                    SELECT 'claude', session_id, content
                    FROM session_content_migration
                """)
            # v1 FTS was built by the old rules; leave claude uninitialized so it reparses

            for table in (
                "session_topics_v1", "session_tools_v1",
                "session_agents_v1", "sessions_v1",
            ):
                self.conn.execute(f"DROP TABLE IF EXISTS {table}")
            self.conn.execute("DROP TABLE IF EXISTS session_content_migration")
            self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            self.conn.execute("PRAGMA foreign_keys=ON")

    @staticmethod
    def _file_hash(path: Path) -> str:
        stat = path.stat()
        value = f"{stat.st_size}:{stat.st_mtime_ns}".encode()
        return hashlib.sha256(value).hexdigest()

    def _parse_session(self, session_path: Path, source: str = "claude") -> Optional[dict]:
        """Compatibility wrapper around the source adapter parser."""
        adapter = self.adapters.get(source)
        if not adapter:
            return None
        data = adapter.parse(session_path)
        if data:
            data["file_hash"] = self._file_hash(session_path)
        return data

    def _iter_session_files(self, source: str | None = None):
        requested = source.lower() if source else None
        if requested and requested not in VALID_SOURCES:
            raise ValueError(f"Unknown source: {source}")
        for name, adapter in self.adapters.items():
            if requested and name != requested:
                continue
            for path in adapter.discover():
                yield name, path

    def _find_session_files(self, session_id: str,
                            source: str | None = None) -> list[tuple[str, Path]]:
        if ":" in session_id and not source:
            prefix, candidate_id = session_id.split(":", 1)
            if prefix in VALID_SOURCES:
                source, session_id = prefix, candidate_id
        matches = []
        for name, path in self._iter_session_files(source):
            if path.stem == session_id or session_id in path.stem:
                data = self.adapters[name].parse(path)
                if data and data["session_id"] == session_id:
                    matches.append((name, path))
        return matches

    def index_session(self, session_id: str = None, file_path: str = None,
                      source: str | None = None) -> bool:
        """Index one source-qualified session or an explicitly supplied file."""
        matches: list[tuple[str, Path]] = []
        if file_path:
            path = Path(file_path).expanduser()
            if source:
                matches = [(source, path)]
            else:
                for name, adapter in self.adapters.items():
                    if adapter.owns(path):
                        matches.append((name, path))
                if not matches:
                    print("Cannot determine session source; pass --source",
                          file=sys.stderr)
                    return False
        elif session_id:
            matches = self._find_session_files(session_id, source)

        if len(matches) > 1:
            print(
                "Session ID is ambiguous across sources; pass --source",
                file=sys.stderr,
            )
            return False
        if not matches or not matches[0][1].exists():
            print(f"Session file not found: {session_id or file_path}",
                  file=sys.stderr)
            return False

        name, path = matches[0]
        data = self._parse_session(path, name)
        return bool(data and self._upsert_session(data))

    def _semantic_index(self):
        """Lazily attach the vector index; None whenever the extra is absent."""
        if self._semantic is not False:
            return self._semantic
        self._semantic = None
        if not self.embed_enabled or not semantic.available():
            return None
        index = semantic.SemanticIndex(self.conn)
        if not index.enabled:
            return None
        try:
            index.ensure_schema(semantic.model_dims())
        except semantic.SemanticUnavailable as exc:
            print(f"Semantic index disabled: {exc}", file=sys.stderr)
            return None
        self._semantic = index
        return index

    def _embed_session(self, identity: tuple[str, str], data: dict):
        index = self._semantic_index()
        if index is None:
            return
        # Only the prose half is embedded; the tool digest is lexical territory
        prose = (data.get("fts_content") or "")[:data.get("prose_chars") or 0]
        try:
            index.index_session(*identity, prose)
        except (sqlite3.Error, semantic.SemanticUnavailable) as exc:
            print(
                f"Embedding failed for {identity[0]}:{identity[1]}: {exc}",
                file=sys.stderr,
            )

    def _upsert_session(self, data: dict) -> bool:
        try:
            now = datetime.now().isoformat()
            values = (
                data["source"], data["session_id"], data["project"],
                data["project_name"], data.get("cwd"), data["title"],
                data["title_display"], data["tags"], data["client"],
                data["file_path"], data["file_size"], data["exchange_count"],
                data["start_time"], data["end_time"], data["duration_minutes"],
                data["model"], data["has_compaction"],
                data.get("metadata_json"), now, now, data["file_hash"],
                data.get("parent_session_id"), data.get("agent_name"),
                data.get("prose_chars") or 0,
                len(data.get("fts_content") or ""),
            )
            self.conn.execute("""
                INSERT INTO sessions (
                    source, session_id, project, project_name, cwd, title,
                    title_display, tags, client, file_path, file_size,
                    exchange_count, start_time, end_time, duration_minutes,
                    model, has_compaction, metadata_json, indexed_at,
                    last_modified, file_hash, parent_session_id, agent_name,
                    prose_chars, content_chars
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?
                )
                ON CONFLICT(source, session_id) DO UPDATE SET
                    parent_session_id=excluded.parent_session_id,
                    agent_name=excluded.agent_name,
                    project=excluded.project,
                    project_name=excluded.project_name,
                    cwd=excluded.cwd,
                    title=excluded.title,
                    title_display=excluded.title_display,
                    tags=excluded.tags,
                    client=excluded.client,
                    file_path=excluded.file_path,
                    file_size=excluded.file_size,
                    exchange_count=excluded.exchange_count,
                    start_time=excluded.start_time,
                    end_time=excluded.end_time,
                    duration_minutes=excluded.duration_minutes,
                    model=excluded.model,
                    has_compaction=excluded.has_compaction,
                    metadata_json=excluded.metadata_json,
                    indexed_at=excluded.indexed_at,
                    last_modified=excluded.last_modified,
                    file_hash=excluded.file_hash,
                    prose_chars=excluded.prose_chars,
                    content_chars=excluded.content_chars
            """, values)

            identity = (data["source"], data["session_id"])
            self.conn.execute(
                "DELETE FROM session_tools "
                "WHERE session_source=? AND session_id=?", identity
            )
            tool_tokens = data.get("tool_tokens") or {}
            for tool in set(data["tools"]) | set(tool_tokens):
                counts = tool_tokens.get(tool) or {}
                self.conn.execute("""
                    INSERT INTO session_tools (
                        session_source, session_id, tool_name, use_count,
                        write_tokens, inject_tokens, result_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    *identity, tool, data["tools"].get(tool, 0),
                    counts.get("write_tokens", 0),
                    counts.get("inject_tokens", 0),
                    counts.get("result_bytes", 0),
                ))

            self.conn.execute(
                "DELETE FROM session_tool_detail "
                "WHERE session_source=? AND session_id=?", identity
            )
            for (tool, detail), counts in (data.get("tool_details") or {}).items():
                self.conn.execute("""
                    INSERT INTO session_tool_detail (
                        session_source, session_id, tool_name, detail,
                        use_count, write_tokens, inject_tokens, result_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    *identity, tool, detail, counts["use_count"],
                    counts["write_tokens"], counts["inject_tokens"],
                    counts["result_bytes"],
                ))

            self.conn.execute(
                "DELETE FROM session_usage "
                "WHERE session_source=? AND session_id=?", identity
            )
            for model, counts in (data.get("usage") or {}).items():
                self.conn.execute("""
                    INSERT INTO session_usage (
                        session_source, session_id, model, calls,
                        input_tokens, output_tokens, cache_write_tokens,
                        cache_write_1h_tokens, cache_write_5m_tokens,
                        cache_read_tokens, reasoning_tokens
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    *identity, model, counts["calls"],
                    counts["input_tokens"], counts["output_tokens"],
                    counts["cache_write_tokens"],
                    counts["cache_write_1h_tokens"],
                    counts["cache_write_5m_tokens"],
                    counts["cache_read_tokens"], counts["reasoning_tokens"],
                ))

            self.conn.execute(
                "DELETE FROM session_agents "
                "WHERE session_source=? AND session_id=?", identity
            )
            for agent, count in data["agents"].items():
                self.conn.execute("""
                    INSERT INTO session_agents (
                        session_source, session_id, agent_name, invocation_count
                    ) VALUES (?, ?, ?, ?)
                """, (*identity, agent, count))

            self.conn.execute(
                "DELETE FROM session_content WHERE source=? AND session_id=?",
                identity,
            )
            if data["fts_content"]:
                self.conn.execute("""
                    INSERT INTO session_content (source, session_id, content)
                    VALUES (?, ?, ?)
                """, (*identity, data["fts_content"]))
            self._embed_session(identity, data)

            for topic in data["topics"]:
                existing = self.conn.execute("""
                    SELECT id FROM session_topics
                    WHERE session_source=? AND session_id=?
                      AND topic=? AND source=?
                """, (*identity, topic["topic"], topic["source"])).fetchone()
                if not existing:
                    self.conn.execute("""
                        INSERT INTO session_topics (
                            session_source, session_id, topic, captured_at,
                            exchange_number, source
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        *identity, topic["topic"], topic["captured_at"],
                        topic["exchange_number"], topic["source"],
                    ))
            self.conn.commit()
            return True
        except Exception as exc:
            print(
                f"Error upserting {data.get('source')}:{data.get('session_id')}: "
                f"{exc}",
                file=sys.stderr,
            )
            self.conn.rollback()
            return False

    def _index_files(self, source: str | None, progress_interval: int | None,
                     backfill: bool) -> dict:
        files = list(self._iter_session_files(source))
        stats = {
            "total" if backfill else "checked": len(files),
            "indexed": 0,
            "skipped" if backfill else "unchanged": 0,
            "errors": 0,
            "by_source": {},
        }
        unchanged_key = "skipped" if backfill else "unchanged"
        if backfill:
            print(f"Found {len(files)} session files to index")

        for index, (name, path) in enumerate(files, 1):
            source_stats = stats["by_source"].setdefault(
                name, {"checked": 0, "indexed": 0, "unchanged": 0, "errors": 0}
            )
            source_stats["checked"] += 1
            if progress_interval and index % progress_interval == 0:
                print(
                    f"  Progress: {index}/{len(files)} "
                    f"({stats['indexed']} indexed, {stats['errors']} errors)"
                )

            current_hash = self._file_hash(path)
            parsed = None
            # A filename-derived id may be wrong; the skip below still checks path + hash.
            session_id = self.adapters[name].session_id_from_path(path)
            if session_id is None:
                parsed = self._parse_session(path, name)
                if not parsed:
                    stats["errors"] += 1
                    source_stats["errors"] += 1
                    continue
                session_id = parsed["session_id"]

            existing = self.conn.execute("""
                SELECT file_hash, file_path FROM sessions
                WHERE source=? AND session_id=?
            """, (name, session_id)).fetchone()
            # A relocated transcript keeps its hash, so file_path must match too
            if (existing and existing["file_hash"] == current_hash
                    and existing["file_path"] == str(path)):
                stats[unchanged_key] += 1
                source_stats["unchanged"] += 1
                continue

            data = parsed or self._parse_session(path, name)
            if not data:
                stats["errors"] += 1
                source_stats["errors"] += 1
            elif self._upsert_session(data):
                stats["indexed"] += 1
                source_stats["indexed"] += 1
            else:
                stats["errors"] += 1
                source_stats["errors"] += 1
        scanned_sources = (
            [source] if source else list(self.adapters)
        )
        now = datetime.now().isoformat()
        for name in scanned_sources:
            self.conn.execute("""
                INSERT INTO index_state (source, initialized_at)
                VALUES (?, ?)
                ON CONFLICT(source) DO UPDATE SET initialized_at=excluded.initialized_at
            """, (name, now))
        self.conn.commit()
        return stats

    def backfill_all(self, progress_interval: int = 100,
                     source: str | None = None) -> dict:
        stats = self._index_files(source, progress_interval, backfill=True)
        print(
            f"\nBackfill complete: {stats['indexed']} indexed, "
            f"{stats['skipped']} unchanged, {stats['errors']} errors"
        )
        return stats

    def index_incremental(self, source: str | None = None) -> dict:
        return self._index_files(source, None, backfill=False)

    def get_stats(self, source: str | None = None) -> dict:
        where = " WHERE source=?" if source else ""
        params = (source,) if source else ()
        stats = {
            "total_sessions": self.conn.execute(
                f"SELECT COUNT(*) FROM sessions{where}", params
            ).fetchone()[0],
            "total_topics": self.conn.execute(
                "SELECT COUNT(*) FROM session_topics"
                + (" WHERE session_source=?" if source else ""), params
            ).fetchone()[0],
            "total_tools": self.conn.execute(
                "SELECT COUNT(DISTINCT tool_name) FROM session_tools"
                + (" WHERE session_source=?" if source else ""), params
            ).fetchone()[0],
            "total_agents": self.conn.execute(
                "SELECT COUNT(DISTINCT agent_name) FROM session_agents"
                + (" WHERE session_source=?" if source else ""), params
            ).fetchone()[0],
            "total_subagents": self.conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE parent_session_id IS NOT NULL"
                + (" AND source=?" if source else ""), params
            ).fetchone()[0],
        }
        rows = self.conn.execute(
            "SELECT source, COUNT(*) AS cnt FROM sessions"
            f"{where} GROUP BY source ORDER BY source", params
        ).fetchall()
        stats["by_source"] = {row["source"]: row["cnt"] for row in rows}
        rows = self.conn.execute(
            "SELECT project_name, COUNT(*) AS cnt FROM sessions"
            f"{where} GROUP BY project_name ORDER BY cnt DESC", params
        ).fetchall()
        stats["by_project"] = {row["project_name"]: row["cnt"] for row in rows}
        rows = self.conn.execute(
            "SELECT client, COUNT(*) AS cnt FROM sessions"
            f"{where + (' AND' if where else ' WHERE')} client IS NOT NULL "
            "GROUP BY client ORDER BY cnt DESC", params
        ).fetchall()
        stats["by_client"] = {row["client"]: row["cnt"] for row in rows}
        tool_where = " WHERE st.session_source=?" if source else ""
        rows = self.conn.execute("""
            SELECT st.tool_name, SUM(st.use_count) AS total
            FROM session_tools st
        """ + tool_where + """
            GROUP BY st.tool_name ORDER BY total DESC LIMIT 10
        """, params).fetchall()
        stats["top_tools"] = {row["tool_name"]: row["total"] for row in rows}
        row = self.conn.execute(
            "SELECT MIN(start_time) AS earliest, MAX(start_time) AS latest "
            f"FROM sessions{where + (' AND' if where else ' WHERE')} "
            "start_time IS NOT NULL", params
        ).fetchone()
        stats["date_range"] = {
            "earliest": row["earliest"][:10] if row["earliest"] else None,
            "latest": row["latest"][:10] if row["latest"] else None,
        }
        return stats

    def add_topic(self, session_id: str, topic: str, source: str,
                  exchange_number: int = None, session_source: str = "claude"):
        self.conn.execute("""
            INSERT INTO session_topics (
                session_source, session_id, topic, captured_at,
                exchange_number, source
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_source, session_id, topic, datetime.now().isoformat(),
            exchange_number, source,
        ))
        self.conn.commit()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Agent Session Index indexer")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--index", metavar="SESSION_ID")
    parser.add_argument("--source", choices=VALID_SOURCES)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--db-path", metavar="PATH")
    parser.add_argument("--projects-dir", metavar="PATH",
                        help="Override Claude projects root")
    parser.add_argument("--codex-sessions-dir", metavar="PATH",
                        help="Override Codex sessions root")
    args = parser.parse_args()

    indexer = SessionIndexer(
        db_path=config.get_db_path(args.db_path) if args.db_path else None,
        projects_dir=Path(args.projects_dir).expanduser()
        if args.projects_dir else None,
        codex_sessions_dir=Path(args.codex_sessions_dir).expanduser()
        if args.codex_sessions_dir else None,
    )
    indexer.connect()
    try:
        if args.backfill:
            indexer.backfill_all(source=args.source)
        elif args.incremental:
            stats = indexer.index_incremental(source=args.source)
            print(json.dumps(stats, indent=2))
        elif args.index:
            if not indexer.index_session(
                session_id=args.index, source=args.source
            ):
                sys.exit(1)
        elif args.stats:
            print(json.dumps(indexer.get_stats(args.source), indent=2))
        else:
            parser.print_help()
    finally:
        indexer.close()


if __name__ == "__main__":
    main()
