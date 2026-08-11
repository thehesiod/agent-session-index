import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from session_index import cli, config
from session_index.analyzer import (
    analytics,
    format_context,
    get_context,
    indexed_excerpts,
    synthesize,
)
from session_index.indexer import SCHEMA_VERSION, SessionIndexer
from session_index.search import SessionSearch, format_result
from session_index.sources import (
    MAX_MESSAGE_CHARS,
    ClaudeSourceAdapter,
    CodexSourceAdapter,
    _ToolTokens,
    _add_claude_usage,
    bash_command,
    canonical_tool_name,
    sanitize_text,
    strip_codex_injected_context,
)


FIXTURES = Path(__file__).parent / "fixtures"
CLAUDE_ROOT = FIXTURES / "claude" / "projects"
CODEX_ROOT = FIXTURES / "codex" / "sessions"


def source_configs():
    return {
        "claude": {"enabled": True, "root": str(CLAUDE_ROOT)},
        "codex": {"enabled": True, "root": str(CODEX_ROOT)},
    }


class AdapterTests(unittest.TestCase):
    def test_claude_adapter_normalizes_and_excludes_system_prompt(self):
        path = next(ClaudeSourceAdapter(CLAUDE_ROOT).discover())
        data = ClaudeSourceAdapter(CLAUDE_ROOT).parse(path)

        self.assertEqual(data["source"], "claude")
        self.assertEqual(data["session_id"], "shared-session")
        self.assertEqual(data["model"], "claude-test")
        self.assertEqual(data["tools"], {"Read": 1})
        self.assertIn("cobalt-needle", data["fts_content"])
        self.assertNotIn("forbidden-claude-system", data["fts_content"])

    def test_claude_split_response_is_billed_once(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "projects"
            (root / "-proj").mkdir(parents=True)
            usage = {"input_tokens": 100, "output_tokens": 20,
                     "cache_read_input_tokens": 5}
            # one API response arrives as several rows repeating id and usage
            rows = [
                {"type": "assistant", "timestamp": "2026-01-01T00:00:00Z",
                 "message": {"id": "msg_1", "role": "assistant",
                             "model": "claude-test", "usage": usage,
                             "content": [{"type": "text", "text": "part one"}]}},
                {"type": "assistant", "timestamp": "2026-01-01T00:00:01Z",
                 "message": {"id": "msg_1", "role": "assistant",
                             "model": "claude-test", "usage": usage,
                             "content": [{"type": "text", "text": "part two"}]}},
            ]
            path = root / "-proj" / "split.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

            data = ClaudeSourceAdapter(root).parse(path)

        bucket = data["usage"]["claude-test"]
        self.assertEqual(bucket["calls"], 1)
        self.assertEqual(bucket["input_tokens"], 100)
        self.assertEqual(bucket["output_tokens"], 20)

    def test_claude_adapter_excludes_harness_injected_context(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "projects"
            (root / "-proj").mkdir(parents=True)
            rows = [
                {"type": "user", "timestamp": "2026-01-01T00:00:00Z", "message": {
                    "role": "user", "content":
                    "<task-notification>forbidden-task-note</task-notification>"}},
                {"type": "user", "timestamp": "2026-01-01T00:00:01Z", "message": {
                    "role": "user", "content":
                    "<local-command-caveat>Caveat: forbidden-caveat"
                    "</local-command-caveat>"}},
                {"type": "user", "timestamp": "2026-01-01T00:00:02Z", "message": {
                    "role": "user", "content":
                    "<system-reminder>forbidden-reminder</system-reminder>"}},
                {"type": "user", "timestamp": "2026-01-01T00:00:03Z", "message": {
                    "role": "user", "content":
                    "<local-command-stdout>forbidden-stdout</local-command-stdout>"}},
                {"type": "user", "timestamp": "2026-01-01T00:00:04Z", "message": {
                    "role": "user", "content":
                    "<task-notification>forbidden-unclosed and never closed"}},
                {"type": "user", "timestamp": "2026-01-01T00:00:05Z", "message": {
                    "role": "user", "content":
                    "<command-name>/ns-review</command-name> keep-the-args"}},
                {"type": "user", "timestamp": "2026-01-01T00:00:06Z", "message": {
                    "role": "user", "content": "Find the teal scheduler bug"}},
                {"type": "assistant", "timestamp": "2026-01-01T00:00:07Z", "message": {
                    "role": "assistant", "model": "claude-test",
                    "content": [{"type": "text", "text": "answer is jade-needle"}]}},
            ]
            path = root / "-proj" / "injected.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

            data = ClaudeSourceAdapter(root).parse(path)
            exchanges = ClaudeSourceAdapter(root).extract_exchanges(path)

        for forbidden in (
            "forbidden-task-note", "forbidden-caveat", "forbidden-reminder",
            "forbidden-stdout", "forbidden-unclosed",
        ):
            self.assertNotIn(forbidden, data["fts_content"])
        # a slash command records what the user asked for, so it stays searchable
        self.assertIn("keep-the-args", data["fts_content"])
        # ...but never titles the session
        self.assertEqual(data["title"], "Find the teal scheduler bug")
        self.assertEqual(data["exchange_count"], 3)
        # extract_exchanges must filter the same way parse() does
        user_side = "\n".join(item["user"] for item in exchanges)
        self.assertNotIn("forbidden-task-note", user_side)
        self.assertNotIn("forbidden-reminder", user_side)
        self.assertIn("Find the teal scheduler bug", user_side)

    def test_codex_adapter_normalizes_safe_rollout_records(self):
        path = next(CodexSourceAdapter(CODEX_ROOT).discover())
        data = CodexSourceAdapter(CODEX_ROOT).parse(path)

        self.assertEqual(data["source"], "codex")
        self.assertEqual(data["session_id"], "shared-session")
        self.assertEqual(data["cwd"], "/Users/test/codex-demo")
        self.assertEqual(data["model"], "gpt-test")
        self.assertEqual(data["tools"], {
            "shell_command": 1,
            "linear.get_issue": 1,
        })
        self.assertIn("amber-needle", data["fts_content"])
        # tool output is indexed as a digest, after the prose boundary
        self.assertIn("forbidden-tool-output", data["fts_content"])
        self.assertIn("forbidden-mcp-result", data["fts_content"])
        self.assertLess(data["prose_chars"], len(data["fts_content"]))
        self.assertNotIn(
            "forbidden-tool-output", data["fts_content"][:data["prose_chars"]]
        )
        for forbidden in (
            "forbidden-codex-developer",
            "forbidden-codex-base-instructions",
            "forbidden-encrypted-reasoning",
            "forbidden-reasoning-summary",
            "huge-binary-tool-payload",
            "forbidden-environment-context",
            "forbidden-mcp-arguments",
        ):
            self.assertNotIn(forbidden, data["fts_content"])
        metadata = json.loads(data["metadata_json"])
        self.assertEqual(metadata["git"]["branch"], "fixture")
        self.assertNotIn("base_instructions", metadata)

    def test_codex_adapter_excludes_injected_context(self):
        path = next(CodexSourceAdapter(CODEX_ROOT).discover())
        data = CodexSourceAdapter(CODEX_ROOT).parse(path)

        for forbidden in (
            "forbidden-injected-delegation",
            "forbidden-injected-history",
            "forbidden-injected-agents-md",
            "forbidden-injected-plugins",
            "forbidden-compaction-replacement",
        ):
            self.assertNotIn(forbidden, data["fts_content"])
        # the AGENTS.md record leaks a filesystem path once its blocks are gone
        self.assertNotIn("AGENTS.md instructions", data["fts_content"])
        # a wrapper opening a record must be stripped, not drop the real prose
        self.assertIn("Also check the amber scheduler.", data["fts_content"])
        self.assertEqual(data["title"], "Find the orange scheduler bug")
        self.assertEqual(data["exchange_count"], 3)

    def test_codex_adapter_keeps_its_own_identity_and_counts_mcp_once(self):
        path = next(CodexSourceAdapter(CODEX_ROOT).discover())
        data = CodexSourceAdapter(CODEX_ROOT).parse(path)

        # a subagent rollout replays its parent's meta; adopting it collides the rows
        self.assertEqual(data["session_id"], "shared-session")
        self.assertEqual(data["cwd"], "/Users/test/codex-demo")
        self.assertEqual(data["start_time"], "2026-01-02T11:00:00Z")
        # call-2 emits both a function_call and an mcp_tool_call_end
        self.assertEqual(data["tools"], {
            "shell_command": 1,
            "linear.get_issue": 1,
        })

    def test_codex_injected_block_survives_no_closing_tag(self):
        oversized = "plugin name " * (MAX_MESSAGE_CHARS // 6)
        text = sanitize_text(
            f"<recommended_plugins>{oversized}</recommended_plugins>"
        )

        self.assertNotIn("</recommended_plugins>", text)
        self.assertEqual(strip_codex_injected_context(text), "")

    def test_codex_adapter_reads_compaction_from_records_not_settings(self):
        path = next(CodexSourceAdapter(CODEX_ROOT).discover())
        data = CodexSourceAdapter(CODEX_ROOT).parse(path)

        self.assertEqual(data["has_compaction"], 1)
        # turn_context.summary is a setting ("auto"/"none"), never a summary
        self.assertEqual(data["topics"], [])

    def test_codex_summary_setting_alone_is_not_compaction(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "sessions"
            root.mkdir()
            (root / "plain.jsonl").write_text(
                json.dumps({
                    "timestamp": "2026-01-02T11:00:00Z",
                    "type": "turn_context",
                    "payload": {"cwd": "/Users/test/demo", "summary": "none"},
                }) + "\n"
            )

            data = CodexSourceAdapter(root).parse(root / "plain.jsonl")

        self.assertEqual(data["has_compaction"], 0)
        self.assertEqual(data["topics"], [])

    def test_codex_event_only_rollout_is_still_indexed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "sessions"
            root.mkdir()
            (root / "event-only.jsonl").write_text("\n".join(json.dumps(rec) for rec in [
                {"timestamp": "2026-01-02T11:00:00Z", "type": "session_meta",
                 "payload": {"id": "event-only", "cwd": "/Users/test/demo"}},
                {"timestamp": "2026-01-02T11:00:01Z", "type": "event_msg",
                 "payload": {"type": "user_message",
                             "message": "Review the teal scheduler regression"}},
                {"timestamp": "2026-01-02T11:00:02Z", "type": "event_msg",
                 "payload": {"type": "agent_message",
                             "message": "The event-only answer is jade-needle."}},
            ]) + "\n")
            path = root / "event-only.jsonl"

            data = CodexSourceAdapter(root).parse(path)
            exchanges = CodexSourceAdapter(root).extract_exchanges(path)

        # review/exec subagent rollouts store their turns only as events
        self.assertEqual(data["title"], "Review the teal scheduler regression")
        self.assertEqual(data["exchange_count"], 2)
        self.assertIn("jade-needle", data["fts_content"])
        self.assertEqual(len(exchanges), 1)
        self.assertIn("jade-needle", exchanges[0]["assistant"])

    def test_codex_event_messages_do_not_duplicate_response_items(self):
        path = next(CodexSourceAdapter(CODEX_ROOT).discover())
        data = CodexSourceAdapter(CODEX_ROOT).parse(path)

        # the shared fixture carries response_item messages, so events must not re-add them
        self.assertEqual(data["fts_content"].count("amber-needle"), 1)

    def test_codex_adapter_discovers_archived_rollouts(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "sessions"
            archived = Path(tempdir) / "archived_sessions"
            root.mkdir()
            archived.mkdir()
            (root / "live.jsonl").write_text("")
            (archived / "archived.jsonl").write_text("")

            found = {path.name for path in CodexSourceAdapter(root).discover()}
            adapter = CodexSourceAdapter(root)
            owns_archived = adapter.owns(archived / "archived.jsonl")
            owns_live = adapter.owns(root / "live.jsonl")
            owns_foreign = adapter.owns(Path(tempdir) / "elsewhere.jsonl")

        self.assertEqual(found, {"live.jsonl", "archived.jsonl"})
        # a rollout the adapter discovers must also resolve back to this source
        self.assertTrue(owns_archived)
        self.assertTrue(owns_live)
        self.assertFalse(owns_foreign)

    def test_codex_adapter_reads_session_id_off_the_rollout_name(self):
        adapter = CodexSourceAdapter(CODEX_ROOT)
        session_id = "deadbeef-1234-7abc-8def-000000000001"

        self.assertEqual(
            adapter.session_id_from_path(
                Path(f"rollout-2026-06-11T00-42-52-{session_id}.jsonl")
            ),
            session_id,
        )

    def test_codex_adapter_defers_to_session_meta_when_the_name_has_no_id(self):
        adapter = CodexSourceAdapter(CODEX_ROOT)

        self.assertIsNone(
            adapter.session_id_from_path(Path("rollout-shared-session.jsonl"))
        )

    def test_claude_adapter_reads_session_id_off_the_transcript_name(self):
        adapter = ClaudeSourceAdapter(CLAUDE_ROOT)

        self.assertEqual(
            adapter.session_id_from_path(Path("shared-session.jsonl")),
            "shared-session",
        )


class IncrementalReparseTests(unittest.TestCase):
    def test_unchanged_codex_rollout_is_not_reparsed(self):
        source = next(CodexSourceAdapter(CODEX_ROOT).discover())
        session_id = "deadbeef-1234-7abc-8def-000000000001"
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "sessions"
            (root / "2026" / "01" / "02").mkdir(parents=True)
            rollout = (
                root / "2026" / "01" / "02"
                / f"rollout-2026-01-02T00-00-00-{session_id}.jsonl"
            )
            lines = source.read_text().splitlines()
            meta = json.loads(lines[0])
            meta["payload"]["id"] = session_id
            lines[0] = json.dumps(meta)
            rollout.write_text("\n".join(lines) + "\n")

            indexer = SessionIndexer(
                db_path=Path(tempdir) / "sessions.db",
                source_configs={
                    "claude": {"enabled": False},
                    "codex": {"enabled": True, "root": str(root)},
                },
            )
            indexer.connect()
            try:
                self.assertEqual(indexer.backfill_all(progress_interval=0)["indexed"], 1)
                with patch.object(
                    CodexSourceAdapter, "parse", side_effect=AssertionError("reparsed")
                ):
                    stats = indexer.index_incremental()
            finally:
                indexer.close()

        self.assertEqual(stats["unchanged"], 1)
        self.assertEqual(stats["indexed"], 0)


class IndexIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "sessions.db"
        self.indexer = SessionIndexer(
            db_path=self.db_path, source_configs=source_configs()
        )
        self.indexer.connect()
        self.stats = self.indexer.backfill_all(progress_interval=0)

    def tearDown(self):
        self.indexer.close()
        self.tempdir.cleanup()

    def test_cross_source_identity_search_filters_and_stats(self):
        self.assertEqual(self.stats["indexed"], 2)
        rows = self.indexer.conn.execute(
            "SELECT source, session_id FROM sessions ORDER BY source"
        ).fetchall()
        self.assertEqual(
            [(row["source"], row["session_id"]) for row in rows],
            [("claude", "shared-session"), ("codex", "shared-session")],
        )

        search = SessionSearch(self.db_path)
        search.connect()
        try:
            self.assertEqual(
                search.search("cobalt-needle", source="claude")[0]["source"],
                "claude",
            )
            self.assertEqual(
                search.search("amber-needle", source="codex")[0]["source"],
                "codex",
            )
            self.assertEqual(search.search("amber-needle", source="claude"), [])
            self.assertEqual(
                search.find(source="codex", tool="shell_command")[0]["source"],
                "codex",
            )
            self.assertEqual(
                search.recent(source="claude")[0]["source"], "claude"
            )
            self.assertEqual(
                search.tools_usage(source="codex")[0]["session_count"], 1
            )
            self.assertEqual(
                search.topics("shared-session", source="claude")[0]["source"],
                "compaction_summary",
            )
            self.assertEqual(
                search.topics("shared-session", source="codex"), []
            )
            self.assertEqual(search.stats()["by_source"], {
                "claude": 1, "codex": 1,
            })
            self.assertEqual(search.stats("codex")["total_sessions"], 1)
            self.assertEqual(
                search.stats("codex")["by_source"], {"codex": 1}
            )
        finally:
            search.close()

    def test_context_uses_matching_source_parser(self):
        ambiguous = get_context("shared", db_path=self.db_path)
        self.assertIn("ambiguous", ambiguous["error"])

        claude = get_context(
            "shared-session", source="claude", db_path=self.db_path
        )
        codex = get_context(
            "shared-session", source="codex", db_path=self.db_path
        )
        self.assertIn("cobalt-needle", claude["exchanges"][0]["assistant"])
        self.assertIn("amber-needle", codex["exchanges"][0]["assistant"])
        self.assertIn("[shell_command]", codex["exchanges"][0]["assistant"])
        self.assertIn("[linear.get_issue]", codex["exchanges"][0]["assistant"])
        self.assertNotIn(
            "forbidden-tool-output", codex["exchanges"][0]["assistant"]
        )

    def test_analytics_filters_source_and_synthesis_stays_local(self):
        codex = analytics(db_path=self.db_path, source="codex")
        self.assertEqual(codex["source"], "codex")
        self.assertEqual(codex["overview"]["total_sessions"], 1)
        self.assertEqual(codex["top_tools"][0]["tool_name"], "shell_command")
        result = synthesize("needle", db_path=self.db_path)
        self.assertIn("never uploads", result["error"])

    def test_cli_labels_and_filters_source(self):
        stdout = io.StringIO()
        argv = [
            "sessions", "--db-path", str(self.db_path),
            "search", "amber-needle", "--source", "codex",
        ]
        with patch("sys.argv", argv), redirect_stdout(stdout):
            cli.main()
        output = stdout.getvalue()
        self.assertIn("[codex]", output)
        self.assertIn("codex resume shared-session", output)
        self.assertNotIn("[claude]", output)


class RelocationTests(unittest.TestCase):
    def test_moved_transcript_refreshes_file_path(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "projects"
            (root / "-old-project").mkdir(parents=True)
            transcript = root / "-old-project" / "moved-session.jsonl"
            transcript.write_text(json.dumps({
                "type": "user",
                "message": {"role": "user", "content": "cobalt-needle"},
                "timestamp": "2026-01-01T00:00:00Z",
            }) + "\n")

            indexer = SessionIndexer(
                db_path=Path(tempdir) / "sessions.db",
                source_configs={
                    "claude": {"enabled": True, "root": str(root)},
                    "codex": {"enabled": False, "root": str(CODEX_ROOT)},
                },
            )
            indexer.connect()
            try:
                indexer.backfill_all(progress_interval=0)
                moved = root / "-new-project" / transcript.name
                moved.parent.mkdir()
                transcript.rename(moved)

                stats = indexer.backfill_all(progress_interval=0)

                self.assertEqual(stats["indexed"], 1)
                row = indexer.conn.execute(
                    "SELECT file_path, project FROM sessions "
                    "WHERE source='claude' AND session_id='moved-session'"
                ).fetchone()
                self.assertEqual(row["file_path"], str(moved))
                self.assertEqual(row["project"], "-new-project")
            finally:
                indexer.close()


class DeletedTranscriptTests(unittest.TestCase):
    def test_context_falls_back_to_indexed_text(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "projects"
            (root / "-project").mkdir(parents=True)
            transcript = root / "-project" / "reaped-session.jsonl"
            transcript.write_text(json.dumps({
                "type": "user",
                "message": {"role": "user", "content": "cobalt-needle stays"},
                "timestamp": "2026-01-01T00:00:00Z",
            }) + "\n")

            db_path = Path(tempdir) / "sessions.db"
            indexer = SessionIndexer(
                db_path=db_path,
                source_configs={
                    "claude": {"enabled": True, "root": str(root)},
                    "codex": {"enabled": False, "root": str(CODEX_ROOT)},
                },
            )
            indexer.connect()
            try:
                indexer.backfill_all(progress_interval=0)
            finally:
                indexer.close()
            transcript.unlink()

            result = get_context(
                "reaped-session", query="cobalt-needle", db_path=db_path,
                source="claude",
            )

        self.assertTrue(result["transcript_missing"])
        self.assertEqual(result["exchanges"], [])
        self.assertEqual(result["excerpts"], ["cobalt-needle stays"])
        self.assertIn("recovered from the index", format_context(result))

    def test_indexed_excerpts_filter_on_an_invalid_regex(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE VIRTUAL TABLE session_content USING fts5(
                source, session_id, content
            );
        """)
        conn.execute(
            "INSERT INTO session_content VALUES (?, ?, ?)",
            ("claude", "s1", "cobalt-needle stays\nunrelated filler line"),
        )

        try:
            # "(unclosed" cannot compile; without a fallback every block matched
            excerpts = indexed_excerpts(conn, "claude", "s1", query="(unclosed")
            literal = indexed_excerpts(conn, "claude", "s1", query="cobalt-needle")
        finally:
            conn.close()

        self.assertEqual(excerpts, [])
        self.assertEqual(literal, ["cobalt-needle stays"])


class MigrationTests(unittest.TestCase):
    def test_v1_database_is_backfilled_as_claude(self):
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "legacy.db"
            conn = sqlite3.connect(db_path)
            conn.executescript("""
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    project TEXT, project_name TEXT, title TEXT,
                    title_display TEXT, tags TEXT, client TEXT,
                    file_path TEXT NOT NULL, file_size INTEGER,
                    exchange_count INTEGER DEFAULT 0, start_time TEXT,
                    end_time TEXT, duration_minutes INTEGER, model TEXT,
                    has_compaction INTEGER DEFAULT 0,
                    indexed_at TEXT NOT NULL, last_modified TEXT,
                    file_hash TEXT
                );
                CREATE TABLE session_topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL, topic TEXT NOT NULL,
                    captured_at TEXT NOT NULL, exchange_number INTEGER,
                    source TEXT NOT NULL
                );
                CREATE TABLE session_tools (
                    session_id TEXT NOT NULL, tool_name TEXT NOT NULL,
                    use_count INTEGER DEFAULT 0,
                    PRIMARY KEY (session_id, tool_name)
                );
                CREATE TABLE session_agents (
                    session_id TEXT NOT NULL, agent_name TEXT NOT NULL,
                    invocation_count INTEGER DEFAULT 0,
                    PRIMARY KEY (session_id, agent_name)
                );
                CREATE VIRTUAL TABLE session_content USING fts5(
                    session_id, content
                );
            """)
            conn.execute("""
                INSERT INTO sessions (
                    session_id, project, project_name, file_path, indexed_at
                ) VALUES ('legacy-id', 'legacy', 'Legacy', '/tmp/legacy.jsonl',
                          '2026-01-01T00:00:00')
            """)
            conn.execute(
                "INSERT INTO session_content VALUES (?, ?)",
                ("legacy-id", "legacy migration needle"),
            )
            conn.execute(
                "INSERT INTO session_tools VALUES (?, ?, ?)",
                ("legacy-id", "Read", 2),
            )
            conn.commit()
            conn.close()

            indexer = SessionIndexer(
                db_path=db_path,
                source_configs={
                    "claude": {"enabled": False, "root": str(CLAUDE_ROOT)},
                    "codex": {"enabled": False, "root": str(CODEX_ROOT)},
                },
            )
            indexer.connect()
            try:
                row = indexer.conn.execute(
                    "SELECT source, session_id FROM sessions"
                ).fetchone()
                self.assertEqual(tuple(row), ("claude", "legacy-id"))
                version = indexer.conn.execute(
                    "PRAGMA user_version"
                ).fetchone()[0]
                self.assertEqual(version, SCHEMA_VERSION)
                fts = indexer.conn.execute(
                    "SELECT source, session_id FROM session_content "
                    "WHERE session_content MATCH 'migration'"
                ).fetchone()
                self.assertEqual(tuple(fts), ("claude", "legacy-id"))
                tool = indexer.conn.execute(
                    "SELECT session_source, tool_name FROM session_tools"
                ).fetchone()
                self.assertEqual(tuple(tool), ("claude", "Read"))
            finally:
                indexer.close()

    def test_empty_v1_database_stays_eligible_for_backfill(self):
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "empty-legacy.db"
            conn = sqlite3.connect(db_path)
            conn.executescript("""
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    project TEXT, project_name TEXT, title TEXT,
                    title_display TEXT, tags TEXT, client TEXT,
                    file_path TEXT NOT NULL, file_size INTEGER,
                    exchange_count INTEGER DEFAULT 0, start_time TEXT,
                    end_time TEXT, duration_minutes INTEGER, model TEXT,
                    has_compaction INTEGER DEFAULT 0,
                    indexed_at TEXT NOT NULL, last_modified TEXT,
                    file_hash TEXT
                );
            """)
            conn.commit()
            conn.close()

            indexer = SessionIndexer(
                db_path=db_path,
                source_configs={
                    "claude": {"enabled": True, "root": str(CLAUDE_ROOT)},
                    "codex": {"enabled": False, "root": str(CODEX_ROOT)},
                },
            )
            indexer.connect()
            try:
                initialized = {
                    row[0] for row in indexer.conn.execute(
                        "SELECT source FROM index_state"
                    ).fetchall()
                }
            finally:
                indexer.close()

        # nothing was migrated, so the first run must still backfill claude
        self.assertEqual(initialized, set())

    def test_v1_upgrade_gains_columns_added_after_v2(self):
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "legacy-columns.db"
            conn = sqlite3.connect(db_path)
            conn.executescript("""
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    project TEXT, project_name TEXT, title TEXT,
                    title_display TEXT, tags TEXT, client TEXT,
                    file_path TEXT NOT NULL, file_size INTEGER,
                    exchange_count INTEGER DEFAULT 0, start_time TEXT,
                    end_time TEXT, duration_minutes INTEGER, model TEXT,
                    has_compaction INTEGER DEFAULT 0,
                    indexed_at TEXT NOT NULL, last_modified TEXT,
                    file_hash TEXT
                );
                CREATE VIRTUAL TABLE session_content USING fts5(
                    session_id, content
                );
            """)
            conn.execute("""
                INSERT INTO sessions (
                    session_id, project, project_name, file_path, indexed_at
                ) VALUES ('legacy-id', 'legacy', 'Legacy', '/tmp/legacy.jsonl',
                          '2026-01-01T00:00:00')
            """)
            conn.commit()
            conn.close()

            indexer = SessionIndexer(
                db_path=db_path, source_configs=source_configs()
            )
            indexer.connect()
            try:
                columns = {
                    row["name"] for row in
                    indexer.conn.execute("PRAGMA table_info(sessions)")
                }
                stats = indexer.backfill_all(progress_interval=0)
            finally:
                indexer.close()

        # a migrated v1 database rebuilds sessions and must not lose later columns
        self.assertIn("prose_chars", columns)
        self.assertIn("content_chars", columns)
        self.assertIn("parent_session_id", columns)
        self.assertEqual(stats["errors"], 0)
        self.assertEqual(stats["indexed"], 2)


class ConfigTests(unittest.TestCase):
    def tearDown(self):
        config._cached_config = None

    def test_legacy_projects_env_and_source_enablement(self):
        with tempfile.TemporaryDirectory() as tempdir:
            missing_config = Path(tempdir) / "missing.json"
            env = {
                "SESSION_INDEX_PROJECTS": "/tmp/legacy-claude",
                "SESSION_INDEX_CODEX_ROOT": "/tmp/custom-codex",
                "SESSION_INDEX_SOURCES": "codex",
            }
            with patch.object(config, "CONFIG_FILE", missing_config), patch.dict(
                os.environ, env, clear=True
            ):
                config._cached_config = None
                sources = config.get_source_configs()
            self.assertEqual(
                sources["claude"]["root"], "/tmp/legacy-claude"
            )
            self.assertFalse(sources["claude"]["enabled"])
            self.assertEqual(
                sources["codex"]["root"], "/tmp/custom-codex"
            )
            self.assertTrue(sources["codex"]["enabled"])


class SubagentTests(unittest.TestCase):
    def test_subagent_transcript_is_indexed_under_its_parent(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "projects"
            subagents = root / "-demo-project" / "parent-session" / "subagents"
            subagents.mkdir(parents=True)
            (subagents / "agent-abc123.jsonl").write_text("".join(
                json.dumps(record) + "\n" for record in (
                    {
                        "type": "user",
                        "isSidechain": True,
                        "sessionId": "parent-session",
                        "message": {"role": "user", "content": "find the leak"},
                        "timestamp": "2026-01-01T00:00:00Z",
                    },
                    {
                        "type": "assistant",
                        "isSidechain": True,
                        "sessionId": "parent-session",
                        "attributionAgent": "Explore",
                        "message": {"role": "assistant", "content": [
                            {"type": "text", "text": "found viridian-needle"},
                        ], "model": "claude-test", "usage": {
                            "input_tokens": 11,
                            "output_tokens": 22,
                            "cache_read_input_tokens": 33,
                        }},
                        "timestamp": "2026-01-01T00:01:00Z",
                    },
                )
            ))

            indexer = SessionIndexer(
                db_path=Path(tempdir) / "sessions.db",
                source_configs={
                    "claude": {"enabled": True, "root": str(root)},
                    "codex": {"enabled": False, "root": str(CODEX_ROOT)},
                },
            )
            indexer.connect()
            try:
                stats = indexer.backfill_all(progress_interval=0)
                row = indexer.conn.execute(
                    "SELECT parent_session_id, agent_name, project "
                    "FROM sessions WHERE session_id='agent-abc123'"
                ).fetchone()
            finally:
                indexer.close()

            self.assertEqual(stats["indexed"], 1)
            self.assertEqual(row["parent_session_id"], "parent-session")
            self.assertEqual(row["agent_name"], "Explore")
            self.assertEqual(row["project"], "-demo-project")

            search = SessionSearch(db_path=Path(tempdir) / "sessions.db")
            search.connect()
            try:
                results = search.search("viridian-needle")
            finally:
                search.close()

            self.assertEqual(len(results), 1)
            rendered = format_result(results[0])
            self.assertIn("[claude/subagent]", rendered)
            self.assertIn("agent Explore", rendered)
            self.assertIn("claude --resume parent-session", rendered)

    def test_subagent_mode_filters_search_and_find(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "projects"
            project = root / "-demo-project"
            subagents = project / "parent-session" / "subagents"
            subagents.mkdir(parents=True)
            for path, needle in (
                (project / "parent-session.jsonl", "cobalt-needle"),
                (subagents / "agent-abc123.jsonl", "viridian-needle"),
            ):
                record = {
                    "type": "user",
                    "message": {"role": "user", "content": f"{needle} shared"},
                    "timestamp": "2026-01-01T00:00:00Z",
                }
                if "subagents" in path.parts:
                    record["isSidechain"] = True
                    record["sessionId"] = "parent-session"
                path.write_text(json.dumps(record) + "\n")

            db_path = Path(tempdir) / "sessions.db"
            indexer = SessionIndexer(
                db_path=db_path,
                source_configs={
                    "claude": {"enabled": True, "root": str(root)},
                    "codex": {"enabled": False, "root": str(CODEX_ROOT)},
                },
            )
            indexer.connect()
            try:
                indexer.backfill_all(progress_interval=0)
            finally:
                indexer.close()

            search = SessionSearch(db_path=db_path)
            search.connect()
            try:
                modes = {
                    mode: {
                        row["session_id"]
                        for row in search.search("shared", subagents=mode)
                    }
                    for mode in ("include", "exclude", "only")
                }
                found = {
                    mode: {
                        row["session_id"]
                        for row in search.find(subagents=mode)
                    }
                    for mode in ("include", "exclude", "only")
                }
                with self.assertRaises(ValueError):
                    search.search("shared", subagents="sometimes")
            finally:
                search.close()

            for result in (modes, found):
                self.assertEqual(
                    result["include"], {"parent-session", "agent-abc123"}
                )
                self.assertEqual(result["exclude"], {"parent-session"})
                self.assertEqual(result["only"], {"agent-abc123"})


class UsageTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "sessions.db"
        indexer = SessionIndexer(
            db_path=self.db_path, source_configs=source_configs()
        )
        indexer.connect()
        indexer.backfill_all(progress_interval=0)
        indexer.close()
        self.search = SessionSearch(db_path=self.db_path)
        self.search.connect()

    def tearDown(self):
        self.search.close()
        self.tempdir.cleanup()

    def test_claude_usage_splits_cache_tiers(self):
        rows = {
            row["grouping"]: row
            for row in self.search.usage(by="model", source="claude")
        }
        row = rows["claude-test"]
        self.assertEqual(row["calls"], 2)
        self.assertEqual(row["input_tokens"], 10)
        self.assertEqual(row["output_tokens"], 10)
        # 100 with a 60/40 tier split, then 200 reported without one
        self.assertEqual(row["cache_write_tokens"], 300)
        self.assertEqual(row["cache_write_1h_tokens"], 60)
        self.assertEqual(row["cache_write_5m_tokens"], 40)
        self.assertEqual(row["cache_read_tokens"], 1800)

    def test_codex_usage_reports_uncached_input_and_reasoning(self):
        rows = {
            row["grouping"]: row
            for row in self.search.usage(by="model", source="codex")
        }
        row = rows["gpt-test"]
        # codex counts cached tokens inside input_tokens: 1000 - 700 - 50
        self.assertEqual(row["input_tokens"], 250)
        self.assertEqual(row["cache_read_tokens"], 700)
        self.assertEqual(row["cache_write_tokens"], 50)
        self.assertEqual(row["reasoning_tokens"], 30)
        self.assertEqual(row["output_tokens"], 40)

    def test_cache_write_total_never_undercounts_its_tiers(self):
        # Real transcripts report cache_creation_input_tokens=0 beside a tier
        totals = {}
        _add_claude_usage(totals, "m", {
            "cache_creation_input_tokens": 0,
            "cache_creation": {"ephemeral_1h_input_tokens": 916},
        })
        bucket = totals["m"]
        self.assertEqual(bucket["cache_write_tokens"], 916)
        self.assertEqual(bucket["cache_write_1h_tokens"], 916)

    def test_tool_tokens_attribute_write_and_injected_input(self):
        rows = {
            row["grouping"]: row
            for row in self.search.tool_tokens(source="claude")
        }
        row = rows["Read"]
        self.assertEqual(row["calls"], 1)
        # output tokens of the call that emitted the tool_use
        self.assertEqual(row["write_tokens"], 7)
        # billed input grew 1005 -> 1105 with only this result in between
        self.assertEqual(row["inject_tokens"], 100)
        self.assertEqual(row["result_bytes"], 400)

    def test_tool_tokens_by_session_when_one_tool_named(self):
        rows = self.search.tool_tokens(tool="Read")
        self.assertEqual([row["grouping"] for row in rows],
                         ["claude:shared-session"])
        self.assertEqual(rows[0]["inject_tokens"], 100)

    def test_codex_tool_result_bytes_are_measured(self):
        rows = {
            row["grouping"]: row
            for row in self.search.tool_tokens(source="codex")
        }
        self.assertGreater(rows["shell_command"]["result_bytes"], 0)
        # codex reports no per-call token split, so only payload size is known
        self.assertEqual(rows["shell_command"]["write_tokens"], 0)

    def test_unlinked_tool_result_lands_under_unknown(self):
        tokens = _ToolTokens()
        tokens.assistant({"content": [], "usage": {
            "input_tokens": 10, "output_tokens": 1,
        }})
        tokens.result({"content": [
            {"type": "tool_result", "tool_use_id": "absent", "content": "y" * 50},
        ]})
        tokens.assistant({"content": [], "usage": {
            "input_tokens": 60, "output_tokens": 1,
        }})
        self.assertEqual(tokens.totals["unknown"]["result_bytes"], 50)
        self.assertEqual(tokens.totals["unknown"]["inject_tokens"], 50)

    def test_bash_command_names_the_real_command(self):
        cases = {
            "cd /repo; sessions usage --by tool": "sessions",
            "cd ~/repo && git -C ~/repo log --oneline": "git log",
            "grep -rn foo . | head -20": "grep",
            'echo "=== a ==="; sqlite3 db "select 1"': "sqlite3",
            "AWS_PROFILE=eng aws s3 ls": "aws s3",
            "sudo /usr/local/bin/docker compose down": "docker compose",
            "cd /tmp": "cd",
            "": "",
        }
        for command, expected in cases.items():
            self.assertEqual(bash_command(command), expected, command)

    def test_canonical_tool_name_collapses_mcp_spellings(self):
        self.assertEqual(
            canonical_tool_name("mcp__codegraph__codegraph_search"),
            "codegraph.codegraph_search",
        )
        self.assertEqual(
            canonical_tool_name("codegraph.codegraph_search"),
            "codegraph.codegraph_search",
        )
        self.assertEqual(canonical_tool_name("Bash"), "Bash")

    def test_usage_groupings_and_rejects_unknown(self):
        for by in ("model", "source", "project", "session", "day"):
            self.assertTrue(self.search.usage(by=by))
        with self.assertRaises(ValueError):
            self.search.usage(by="wingspan")

    def test_orphaned_child_rows_are_swept(self):
        indexer = SessionIndexer(
            db_path=self.db_path, source_configs=source_configs()
        )
        indexer.connect()
        try:
            indexer.conn.execute("PRAGMA foreign_keys=OFF")
            indexer.conn.execute(
                "INSERT INTO session_tools (session_source, session_id, "
                "tool_name, use_count) VALUES ('claude', 'ghost', 'Bash', 99)"
            )
            indexer.conn.commit()
            indexer._sweep_orphans()
            remaining, = indexer.conn.execute(
                "SELECT COUNT(*) FROM session_tools WHERE session_id='ghost'"
            ).fetchone()
            violations = indexer.conn.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()
        finally:
            indexer.close()
        self.assertEqual(remaining, 0)
        self.assertEqual(violations, [])

    def test_usage_rows_disappear_with_their_session(self):
        self.search.conn.execute("PRAGMA foreign_keys=ON")
        self.search.conn.execute(
            "DELETE FROM sessions WHERE source='claude'"
        )
        self.assertEqual(self.search.usage(by="model", source="claude"), [])


class LongSessionTests(unittest.TestCase):
    def _write_long_transcript(self, root: Path) -> Path:
        (root / "-long-project").mkdir(parents=True)
        transcript = root / "-long-project" / "long-session.jsonl"
        filler = "reviewed the paginator and the tombstone sweep in detail. "
        with transcript.open("w") as handle:
            for index in range(4000):
                handle.write(json.dumps({
                    "type": "assistant",
                    "message": {"role": "assistant", "content": [
                        {"type": "text", "text": f"turn {index}: {filler * 2}"},
                    ]},
                    "timestamp": "2026-01-01T00:00:00Z",
                }) + "\n")
            handle.write(json.dumps({
                "type": "user",
                "message": {"role": "user", "content": "wrap up PR #4033"},
                "timestamp": "2026-01-01T01:00:00Z",
            }) + "\n")
        return transcript

    def _indexer(self, tempdir: str, root: Path) -> SessionIndexer:
        indexer = SessionIndexer(
            db_path=Path(tempdir) / "sessions.db",
            source_configs={
                "claude": {"enabled": True, "root": str(root)},
                "codex": {"enabled": False, "root": str(CODEX_ROOT)},
            },
        )
        indexer.connect()
        return indexer

    def test_prose_past_the_old_cap_stays_searchable(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "projects"
            self._write_long_transcript(root)
            indexer = self._indexer(tempdir, root)
            try:
                indexer.backfill_all(progress_interval=0)
                (content,) = indexer.conn.execute(
                    "SELECT content FROM session_content "
                    "WHERE session_id='long-session'"
                ).fetchone()
            finally:
                indexer.close()

            self.assertGreater(len(content), 100_000)
            self.assertIn("4033", content)

    def test_index_file_refreshes_a_row_whose_hash_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "projects"
            transcript = self._write_long_transcript(root)
            db_path = Path(tempdir) / "sessions.db"
            indexer = self._indexer(tempdir, root)
            try:
                indexer.backfill_all(progress_interval=0)
                indexer.conn.execute(
                    "UPDATE session_content SET content=substr(content,1,100000) "
                    "WHERE session_id='long-session'"
                )
                indexer.conn.commit()
                truncated = indexer.backfill_all(progress_interval=0)
            finally:
                indexer.close()

            self.assertEqual(truncated["indexed"], 0)

            argv = [
                "sessions", "--db-path", str(db_path),
                "index", "--claude-root", str(root), "--file", str(transcript),
            ]
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                cli.main()

            connection = sqlite3.connect(db_path)
            try:
                (content,) = connection.execute(
                    "SELECT content FROM session_content "
                    "WHERE session_id='long-session'"
                ).fetchone()
            finally:
                connection.close()

            self.assertIn("4033", content)


if __name__ == "__main__":
    unittest.main()
