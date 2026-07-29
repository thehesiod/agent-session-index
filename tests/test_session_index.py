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
from session_index.analyzer import analytics, get_context, synthesize
from session_index.indexer import SCHEMA_VERSION, SessionIndexer
from session_index.search import SessionSearch
from session_index.sources import ClaudeSourceAdapter, CodexSourceAdapter


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
        for forbidden in (
            "forbidden-codex-developer",
            "forbidden-codex-base-instructions",
            "forbidden-encrypted-reasoning",
            "forbidden-reasoning-summary",
            "forbidden-tool-output",
            "huge-binary-tool-payload",
            "forbidden-environment-context",
            "forbidden-mcp-arguments",
            "forbidden-mcp-result",
        ):
            self.assertNotIn(forbidden, data["fts_content"])
        metadata = json.loads(data["metadata_json"])
        self.assertEqual(metadata["git"]["branch"], "fixture")
        self.assertNotIn("base_instructions", metadata)

    def test_codex_adapter_discovers_archived_rollouts(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "sessions"
            archived = Path(tempdir) / "archived_sessions"
            root.mkdir()
            archived.mkdir()
            (root / "live.jsonl").write_text("")
            (archived / "archived.jsonl").write_text("")

            found = {path.name for path in CodexSourceAdapter(root).discover()}

        self.assertEqual(found, {"live.jsonl", "archived.jsonl"})


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
                search.topics("shared-session", source="codex")[0]["source"],
                "rollout_summary",
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


if __name__ == "__main__":
    unittest.main()
