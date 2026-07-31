"""Tool digest, dense-vector layer, and rank fusion."""

from __future__ import annotations

import sqlite3
import unittest
from datetime import datetime, timedelta

from session_index import search as search_module
from session_index.indexer import SessionIndexer
from session_index.search import SessionSearch
from session_index import semantic
from session_index.search import _mark_superseded, _recency_factor
from session_index.sources import (
    _build_digest,
    _codex_call_args,
    _combine_fts,
    _tool_call_digest,
    _tool_result_digest,
)


class ToolDigestTests(unittest.TestCase):
    def test_call_digest_keeps_identifying_arguments(self):
        digest = _tool_call_digest(
            "Bash", {"command": "gh pr view 128", "description": "check PR"}
        )
        self.assertIn("gh pr view 128", digest)
        self.assertIn("[Bash]", digest)

    def test_call_digest_flattens_list_arguments(self):
        digest = _tool_call_digest("shell", {"command": ["bash", "-lc", "ls -la"]})
        self.assertIn("bash -lc ls -la", digest)

    def test_call_digest_drops_bulk_payload_arguments(self):
        digest = _tool_call_digest("Write", {"content": "x" * 5000})
        self.assertEqual(digest, "[Write]")

    def test_result_digest_keeps_head_and_trailing_errors(self):
        body = "\n".join(f"line {n} of routine output" for n in range(400))
        body += "\nERROR: connection refused to host db-1\n"
        digest = _tool_result_digest(body)
        self.assertLess(len(digest), len(body))
        self.assertTrue(digest.startswith("line 0 of routine output"))
        self.assertIn("connection refused", digest)

    def test_result_digest_is_bounded_for_short_output(self):
        self.assertEqual(_tool_result_digest("all good"), "all good")
        self.assertEqual(_tool_result_digest(""), "")

    def test_digest_respects_the_per_session_cap(self):
        parts = ["y" * 10_000 for _ in range(500)]
        self.assertLessEqual(len(_build_digest(parts)), 2_000_000)

    def test_combine_reports_the_prose_boundary(self):
        content, prose_chars = _combine_fts(["hello world"], ["[Bash] command=ls"])
        self.assertEqual(content[:prose_chars], "hello world")
        self.assertIn("command=ls", content[prose_chars:])

    def test_combine_without_tool_output_leaves_no_boundary_gap(self):
        content, prose_chars = _combine_fts(["only prose"], [])
        self.assertEqual(prose_chars, len(content))

    def test_codex_arguments_parse_from_json_string(self):
        args = _codex_call_args({"arguments": '{"command": ["bash", "-lc", "id"]}'})
        self.assertEqual(args["command"], ["bash", "-lc", "id"])

    def test_codex_arguments_survive_unparseable_json(self):
        self.assertEqual(
            _codex_call_args({"arguments": "not json"}), {"command": "not json"}
        )


class RecencyTests(unittest.TestCase):
    def test_fresh_sessions_outrank_stale_ones_at_equal_relevance(self):
        now = datetime.now().isoformat()
        old = (datetime.now() - timedelta(days=365)).isoformat()
        self.assertGreater(
            _recency_factor(now, 90, 0.5), _recency_factor(old, 90, 0.5)
        )

    def test_decay_halves_at_the_half_life(self):
        aged = (datetime.now() - timedelta(days=90)).isoformat()
        self.assertAlmostEqual(_recency_factor(aged, 90, 1.0), 1.5, places=2)

    def test_zero_weight_disables_the_adjustment(self):
        old = (datetime.now() - timedelta(days=999)).isoformat()
        self.assertEqual(_recency_factor(old, 90, 0.0), 1.0)

    def test_unparseable_timestamps_do_not_reorder(self):
        self.assertEqual(_recency_factor("not-a-date", 90, 0.5), 1.0)
        self.assertEqual(_recency_factor(None, 90, 0.5), 1.0)


class SupersessionTests(unittest.TestCase):
    @staticmethod
    def _result(session_id, start_time, topic, project="ns"):
        return {
            "session_id": session_id,
            "source": "claude",
            "start_time": start_time,
            "project_name": project,
            "topics": [{"topic": topic, "source": "compaction_summary"}],
        }

    def test_newer_session_supersedes_older_on_the_same_topic(self):
        results = _mark_superseded([
            self._result("old", "2026-01-01T00:00:00Z", "cache eviction rollout"),
            self._result("new", "2026-07-01T00:00:00Z", "cache eviction rollout"),
        ])
        by_id = {item["session_id"]: item for item in results}
        self.assertEqual(by_id["old"]["superseded_by"], "new")
        self.assertNotIn("superseded_by", by_id["new"])

    def test_a_different_project_is_not_superseded(self):
        results = _mark_superseded([
            self._result("a", "2026-01-01T00:00:00Z", "same topic", project="one"),
            self._result("b", "2026-07-01T00:00:00Z", "same topic", project="two"),
        ])
        self.assertFalse(any("superseded_by" in item for item in results))

    def test_supersession_marks_but_never_drops(self):
        results = _mark_superseded([
            self._result("old", "2026-01-01T00:00:00Z", "shared"),
            self._result("new", "2026-07-01T00:00:00Z", "shared"),
        ])
        self.assertEqual(len(results), 2)


class QueryGateTests(unittest.TestCase):
    def test_identifier_queries_stay_below_the_semantic_gate(self):
        for query in ("amber-needle", "TICKET-4821", "worker OOM"):
            self.assertLess(
                len(query.split()), search_module.MIN_SEMANTIC_QUERY_TOKENS
            )

    def test_natural_language_queries_clear_the_gate(self):
        for query in ("why did the job run out of memory",
                      "OOM worker scheduler restart"):
            self.assertGreaterEqual(
                len(query.split()), search_module.MIN_SEMANTIC_QUERY_TOKENS
            )


class RecencyAppliesWithoutVectorsTests(unittest.TestCase):
    """Recency must rank the FTS-only path too, not just the fused one."""

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        indexer = SessionIndexer(embed=False)
        indexer.conn = self.conn
        indexer._create_schema()
        old = (datetime.now() - timedelta(days=900)).isoformat()
        new = (datetime.now() - timedelta(days=1)).isoformat()
        for session_id, start in (("stale", old), ("fresh", new)):
            self.conn.execute(
                "INSERT INTO sessions (source, session_id, project, project_name,"
                " file_path, start_time, indexed_at, content_chars)"
                " VALUES ('claude', ?, 'p', 'p', '/tmp/x', ?, ?, 40)",
                (session_id, start, start),
            )
            self.conn.execute(
                "INSERT INTO session_content (source, session_id, content)"
                " VALUES ('claude', ?, 'shared vocabulary about widget latency')",
                (session_id,),
            )
        self.conn.commit()
        self.search = SessionSearch()
        self.search.conn = self.conn

    def tearDown(self):
        self.conn.close()

    def test_recency_reorders_when_no_vectors_exist(self):
        results = self.search.search(
            "shared vocabulary about widget latency",
            semantic_search=False, recency=True,
        )
        self.assertEqual([r["session_id"] for r in results][0], "fresh")

    def test_disabling_recency_leaves_both_results(self):
        results = self.search.search(
            "shared vocabulary about widget latency",
            semantic_search=False, recency=False,
        )
        self.assertEqual(len(results), 2)


class ChunkingTests(unittest.TestCase):
    def test_chunks_cover_the_text_and_overlap(self):
        text = "\n".join(f"paragraph {n} with enough words to matter here"
                         for n in range(200))
        chunks = semantic.chunk_prose(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(piece) >= semantic.MIN_CHUNK_CHARS
                            for _, piece in chunks))
        self.assertTrue(all(piece in text or piece.strip() in text
                            for _, piece in chunks))

    def test_offsets_are_monotonic(self):
        text = "z" * 10_000
        offsets = [offset for offset, _ in semantic.chunk_prose(text)]
        self.assertEqual(offsets, sorted(offsets))

    def test_line_dense_text_does_not_stall_the_stride(self):
        text = "\n".join("short line here" for _ in range(4000))
        chunks = semantic.chunk_prose(text)
        stride = len(text) / len(chunks)
        self.assertGreater(stride, semantic.CHUNK_CHARS / 2)

    def test_chunks_never_fall_below_the_overlap(self):
        text = "\n".join(f"line {n}" for n in range(3000))
        for _, piece in semantic.chunk_prose(text)[:-1]:
            self.assertGreater(len(piece), semantic.CHUNK_OVERLAP)

    def test_trivial_input_yields_no_chunks(self):
        self.assertEqual(semantic.chunk_prose(""), [])
        self.assertEqual(semantic.chunk_prose("tiny"), [])


@unittest.skipUnless(
    semantic.available() and semantic.load_vec_extension(sqlite3.connect(":memory:")),
    "semantic extra not installed",
)
class SemanticIndexTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.index = semantic.SemanticIndex(self.conn)
        self.index.ensure_schema(semantic.model_dims())

    def tearDown(self):
        self.conn.close()

    def test_paraphrase_retrieves_a_session_with_no_shared_keywords(self):
        self.index.index_session("claude", "oom", (
            "The ingest task was killed by the kernel because the container "
            "exceeded its memory limit and had to be restarted repeatedly. "
            "We raised the task memory reservation to stop the churn."
        ))
        self.index.index_session("claude", "unrelated", (
            "We discussed the colour of the dashboard buttons and settled on "
            "a muted blue palette for the navigation bar and the sidebar."
        ))
        hits = self.index.search("why did the job run out of RAM", limit=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0][1], "oom")

    def test_an_unrelated_query_returns_nothing(self):
        self.index.index_session("claude", "only", (
            "Postgres vacuum settings and autovacuum thresholds were tuned "
            "for the reporting tables after the bloat incident last week."
        ))
        self.assertEqual(
            self.index.search("medieval french tapestry restoration", limit=5), []
        )

    def test_reindexing_replaces_rather_than_duplicates(self):
        self.index.index_session("claude", "s", "first body of prose " * 20)
        first = self.index.stats()["chunks"]
        self.index.index_session("claude", "s", "first body of prose " * 20)
        self.assertEqual(self.index.stats()["chunks"], first)

    def test_clearing_a_session_removes_its_vectors(self):
        self.index.index_session("claude", "s", "some prose to embed " * 20)
        self.index.clear_session("claude", "s")
        self.assertEqual(self.index.stats()["chunks"], 0)


if __name__ == "__main__":
    unittest.main()
