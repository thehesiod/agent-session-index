# Agent Session Index

**Search your local Claude Code and Codex history from one private index.**

Agent Session Index discovers local transcripts from both tools, normalizes
them into one SQLite database, and exposes fast FTS5 search, source-aware
conversation context, filters, and analytics.

The indexer is local-only. It does not upload sessions or use a network API.
Developer/system instructions, reasoning records, tool results, and large
binary or encoded payloads are excluded from full-text search.

## Quick start

```bash
pip install agent-session-index
sessions index --backfill
sessions "webhook debugging"
```

Default transcript roots:

- Claude Code: `~/.claude/projects`
- Codex: `~/.codex/sessions`
- Unified database: `~/.session-index/sessions.db`

Existing databases created by `claude-session-index` are migrated in place.
Their existing rows are backfilled with `source=claude`; the Codex source is
then initialized without discarding Claude data.

## CLI

Plain text defaults to full-text search:

```bash
sessions "webhook debugging"
sessions "webhook debugging" --source codex
sessions "webhook debugging" --source claude --context
```

Results are labeled `[claude]` or `[codex]` and include the appropriate resume
command:

```text
◆ [codex] 019f82f6 · Fix service config merge
  2026-07-20 · cfg-clobber-fix · 18 exchanges
  → codex resume 019f82f6-f2e6-79f1-a241-14d378de3ae0
```

Source filters are available throughout the CLI:

```bash
# Read matching exchanges with the correct source parser
sessions context <id> "search term" --source codex
sessions context claude:<id>

# Browse and filter
sessions recent 20 --source codex
sessions find --tool shell_command --source codex
sessions find --project myapp --week --source claude
sessions topics <id> --source claude

# Analytics, tools, and database statistics
sessions analytics --week --source codex
sessions tools --source codex
sessions stats
sessions stats --source claude

# Index all enabled sources, or one source
sessions index
sessions index --backfill
sessions index --backfill --source codex
sessions index --session <id> --source claude
```

The legacy `session-index`, `session-search`, `session-analyze`, and
`session-topic-capture` entry points remain available.

## Configuration

Configuration lives at `~/.session-index/config.json`. Both sources are enabled
by default:

```json
{
  "db_path": "~/.session-index/sessions.db",
  "topics_dir": "~/.claude/session-topics",
  "clients": [],
  "project_names": {},
  "sources": {
    "claude": {
      "enabled": true,
      "root": "~/.claude/projects"
    },
    "codex": {
      "enabled": true,
      "root": "~/.codex/sessions"
    }
  }
}
```

Environment overrides:

```bash
SESSION_INDEX_DB=/path/to/sessions.db
SESSION_INDEX_CLAUDE_ROOT=/path/to/claude/projects
SESSION_INDEX_CODEX_ROOT=/path/to/codex/sessions
SESSION_INDEX_SOURCES=claude,codex
SESSION_INDEX_CLAUDE_ENABLED=true
SESSION_INDEX_CODEX_ENABLED=false
```

`SESSION_INDEX_PROJECTS` and the old top-level `projects_dir` config key remain
supported as aliases for the Claude root.

One-off root overrides are available on indexing commands:

```bash
sessions index --backfill \
  --claude-root /path/to/.claude/projects \
  --codex-root /path/to/.codex/sessions
```

## What gets indexed

The normalized session record includes:

- source and source-qualified session ID
- transcript path and size
- project, CWD, timestamps, duration, model, and safe metadata
- user and assistant text
- bounded compaction/rollout summaries
- tool and agent invocation counts
- optional Claude topic-capture entries

The FTS document is bounded and sanitized. The following are not added:

- Codex developer/system messages or Claude system-like injected prompts
- Codex encrypted reasoning or reasoning summaries
- tool arguments, tool results, and custom tool outputs
- data URLs, large base64-like strings, and binary/control payloads

## Local synthesis

The old network-backed `sessions synthesize` behavior is disabled. To answer
cross-session questions without uploading transcripts:

1. Run `sessions "topic" -n 10`.
2. Retrieve relevant excerpts with
   `sessions context <id> "topic" --source <source>`.
3. Synthesize those local results inside the current Claude Code or Codex
   conversation.

The companion skill in `skills/session-index/SKILL.md` follows this workflow.

## Claude topic hooks

Live topic capture remains available for Claude Code. Merge
`hooks/settings-snippet.json` into `~/.claude/settings.json` and ensure
`session-topic-capture` is on `PATH`. Hook-created topics are stored against
the Claude source and continue to work with migrated databases.

## Development

```bash
python3 -m unittest discover -v
python3 -m compileall -q session_index
```

Tests use sanitized Claude and Codex JSONL fixtures and cover adapters,
source-qualified collisions, FTS filtering, context routing, CLI labels,
analytics, configuration, and schema migration.

## License

MIT
