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
- Codex: `~/.codex/sessions`, plus the sibling `~/.codex/archived_sessions`
- Unified database: `~/.session-index/sessions.db`

Archiving a session never removes it from the index. The Claude Desktop app archives
by flagging its own sidebar record and leaves the transcript in place, and `codex`
moves the rollout into `archived_sessions/`, which is indexed as a second Codex root.

Deleting a transcript is what costs you something. The row and its searchable text
stay, so `sessions context` still reads the messages back out of the index, but
without exchange pairing or tool calls.

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

# Subagent transcripts are indexed and searchable; control whether they surface
sessions "needle" --subagents exclude
sessions "needle" --subagents only
sessions find --project ns --subagents only

# Token usage, per session x model, from the same parse pass
sessions usage
sessions usage --by session -n 10
sessions usage --by agent --week
sessions usage --by day --days 30 --subagents exclude
sessions usage --by tool
sessions usage --tool Read
sessions usage --by command
sessions usage --by command --tool Bash --week

# Index all enabled sources, or one source
sessions index
sessions index --backfill
sessions index --backfill --source codex
sessions index --session <id> --source claude

# Force a reindex of specific transcripts, whose content hash has not changed
sessions index --file <path> --file <path>
```

`--session` and `--file` both bypass the unchanged-content check, so they are the
way to refresh rows after an extraction rule changes. `--backfill` will not: it
skips any transcript whose hash and path still match the indexed row.

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

The Codex root's `archived_sessions` sibling is derived from whatever root is in
effect, so an override picks up that installation's archived rollouts too.

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

## Attribution

Agent Session Index is adapted from
[Claude Session Index](https://github.com/lee-fuhr/claude-session-index) by
Lee Fuhr. The original project was released under the MIT License, and its
copyright and license notice are retained in [LICENSE](LICENSE).

## License

MIT

## Token usage

`sessions usage` aggregates the token counts each transcript already records, so no
extra pass over the transcripts is needed. Groupings: `model`, `source`, `project`,
`session`, `agent`, `day`.

Both sources are normalized onto one convention, where `input_tokens` counts uncached
input only and billed input is `input_tokens + cache_read_tokens + cache_write_tokens`.
Codex reports cached and cache-written tokens inside its `input_tokens`, so those are
subtracted on the way in.

Claude records the cache-write tiers separately, which cost different multipliers:
`cache_write_1h_tokens` and `cache_write_5m_tokens`. `cache_write_tokens` never
undercounts their sum, because some entries report a zero total beside a nonzero tier.

### Per tool

`sessions usage --by tool` splits the bill across tools, and `--tool <name>` breaks one
tool down by session. Three separate costs, because they answer different questions:

- `write` - output tokens the model spent emitting the tool call, taken from the usage
  of the message that carried the `tool_use` block and split across the tools in it.
- `inject` - billed input growth the result caused, measured as the delta in
  `input + cache_write + cache_read` between consecutive calls and split across the
  results that arrived in between, in proportion to payload size.
- `result` - raw payload bytes the tool returned.

`inject` is a first-read cost. A result then sits in context and is re-read on every
later turn, which is where most of the cache-read total comes from; that amortized cost
is not measured here.

Context growth with no tool result in between - user messages, thinking blocks - is not
attributed to any tool. A result whose call is not in the same transcript, from before a
compaction or emitted by a parent session, lands under `unknown` rather than being
smeared across the known tools.

Codex records no per-call token split, so its tools report `result` bytes only.

MCP tool names are canonicalized to `<server>.<tool>`. Claude writes
`mcp__codegraph__codegraph_search` and codex writes `codegraph.codegraph_search` for the
same tool, so without this a tool's totals split across sources and each half looks
small. Codex also emits both a `function_call` and an `mcp_tool_call_end` for one call,
so calls are keyed by `call_id` and counted once.

### Bash by command

`sessions usage --by command` breaks Bash down by what it actually ran. Leading
navigation and output decoration are skipped, so `cd repo && grep -rn foo` is credited to
grep, not cd. Only the first real command of a pipeline or `&&` chain is credited, so the
breakdown sums back to the invocation count for every transcript still on disk. Tools
whose transcript was reaped keep their `session_tools` count but can have no breakdown.
`--tool <name>` selects a different tool to break down.

Caveats. Codex reports one cumulative total per rollout rather than per call, so a
codex session gets a single row under its last known model, and `calls` counts model
turns rather than API calls. Sessions whose transcript was deleted keep their indexed
text but have no usage rows.
