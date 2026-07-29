# Changelog

## 0.4.0

- Rename the package and user-facing product to Agent Session Index.
- Add a source adapter boundary and local Codex rollout support.
- Index Codex rollouts that `codex` archived into `archived_sessions/`, discovered as
  a sibling of whichever Codex root is in effect.
- Refresh `file_path` when a transcript moves. The unchanged-check compared only the
  content hash, so a relocated transcript — an archived Codex rollout, a renamed
  Claude project directory — kept a row pointing at a path that no longer existed,
  and `sessions context` could not read it back.
- Fall back to the indexed text in `sessions context` when the transcript itself is
  gone, instead of printing a read error and no exchanges. Retention sweeps delete
  transcripts while the row and its searchable content remain.
- Migrate existing Claude-only databases to source-qualified identities.
- Add source labels and filters to search, context, recent, find, analytics,
  tools, topics, stats, and indexing commands.
- Exclude developer/system instructions, reasoning records, tool outputs, and
  large encoded payloads from FTS.
- Keep all indexing and synthesis workflows local; transcript uploads are
  disabled.

## v0.3.1 — Stop titling everything "## Curation Data"

- **Smarter title auto-generation** — skips markdown headers, agent system prompts, and system caveats when picking a title from user messages. Tries up to 5 messages before giving up.
- Previously, 84+ sessions were titled "## Curation Data" and 20+ were titled "You are QA testing...". Now those get proper titles or null instead of garbage.

## v0.3.0 — Sessions have names now

The biggest annoyance is fixed: most sessions showed "(unnamed)" because only manually-titled sessions had display names. Now titles are auto-generated from compaction summaries or the first user message. Re-index with `sessions index --backfill` to see the difference.

- **Auto-generated session titles** — compaction summaries get parsed (including JSON blobs), and untitled sessions fall back to the first user message. No more walls of "(unnamed)".
- **`--days N` filter** — `sessions find --days 14` for arbitrary date ranges, not just `--week`
- **`--exclude-project` filter** — `sessions find --exclude-project "share memory"` to cut the noise
- **FTS5 crash fixes** — queries with periods (`CLAUDE.md`), hyphens (`session-index`), and reserved words (`index`) no longer crash. All search terms get quoted for safe literal matching.
- **CLI flag parsing fix** — `sessions "query" -n 5` now works correctly (the flag value was getting split from the flag)
- **`npx skills add` support** — skill moved to `skills/session-index/SKILL.md` to match the skills.sh registry convention. Install with `npx skills add lee-fuhr/claude-session-index`.
- **Conversational interface as primary UX** — README and skill rewritten to emphasize the natural language experience in Claude Code. The CLI is still there for power users.

## v0.2.0 — It looks good now

The output got a proper makeover. Conversations read like conversations. Analytics have visual hierarchy. And you don't have to set anything up anymore.

- **One command to rule them all** — `sessions` replaces the old `session-search` / `session-analyze` / `session-index` trio. Plain text defaults to search: `sessions "webhook debugging"` just works.
- Chat-like conversation display with 🧑/🤖 markers — you can actually tell who said what
- Box-drawing characters for session cards and exchange blocks
- Section headers with emoji in analytics (📊 📈 🔧 💬) for scannable output
- Cleaner search results with ◆ bullets and → resume commands
- Auto-indexing on first use — no more separate `--backfill` step, just run any command and it handles the rest
- Better stats display with visual structure instead of raw JSON
- The old commands still work if you prefer them

## v0.1.0 — Initial release

- Full-text search across all Claude Code sessions (SQLite + FTS5)
- Filter by client, project, tool, agent, tag, date
- Conversation context retrieval from session JSONL files
- Analytics: time per client, tool trends, session frequency, topic analysis
- Cross-session synthesis via Anthropic API (optional dependency)
- Live topic capture via Claude Code hooks (UserPromptSubmit, PreCompact, SessionEnd)
- Background indexing via macOS LaunchAgent
- Claude Code skill for natural language session queries
- Configurable paths via CLI flags, env vars, or config file
