# Changelog

## 0.5.0

- Index a digest of tool activity, not just prose. Every tool call contributes its
  identifying arguments (command, file path, URL, pattern) and every result contributes
  its first 2KB plus any error lines that fell past that head. Previously only user and
  assistant prose was indexed, roughly 2.7% of transcript bytes, so a PR number that
  only ever appeared in `gh pr view` output was unfindable. Bulk payloads still drop out
  via the argument length cap, and the whole digest is capped at 2MB per session.
- Add an optional dense-vector layer behind the `semantic` extra, so paraphrases match
  where no keyword does. Chunks of prose are embedded with model2vec and stored in
  sqlite-vec; `sessions embed` builds or refreshes the layer and is resumable. Without
  the extra the package stays stdlib-only and search falls back to FTS alone.
- Fuse keyword and vector rankings by reciprocal rank rather than picking one. Only the
  prose half is embedded — tool output is lexical territory, and embedding it would
  quadruple the vector count for signal BM25 already handles. `sessions` gains
  `--no-semantic` to opt out.
- Reject dense hits past a cosine distance floor. Nearest-neighbour search always
  returns neighbours, so without a floor an unrelated query got confident nonsense
  instead of no results. Queries under three words skip the vector layer entirely, since
  static embeddings smear a rare identifier onto its subwords and pull in look-alikes.
- Decay the fused score by session age with a 90-day half-life, so a fresh session wins
  a near-tie without burying a strongly-relevant old one. `--no-recency`, `--half-life`,
  and `--days` tune or disable it; `recency_half_life_days` and `recency_weight` set the
  defaults in config.
- Set `recency_weight` to 0.1 by measurement rather than taste. Swept against 225
  historical-lookup queries, where recency is a liability by construction: 0.1 scores
  recall@1 42.2% against 41.3% for no recency at all and 35.6% at 0.5. A light touch
  breaks ties; a heavy one buries the right answer.
- Mark, never drop, a result that a newer session in the same project supersedes on an
  identical topic. Search had no notion of time at all, so a stale answer and its
  correction ranked purely on term frequency.
- `sessions` gains a `prose_chars` column recording where prose ends and the tool digest
  begins, so the vector layer can embed one half without re-parsing.

## 0.4.5

- Canonicalize MCP tool names to `<server>.<tool>`. Claude's `mcp__server__tool` and
  codex's `server.tool` are the same tool, so per-tool totals were splitting across
  sources; codegraph's 1,300 calls were spread over three spellings and looked absent.
- Count a codex MCP call once. One call emits both a `function_call` response item and an
  `mcp_tool_call_end` event with the same `call_id`, and both were counted, under
  different names. Calls are now keyed by `call_id`, and the result bytes carried inline
  on `mcp_tool_call_end` are measured instead of dropped.
- Add `sessions usage --by command`, breaking Bash down by what it ran, in a new
  `session_tool_detail` table. Leading `cd`/`echo` segments are skipped so the credited
  command is the real one, and only the first command of a chain is credited so the
  breakdown sums back to the invocation count.
- Sweep child rows whose session row is gone. The v1 migration ran with
  `foreign_keys=OFF` and left 2,245 orphans behind, worth 21,292 phantom tool calls in
  any aggregate that did not join sessions. `session_content` is left alone, since its
  text can be the last copy of a reaped transcript.


## 0.4.4

- Attribute token cost to individual tools. `session_tools` gains `write_tokens`,
  `inject_tokens`, and `result_bytes`; `sessions usage --by tool` splits the bill across
  tools and `--tool <name>` breaks one tool down by session. `tool_use_id` links a call
  to its result exactly, so no estimation is involved.
- `inject_tokens` is the billed input growth between consecutive calls, split across the
  results that arrived in between in proportion to payload size. Growth with no
  intervening result is left unattributed, and a result whose call is absent from the
  transcript lands under `unknown`, rather than either being smeared across real tools.


## 0.4.3

- Add `sessions usage`: per-session, per-model token counts captured during the existing
  parse pass, grouped by model, source, project, session, agent, or day. Both sources
  are normalized so `input_tokens` means uncached input and billed input is
  `input + cache_read + cache_write`.
- Record claude's 1h and 5m cache-write tiers separately, since they bill at different
  multipliers. `cache_write_tokens` takes the larger of the reported total and the sum
  of the tiers, because some entries report a zero total beside a nonzero tier.
- Discover subagent transcripts at any depth. Workflow agents live under
  `subagents/workflows/<id>/`, and some sessions nest a duplicate session directory;
  the previous single-level glob missed 329 files.


## 0.4.2

- Index subagent transcripts. `discover()` globbed one directory level, so every
  `<project>/<session>/subagents/agent-*.jsonl` was invisible: 1,354 files, 468MB, and
  21% of all recorded API calls. Their findings were unsearchable even though the parent
  conversation only summarized them.
- Record `parent_session_id` and `agent_name` on each session. A subagent transcript is
  not resumable, so search labels it `[claude/subagent]`, shows which agent ran, and
  prints the parent's resume command.
- Add `--subagents include|exclude|only` to `search`, `find`, and `recent`, and report
  the subagent count separately in `stats` so session totals stay readable.

## 0.4.1

- Raise the per-session FTS cap from 100K to 8M characters. Tool output is already
  excluded, so a session's indexed text is only ~1.5-2% of its transcript, but long
  sessions still blew past 100K and lost their late prose. A PR review that happened
  75% of the way through an 11MB session was unfindable, and 107 of 1641 indexed
  sessions sat at the cap.
- Accept repeated `--session` and `--file` on `sessions index`, to force a reindex of
  specific transcripts. `--backfill` skips anything whose content hash and path still
  match, so it cannot refresh rows after an extraction rule changes.

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
- Exclude the context Codex injects into user-role records — delegation payloads,
  skill bodies, plugin catalogs, and `AGENTS.md` repository configuration. These
  passed the sanitizer and were both searchable and picked as session titles, so
  sessions were named `<recommended_plugins>` rather than by their prompt.
- Keep a Codex rollout's own identity. A subagent rollout replays its parent's
  `session_meta`, and every such record overwrote the session id, cwd, start time,
  and metadata — so children adopted the parent's id and collided onto one row,
  silently dropping sessions. Only the first metadata record is now read.
- Count an MCP invocation once. A single call emits both a `function_call` and an
  `mcp_tool_call_end`, and each incremented the tool tally separately.
- Index a Codex rollout whose turns exist only as `user_message`/`agent_message`
  events, as review and exec subagent rollouts do. They previously produced no
  title, no exchanges, and no searchable text at all.
- Leave a migrated v1 database uninitialized so its first run reparses it. The
  migration copies FTS built by the old rules, which indexed system-like prompts
  and omitted assistant text, and marking Claude initialized froze that in place.
- Restore the project and tool breakdowns in `sessions stats`, dropped in the
  source-aware rewrite while `get_stats()` still computed them.
- Fall back to literal matching when a `sessions context` query is not a valid
  regex, rather than silently returning unfiltered text.
- Derive Codex compaction from `compacted` records and `context_compacted` events.
  `turn_context.summary` is a setting whose value is `auto` or `none`, so reading it
  as a summary marked uncompacted sessions as compacted and stored the setting as
  the session topic.
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
