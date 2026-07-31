---
name: session-index
description: Search and synthesize across local Claude Code and Codex sessions through the unified Agent Session Index. Use for questions such as "what did I try last time?" and return source-aware resume commands.
version: 0.5.0
author: Lee Fuhr
tags:
  - session-search
  - session-history
  - claude
  - codex
  - analytics
  - memory
requires:
  - python3
  - pip
---

# Agent Session Index skill

Use the local `sessions` CLI whenever the user asks about past agent sessions,
previous conversations, earlier attempts, or historical tool usage. The CLI
queries one private SQLite index containing both Claude Code and Codex
transcripts.

Do not upload transcript content or call an external synthesis API.

## What search actually does

`sessions "..."` fuses two rankings by reciprocal rank, then decays by session age.

- **Keyword (FTS5)** covers prose *and* a digest of tool activity: every command run,
  file path touched, URL fetched, plus the first 2KB of each tool result and any error
  lines past it. A PR number that only ever appeared in `gh pr view` output is findable.
- **Vector (optional)** covers prose only, so paraphrases match where no keyword does.
  Needs the `semantic` extra; without it search silently falls back to keyword alone.
- **Recency** multiplies the fused score by a 90-day half-life decay, so a fresh session
  wins a near-tie without burying a strongly-relevant old one.
- **Supersession** marks (never drops) an older result when a newer one in the same
  project repeats its topic. Look for `superseded by <id>` in the result metadata.

Queries under three words skip the vector layer — they are keyword lookups, and static
embeddings smear rare identifiers onto their subwords.

## Workflow

1. Extract a compact search phrase from the user's question.
2. Run the unified CLI:

```bash
sessions "relevant keywords" -n 10
```

3. If results are noisy, use source, metadata, or ranking filters:

```bash
sessions "relevant keywords" --source codex
sessions "relevant keywords" --days 30
sessions "relevant keywords" --no-recency
sessions "relevant keywords" --no-semantic
sessions find --project myapp --week --source claude
sessions find --tool shell_command --source codex
sessions recent 20 --source codex
```

Reach for `--no-recency` when the question is historical ("when did we first hit this"),
and `--days N` when it is about current state. `--half-life DAYS` tunes the decay.

4. Retrieve the actual exchanges for the most relevant results. Always pass the
   source shown in the result so the CLI uses the correct transcript parser:

```bash
sessions context <session_id> "search term" --source codex
sessions context <session_id> "search term" --source claude
```

5. Synthesize the retrieved excerpts inside the current Codex or Claude Code
   conversation. Summarize:

   - approaches tried
   - what worked and failed
   - recurring decisions or constraints
   - latest known state

6. Cite each historical source using the resume command printed by the CLI:

```text
claude --resume <session_id>
codex resume <session_id>
```

Present a concise answer rather than dumping raw CLI output. Clearly distinguish
facts from Claude sessions and Codex sessions when that matters.

## Analytics

```bash
sessions analytics
sessions analytics --week --source codex
sessions analytics --client "Acme" --source claude
sessions tools --source codex
sessions stats
```

## Installation and data

```bash
pip install 'agent-session-index[semantic]'
sessions index --backfill
sessions embed
```

- Database: `~/.session-index/sessions.db`
- Claude transcripts: `~/.claude/projects`
- Codex transcripts: `~/.codex/sessions` and `~/.codex/archived_sessions`
- Config: `~/.session-index/config.json`

`index --backfill` migrates an existing Claude-only database safely and initializes
newly enabled sources. `embed` builds the vector layer over already-indexed sessions
and is resumable — rerun it after new sessions land, or with `--rebuild` after changing
`embed_model`. Everything stays local; no text leaves the machine.

Without the `semantic` extra the base package is stdlib-only and search runs on FTS
alone. `sessions embed` then exits with the reason.

After changing an extraction rule, `--backfill` will NOT refresh a row whose transcript
hash and path still match. Force specific files with `sessions index --file <path>`
(repeatable), or loop `_iter_session_files` → `_parse_session` → `_upsert_session` in one
process to reparse everything (order of a minute per thousand transcripts). Never delete
rows first — a reaped transcript keeps its only surviving copy of the text in the index.

Archived sessions stay searchable. Claude Desktop's Archive only flags its own
sidebar record and leaves the transcript in `~/.claude/projects`; `codex` moves the
rollout into `archived_sessions/`, which is indexed as a second Codex root.

A *deleted* transcript still searches, because its indexed text lives in the database.
`sessions context` says the file is gone and falls back to that text, so the messages
read back without user/assistant pairing or tool calls.
