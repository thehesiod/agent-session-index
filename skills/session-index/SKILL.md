---
name: session-index
description: Search and synthesize across local Claude Code and Codex sessions through the unified Agent Session Index. Use for questions such as "what did I try last time?" and return source-aware resume commands.
version: 0.4.0
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
queries one private SQLite/FTS index containing both Claude Code and Codex
transcripts.

Do not upload transcript content or call an external synthesis API.

## Workflow

1. Extract a compact search phrase from the user's question.
2. Run the unified CLI:

```bash
sessions "relevant keywords" -n 10
```

3. If results are noisy, use source or metadata filters:

```bash
sessions "relevant keywords" --source codex
sessions find --project myapp --week --source claude
sessions find --tool shell_command --source codex
sessions recent 20 --source codex
```

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
pip install agent-session-index
sessions index --backfill
```

- Database: `~/.session-index/sessions.db`
- Claude transcripts: `~/.claude/projects`
- Codex transcripts: `~/.codex/sessions` and `~/.codex/archived_sessions`
- Config: `~/.session-index/config.json`

The first command migrates an existing Claude-only database safely and
initializes newly enabled sources. Indexing and retrieval remain local.

Archived sessions stay searchable. Claude Desktop's Archive only flags its own
sidebar record and leaves the transcript in `~/.claude/projects`; `codex` moves the
rollout into `archived_sessions/`, which is indexed as a second Codex root. What does
drop out is a *deleted* transcript — the row and its indexed text survive, so search
still finds it, but `sessions context` reads the file and returns no exchanges.
