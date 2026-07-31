"""Source adapters for local agent session transcripts.

Adapters normalize source-specific JSONL into the small, local-only shape used
by the SQLite index.  Deliberately excluded from indexed text:

* system/developer instructions
* encrypted or plaintext reasoning records
* tool results and tool arguments
* binary/data-URL payloads
* context the agent injects into user-role records (delegation payloads, skill
  bodies, plugin catalogs, repository configuration)
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

MAX_MESSAGE_CHARS = 20_000
# Prose is ~1.5-2% of transcript bytes once tool I/O is excluded; largest seen is 4.9MB.
MAX_FTS_CHARS = 8_000_000
MAX_SUMMARY_CHARS = 2_000

_DATA_URL_RE = re.compile(
    r"data:[^;\s]+;base64,[A-Za-z0-9+/=\r\n]{256,}", re.IGNORECASE
)
_LONG_BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/=]{2048,}")
_SYSTEM_BLOCK_RE = re.compile(
    r"<(?:environment_context|permissions instructions|app-context|"
    r"collaboration_mode|INSTRUCTIONS)>.*?</(?:environment_context|"
    r"permissions instructions|app-context|collaboration_mode|INSTRUCTIONS)>",
    re.IGNORECASE | re.DOTALL,
)

# Codex injects these into role="user" records; Codex-scoped as the names are generic.
_CODEX_INJECTED_TAGS = (
    "codex_delegation",
    "realtime_delegation",
    "recommended_plugins",
    "skill",
    "subagent_notification",
    "task",
    "turn_aborted",
    "user_action",
)
_CODEX_INJECTED_BLOCK_RE = re.compile(
    r"<(?P<tag>" + "|".join(_CODEX_INJECTED_TAGS) + r")>.*?</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)
_CODEX_UNCLOSED_BLOCK_RE = re.compile(
    r"<(?:" + "|".join(_CODEX_INJECTED_TAGS) + r")>.*\Z",
    re.IGNORECASE | re.DOTALL,
)

_SKIP_TITLE_PREFIXES = (
    "#",
    "You are",
    "Caveat:",
    "Explore the",
    "<environment_context>",
    "<INSTRUCTIONS>",
)


def sanitize_text(value, max_chars: int = MAX_MESSAGE_CHARS) -> str:
    """Return bounded readable text without binary-looking payloads."""
    if not isinstance(value, str):
        return ""
    if "\x00" in value:
        return ""
    value = _SYSTEM_BLOCK_RE.sub("", value)
    value = _DATA_URL_RE.sub("[binary payload omitted]", value)
    value = _LONG_BASE64_RE.sub("[encoded payload omitted]", value)
    value = "".join(
        ch for ch in value if ch in "\n\r\t" or ord(ch) >= 32
    ).strip()
    if len(value) > max_chars:
        return value[:max_chars] + "..."
    return value


def _read_jsonl(path: Path) -> Iterable[dict]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    yield item
    except OSError as exc:
        print(f"Error reading {path}: {exc}", file=sys.stderr)


def _duration_minutes(start_time: str | None, end_time: str | None) -> int | None:
    if not start_time or not end_time:
        return None
    try:
        start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        return max(0, int((end - start).total_seconds() / 60))
    except (TypeError, ValueError):
        return None


def _pick_title(prompts: list[str]) -> str | None:
    for prompt in prompts[:5]:
        first_line = prompt.strip().splitlines()[0].strip() if prompt.strip() else ""
        if len(first_line) <= 10:
            continue
        if any(first_line.startswith(prefix) for prefix in _SKIP_TITLE_PREFIXES):
            continue
        return first_line[:77] + "..." if len(first_line) > 80 else first_line
    return None


def _looks_like_system_prompt(text: str) -> bool:
    stripped = text.lstrip()
    return any(stripped.startswith(prefix) for prefix in (
        "You are ",
        "Caveat:",
        "Explore the ",
        "<environment_context>",
        "<INSTRUCTIONS>",
        "# AGENTS.md instructions",
        "The following is the Codex agent history",
    ))


def strip_codex_injected_context(text: str) -> str:
    """Return user text with Codex-injected wrapper blocks removed."""
    text = _CODEX_INJECTED_BLOCK_RE.sub("", text)
    # sanitize_text truncates before this runs, so an opener can outlive its closer
    return _CODEX_UNCLOSED_BLOCK_RE.sub("", text).strip()


def _detect_client(clients: list[str], prompts: list[str], project_name: str) -> str | None:
    if not clients:
        return None
    haystack = (" ".join(prompts[:10]) + " " + project_name).lower()
    return next((client for client in clients if client.lower() in haystack), None)


USAGE_FIELDS = (
    "calls", "input_tokens", "output_tokens", "cache_write_tokens",
    "cache_write_1h_tokens", "cache_write_5m_tokens", "cache_read_tokens",
    "reasoning_tokens",
)


def _usage_bucket(totals: dict, model: str | None) -> dict:
    return totals.setdefault(
        sanitize_text(model or "", 100) or "unknown",
        dict.fromkeys(USAGE_FIELDS, 0),
    )


def _add_claude_usage(totals: dict, model: str | None, usage) -> None:
    if not isinstance(usage, dict):
        return
    bucket = _usage_bucket(totals, model)
    creation = usage.get("cache_creation") or {}
    tier_1h = creation.get("ephemeral_1h_input_tokens") or 0
    tier_5m = creation.get("ephemeral_5m_input_tokens") or 0
    bucket["calls"] += 1
    bucket["input_tokens"] += usage.get("input_tokens") or 0
    bucket["output_tokens"] += usage.get("output_tokens") or 0
    # Some entries report cache_creation_input_tokens=0 alongside a nonzero tier
    bucket["cache_write_tokens"] += max(
        usage.get("cache_creation_input_tokens") or 0, tier_1h + tier_5m
    )
    bucket["cache_write_1h_tokens"] += tier_1h
    bucket["cache_write_5m_tokens"] += tier_5m
    bucket["cache_read_tokens"] += usage.get("cache_read_input_tokens") or 0


TOOL_TOKEN_FIELDS = ("write_tokens", "inject_tokens", "result_bytes")
DETAIL_FIELDS = TOOL_TOKEN_FIELDS + ("use_count",)
_MCP_PREFIX_RE = re.compile(r"^mcp__(?P<server>.+?)__(?P<tool>.+)$")

MAX_TOOL_RESULT_CHARS = 2_000
MAX_TOOL_ARG_CHARS = 400
MAX_TOOL_SCAN_CHARS = 200_000
MAX_TOOL_DIGEST_CHARS = 2_000_000
MAX_ERROR_LINES = 20
_DIGEST_ARG_KEYS = (
    "command", "file_path", "path", "notebook_path", "pattern", "glob",
    "query", "url", "description", "prompt", "subagent_type", "skill",
)
_ERROR_LINE_RE = re.compile(
    r"(?im)^.*\b(?:error|traceback|exception|failed|failure|fatal|denied"
    r"|refused|timeout|not found|no such)\b.*$"
)


def canonical_tool_name(name: str) -> str:
    """Collapse per-harness MCP spellings onto one <server>.<tool> name.

    Claude writes mcp__<server>__<tool>; codex writes <server>.<tool> for the
    same tool, so without this a tool's totals split across sources.
    """
    match = _MCP_PREFIX_RE.match(name or "")
    if not match:
        return name
    return f"{match['server']}.{match['tool']}"


def _mcp_result_size(result) -> int:
    if isinstance(result, dict):
        for key in ("Ok", "Err", "ok", "err"):
            if key in result:
                return _mcp_result_size(result[key])
        if "content" in result:
            return _tool_result_size(result["content"])
    return _tool_result_size(result)


def _tool_result_size(value) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(
            len(part.get("text") or "") if isinstance(part, dict) else len(str(part))
            for part in value
        )
    return len(str(value or ""))


def _tool_result_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            (part.get("text") or "") if isinstance(part, dict) else str(part)
            for part in value
        )
    return str(value or "")


def _mcp_result_text(result) -> str:
    if isinstance(result, dict):
        for key in ("Ok", "Err", "ok", "err"):
            if key in result:
                return _mcp_result_text(result[key])
        if "content" in result:
            return _tool_result_text(result["content"])
    return _tool_result_text(result)


def _tool_call_digest(name: str, tool_input) -> str:
    """One searchable line naming what a tool call acted on.

    Values are what make a call findable later — the command that ran, the file
    that was read, the URL fetched — so they are kept while bulk payloads
    (file contents on Write, diffs on Edit) are dropped by the length cap.
    """
    if not isinstance(tool_input, dict):
        return f"[{name}]"
    parts = []
    for key in _DIGEST_ARG_KEYS:
        value = tool_input.get(key)
        if isinstance(value, list):
            value = " ".join(str(item) for item in value)
        if isinstance(value, str) and value.strip():
            parts.append(f"{key}={sanitize_text(value, MAX_TOOL_ARG_CHARS)}")
    return f"[{name}] " + " ".join(parts) if parts else f"[{name}]"


def _codex_call_args(payload: dict) -> dict:
    """codex serializes tool arguments as a JSON string on the call item."""
    arguments = payload.get("arguments")
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {}
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return {"command": arguments}
    return parsed if isinstance(parsed, dict) else {"command": arguments}


def _tool_result_digest(text: str) -> str:
    """Head of a tool result plus any error lines that fell past the head.

    Errors are what a later search is usually hunting for and they surface at
    the end of long output, which the head cap would otherwise cut.
    """
    text = sanitize_text(text, MAX_TOOL_SCAN_CHARS) if text else ""
    if not text:
        return ""
    head = text[:MAX_TOOL_RESULT_CHARS]
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return head
    tail_errors = _ERROR_LINE_RE.findall(text[MAX_TOOL_RESULT_CHARS:])
    if not tail_errors:
        return head
    kept = "\n".join(line.strip()[:MAX_TOOL_ARG_CHARS] for line in tail_errors[:MAX_ERROR_LINES])
    return f"{head}\n{kept}"


def _build_digest(parts: list[str]) -> str:
    digest = "\n".join(part for part in parts if part)
    return digest[:MAX_TOOL_DIGEST_CHARS]


_SUBCOMMAND_DRIVERS = frozenset((
    "git", "gh", "docker", "aws", "uv", "npm", "pnpm", "yarn", "cargo", "go",
    "kubectl", "terraform", "brew", "pip", "systemctl", "overmind", "tmux",
    "bun", "poetry", "helm", "gcloud", "az", "make",
))
_VALUE_FLAGS = frozenset(("-C", "-c", "--git-dir", "--work-tree", "-p", "--profile"))
_PREFIX_COMMANDS = frozenset((
    "cd", "pushd", "popd", "export", "source", ".", "set", "unset",
    "echo", "printf", "true", ":",
))
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SEGMENT_RE = re.compile(r"\s*(?:&&|\|\||[|;\n])\s*")


def _segment_command(segment: str) -> str:
    tokens = [token for token in segment.split() if token]
    while tokens and (
        _ASSIGNMENT_RE.match(tokens[0])
        or tokens[0] in ("sudo", "command", "time", "exec", "env")
    ):
        tokens.pop(0)
    if not tokens:
        return ""
    name = tokens[0].split("/")[-1]
    if name not in _SUBCOMMAND_DRIVERS:
        return name
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in _VALUE_FLAGS:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return f"{name} {token}"
    return name


def bash_command(command) -> str:
    """Name the command a bash invocation is really running.

    Leading navigation and output decoration are skipped, so `cd x && grep y`
    reports grep rather than cd. Only the first real command is credited, so a
    per-command breakdown sums back to the invocation count.
    """
    if not isinstance(command, str) or not command.strip():
        return ""
    names = [
        name for name in (
            _segment_command(segment)
            for segment in _SEGMENT_RE.split(command.strip())
        ) if name
    ]
    if not names:
        return ""
    return next((name for name in names if name not in _PREFIX_COMMANDS), names[0])


class _ToolTokens:
    """Attribute token cost to individual tools within one transcript.

    write_tokens are the output tokens of the call that emitted a tool_use,
    split across the tools it emitted. inject_tokens are the billed input growth
    between consecutive calls, split across the results that arrived in between
    in proportion to payload size. A result whose call is not in this file lands
    under "unknown" rather than being smeared across the known tools.
    """

    def __init__(self):
        self.totals: dict[str, dict[str, int]] = {}
        self.details: dict[tuple[str, str], dict[str, int]] = {}
        self._pending: dict[str, tuple[str, str]] = {}
        self._since_call: list[tuple[str, str, int]] = []
        self._context: int | None = None

    def bucket(self, name: str) -> dict:
        return self.totals.setdefault(name, dict.fromkeys(TOOL_TOKEN_FIELDS, 0))

    def _add(self, name: str, detail: str, field: str, value: int) -> None:
        bucket = self.bucket(name)
        if field in bucket:
            bucket[field] += value
        if detail:
            self.details.setdefault(
                (name, detail), dict.fromkeys(DETAIL_FIELDS, 0)
            )[field] += value

    @staticmethod
    def _identify(block: dict) -> tuple[str, str]:
        name = canonical_tool_name(block.get("name") or "") or "unknown"
        if name != "Bash":
            return name, ""
        return name, bash_command((block.get("input") or {}).get("command"))

    def _attribute_growth(self, context: int) -> None:
        if self._context is not None and self._since_call:
            delta = max(0, context - self._context)
            total = sum(size for _, _, size in self._since_call) or 1
            for name, detail, size in self._since_call:
                self._add(name, detail, "inject_tokens", delta * size // total)
        self._context = context
        self._since_call = []

    def assistant(self, message: dict) -> None:
        content = message.get("content")
        blocks = [
            block for block in (content if isinstance(content, list) else [])
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        usage = message.get("usage")
        if isinstance(usage, dict):
            self._attribute_growth(
                (usage.get("input_tokens") or 0)
                + (usage.get("cache_creation_input_tokens") or 0)
                + (usage.get("cache_read_input_tokens") or 0)
            )
            if blocks:
                share = (usage.get("output_tokens") or 0) // len(blocks)
                for block in blocks:
                    name, detail = self._identify(block)
                    self._add(name, detail, "write_tokens", share)
        for block in blocks:
            name, detail = self._identify(block)
            self._pending[block.get("id")] = (name, detail)
            self._add(name, detail, "use_count", 1)

    def result(self, message: dict) -> None:
        content = message.get("content")
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            name, detail = self._pending.pop(
                block.get("tool_use_id"), ("unknown", "")
            )
            size = _tool_result_size(block.get("content"))
            self._add(name, detail, "result_bytes", size)
            self._since_call.append((name, detail, size))


def _codex_usage(model: str | None, totals: dict, turns: int) -> dict:
    """Normalize a codex rollout total onto the claude convention.

    Codex counts cached and cache-written tokens inside input_tokens; claude
    reports them alongside it. Subtract so input_tokens means uncached input in
    both sources and billed input stays input + cache_read + cache_write.
    """
    if not totals:
        return {}
    cached = totals.get("cached_input_tokens") or 0
    written = totals.get("cache_write_input_tokens") or 0
    usage: dict[str, dict[str, int]] = {}
    bucket = _usage_bucket(usage, model)
    bucket["calls"] = turns
    bucket["input_tokens"] = max(
        0, (totals.get("input_tokens") or 0) - cached - written
    )
    bucket["output_tokens"] = totals.get("output_tokens") or 0
    bucket["cache_write_tokens"] = written
    bucket["cache_read_tokens"] = cached
    bucket["reasoning_tokens"] = totals.get("reasoning_output_tokens") or 0
    return usage


def _build_fts(messages: list[str]) -> str:
    content = "\n".join(part for part in messages if part)
    return content[:MAX_FTS_CHARS]


def _combine_fts(messages: list[str], tool_messages: list[str]) -> tuple[str, int]:
    """Prose first, then the tool digest, with the boundary reported.

    The semantic index embeds only the prose half; tool output is lexical
    territory (error strings, paths, PR numbers) and embedding it would
    quadruple the vector count for signal BM25 already handles.
    """
    prose = _build_fts(messages)
    digest = _build_digest(tool_messages)
    if not digest:
        return prose, len(prose)
    return f"{prose}\n{digest}", len(prose)


def _pair_entries(entries: list[dict], query: str | None, limit: int,
                  max_chars: int) -> list[dict]:
    exchanges = []
    index = 0
    while index < len(entries):
        if entries[index]["type"] != "user":
            index += 1
            continue
        user_text = entries[index]["text"]
        timestamp = entries[index]["timestamp"]
        assistant_parts = []
        cursor = index + 1
        while cursor < len(entries) and entries[cursor]["type"] == "assistant":
            if entries[cursor]["text"]:
                assistant_parts.append(entries[cursor]["text"])
            cursor += 1
        exchanges.append({
            "user": user_text,
            "assistant": "\n".join(assistant_parts),
            "timestamp": timestamp,
        })
        index = cursor

    if query:
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            pattern = None
        if pattern:
            matched = [
                item for item in exchanges
                if pattern.search(item["user"]) or pattern.search(item["assistant"])
            ]
        else:
            matched = []
        if not matched:
            lowered = query.lower()
            matched = [
                item for item in exchanges
                if lowered in item["user"].lower()
                or lowered in item["assistant"].lower()
            ]
        if not matched:
            words = [word.lower() for word in query.split() if len(word) > 2]
            matched = [
                item for item in exchanges
                if any(
                    word in item["user"].lower()
                    or word in item["assistant"].lower()
                    for word in words
                )
            ]
        exchanges = matched

    for exchange in exchanges:
        for role in ("user", "assistant"):
            if len(exchange[role]) > max_chars:
                exchange[role] = exchange[role][:max_chars] + "..."
    return exchanges[:limit]


def _claude_text(content, assistant: bool = False) -> tuple[str, list[str]]:
    if isinstance(content, str):
        return sanitize_text(content), []
    parts = []
    tool_names = []
    if not isinstance(content, list):
        return "", tool_names
    for block in content:
        if isinstance(block, str):
            parts.append(sanitize_text(block))
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(sanitize_text(block.get("text", "")))
        elif assistant and isinstance(block, dict) and block.get("type") == "tool_use":
            name = canonical_tool_name(sanitize_text(block.get("name", ""), 200))
            if name:
                tool_names.append(name)
                parts.append(f"[{name}]")
    return "\n".join(part for part in parts if part), tool_names


def _codex_message_text(payload: dict) -> str:
    parts = []
    content = payload.get("content", [])
    if isinstance(content, str):
        return sanitize_text(content)
    if not isinstance(content, list):
        return ""
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("input_text", "output_text", "text"):
            text = sanitize_text(block.get("text", ""))
            if text:
                parts.append(text)
    return "\n".join(parts)


def _tool_agent_name(payload: dict) -> str | None:
    name = payload.get("name", "")
    if "spawn_agent" not in name and name != "Task":
        return None
    raw = payload.get("arguments") or payload.get("input") or ""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict):
        return None
    return sanitize_text(
        raw.get("task_name") or raw.get("subagent_type") or raw.get("name") or "",
        200,
    ) or None


class SessionSourceAdapter:
    """Base interface implemented by transcript sources."""

    source: str

    def __init__(self, root: Path, project_names: dict[str, str] | None = None,
                 clients: list[str] | None = None):
        self.root = Path(root).expanduser()
        self.project_names = project_names or {}
        self.clients = clients or []

    def roots(self) -> Iterable[Path]:
        """Every directory this source discovers transcripts under."""
        yield self.root

    def owns(self, path: Path) -> bool:
        return any(path.is_relative_to(root) for root in self.roots())

    def discover(self) -> Iterable[Path]:
        raise NotImplementedError

    def session_id_from_path(self, path: Path) -> Optional[str]:
        """Session id derivable from the filename, or None if it needs a parse."""
        return None

    def parse(self, path: Path) -> Optional[dict]:
        raise NotImplementedError

    def extract_exchanges(self, path: Path, query: str | None = None,
                          limit: int = 10, max_chars: int = 1000) -> list[dict]:
        raise NotImplementedError


class ClaudeSourceAdapter(SessionSourceAdapter):
    source = "claude"

    def discover(self) -> Iterable[Path]:
        if not self.root.exists():
            return []
        return (
            path
            for project_dir in self.root.iterdir()
            if project_dir.is_dir()
            # subagents/ nests at varying depths; pathlib.glob ignores symlinks
            for pattern in ("*.jsonl", "**/subagents/**/*.jsonl")
            for path in project_dir.glob(pattern)
        )

    def session_id_from_path(self, path: Path) -> Optional[str]:
        return path.stem

    def _project(self, path: Path) -> str:
        # Subagent transcripts sit two levels deeper, under <session>/subagents/
        try:
            return path.relative_to(self.root).parts[0]
        except ValueError:
            return path.parent.name

    def parse(self, path: Path) -> Optional[dict]:
        session_id = path.stem
        project = self._project(path)
        project_name = self.project_names.get(project, project)
        user_prompts: list[str] = []
        fts_messages: list[str] = []
        tool_messages: list[str] = []
        tools: dict[str, int] = {}
        agents: dict[str, int] = {}
        summaries: list[str] = []
        exchange_count = 0
        start_time = end_time = model = cwd = None
        title = title_display = tags = None
        parent_session_id = agent_name = None
        usage: dict[str, dict[str, int]] = {}
        tool_tokens = _ToolTokens()

        for entry in _read_jsonl(path):
            entry_type = entry.get("type")
            timestamp = entry.get("timestamp")
            if timestamp:
                start_time = start_time or timestamp
                end_time = timestamp
            cwd = cwd or entry.get("cwd")
            if entry.get("isSidechain"):
                if not parent_session_id and entry.get("sessionId") != session_id:
                    parent_session_id = entry.get("sessionId")
                agent_name = agent_name or sanitize_text(
                    entry.get("attributionAgent", ""), 200
                ) or None

            if entry_type == "custom-title":
                title_display = sanitize_text(entry.get("customTitle", ""), 500)
                if title_display.startswith(">>>"):
                    parts = title_display.split("......")
                    title = parts[0].replace(">>>", "").strip()
                    if len(parts) > 1:
                        tags = parts[-1].strip().strip("[]<>").strip() or None
                else:
                    title = title_display
            elif entry_type == "summary":
                summary = sanitize_text(entry.get("summary", ""), MAX_SUMMARY_CHARS)
                if summary:
                    if summary.startswith("{"):
                        try:
                            parsed = json.loads(summary)
                            summary = sanitize_text(
                                parsed.get("title") or parsed.get("summary") or summary,
                                MAX_SUMMARY_CHARS,
                            )
                        except (json.JSONDecodeError, AttributeError):
                            pass
                    summaries.append(summary)
            elif entry_type in ("user", "assistant"):
                exchange_count += 1
                message = entry.get("message", {})
                text, tool_names = _claude_text(
                    message.get("content", ""), assistant=entry_type == "assistant"
                )
                if entry_type == "assistant":
                    model = model or message.get("model")
                    _add_claude_usage(
                        usage, message.get("model"), message.get("usage")
                    )
                    tool_tokens.assistant(message)
                else:
                    tool_tokens.result(message)
                    if text and not _looks_like_system_prompt(text):
                        user_prompts.append(text)
                if text and not (
                    entry_type == "user" and _looks_like_system_prompt(text)
                ):
                    fts_messages.append(text)
                for tool_name in tool_names:
                    tools[tool_name] = tools.get(tool_name, 0) + 1

                content = message.get("content")
                blocks = content if isinstance(content, list) else []
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if entry_type == "assistant" and block_type == "tool_use":
                        tool_input = block.get("input") or {}
                        tool_messages.append(_tool_call_digest(
                            canonical_tool_name(
                                sanitize_text(block.get("name", ""), 200)
                            ),
                            tool_input,
                        ))
                        if block.get("name") == "Task":
                            agent = sanitize_text(
                                tool_input.get("subagent_type", ""), 200
                            )
                            if agent:
                                agents[agent] = agents.get(agent, 0) + 1
                    elif entry_type == "user" and block_type == "tool_result":
                        tool_messages.append(_tool_result_digest(
                            _tool_result_text(block.get("content"))
                        ))

        if not title_display and summaries:
            title_display = summaries[0][:80]
            title = title_display
        elif not title_display:
            title_display = _pick_title(user_prompts)
            title = title_display
        if not title_display and agent_name:
            title = title_display = f"{agent_name} subagent"

        topics = [{
            "topic": summary[:120],
            "source": "compaction_summary",
            "captured_at": end_time or datetime.now().isoformat(),
            "exchange_number": None,
        } for summary in summaries]
        fts_content, prose_chars = _combine_fts(fts_messages, tool_messages)

        return {
            "source": self.source,
            "session_id": session_id,
            "parent_session_id": parent_session_id,
            "agent_name": agent_name,
            "project": project,
            "project_name": project_name,
            "cwd": cwd,
            "title": title,
            "title_display": title_display,
            "tags": tags,
            "client": _detect_client(self.clients, user_prompts, project_name),
            "file_path": str(path),
            "file_size": path.stat().st_size,
            "exchange_count": exchange_count,
            "start_time": start_time,
            "end_time": end_time,
            "duration_minutes": _duration_minutes(start_time, end_time),
            "model": model,
            "has_compaction": int(bool(summaries)),
            "metadata_json": json.dumps({"format": "claude-code"}, sort_keys=True),
            "tools": tools,
            "tool_tokens": tool_tokens.totals,
            "tool_details": tool_tokens.details,
            "agents": agents,
            "usage": usage,
            "fts_content": fts_content,
            "prose_chars": prose_chars,
            "topics": topics,
        }

    def extract_exchanges(self, path: Path, query: str | None = None,
                          limit: int = 10, max_chars: int = 1000) -> list[dict]:
        entries = []
        for entry in _read_jsonl(path):
            entry_type = entry.get("type")
            if entry_type not in ("user", "assistant"):
                continue
            text, _ = _claude_text(
                (entry.get("message") or {}).get("content", ""),
                assistant=entry_type == "assistant",
            )
            if entry_type == "user" and _looks_like_system_prompt(text):
                continue
            entries.append({
                "type": entry_type,
                "text": text,
                "timestamp": entry.get("timestamp", ""),
            })
        return _pair_entries(entries, query, limit, max_chars)


class CodexSourceAdapter(SessionSourceAdapter):
    source = "codex"

    # codex archives a session by MOVING its rollout into a sibling archived_sessions/
    ARCHIVED_DIR = "archived_sessions"

    # rollout-<timestamp>-<session id>.jsonl; the id also appears in session_meta
    ROLLOUT_ID_RE = re.compile(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    )

    def roots(self) -> Iterable[Path]:
        yield self.root
        archived = self.root.parent / self.ARCHIVED_DIR
        if not archived.is_relative_to(self.root):
            yield archived

    def discover(self) -> Iterable[Path]:
        for root in self.roots():
            if root.exists():
                yield from root.rglob("*.jsonl")

    def session_id_from_path(self, path: Path) -> Optional[str]:
        match = self.ROLLOUT_ID_RE.search(path.stem)
        return match.group(0) if match else None

    def parse(self, path: Path) -> Optional[dict]:
        session_id = path.stem
        cwd = model = start_time = end_time = None
        user_prompts: list[str] = []
        fts_messages: list[str] = []
        summaries: list[str] = []
        tools: dict[str, int] = {}
        agents: dict[str, int] = {}
        exchange_count = 0
        saw_compaction = False
        saw_session_meta = False
        metadata: dict = {"format": "codex-rollout"}
        event_messages: list[tuple[str, str]] = []
        # total_token_usage is cumulative per rollout, so keep the largest record
        best_usage: dict = {}
        turns = 0
        previous_turn = None
        tool_tokens = _ToolTokens()
        tool_messages: list[str] = []
        # one MCP call emits a function_call and an mcp_tool_call_end sharing a call_id
        codex_calls: dict[str, dict] = {}

        for entry in _read_jsonl(path):
            timestamp = entry.get("timestamp")
            if timestamp:
                start_time = start_time or timestamp
                end_time = timestamp
            entry_type = entry.get("type")
            payload = entry.get("payload") or {}

            if entry_type == "session_meta":
                # a subagent rollout replays its parent's meta; the first record is this session's
                if saw_session_meta:
                    continue
                saw_session_meta = True
                session_id = payload.get("id") or session_id
                cwd = payload.get("cwd") or cwd
                start_time = payload.get("timestamp") or start_time
                for key in (
                    "originator", "model_provider", "git", "thread_source",
                    "parent_thread_id", "source",
                ):
                    if payload.get(key) is not None:
                        metadata[key] = payload[key]
            elif entry_type == "turn_context":
                cwd = payload.get("cwd") or cwd
                model = payload.get("model") or model
                # payload["summary"] is a setting ("auto"/"none"), not a summary
            elif entry_type == "compacted":
                saw_compaction = True
                summary = sanitize_text(
                    payload.get("message", ""), MAX_SUMMARY_CHARS
                )
                if summary and summary not in summaries:
                    summaries.append(summary)
            elif entry_type == "response_item":
                item_type = payload.get("type")
                if item_type == "message" and payload.get("role") in ("user", "assistant"):
                    text = _codex_message_text(payload)
                    role = payload["role"]
                    if role == "user":
                        text = strip_codex_injected_context(text)
                        if _looks_like_system_prompt(text):
                            continue
                    if not text:
                        continue
                    exchange_count += 1
                    if role == "user":
                        user_prompts.append(text)
                    fts_messages.append(text)
                elif (
                    item_type in ("function_call", "custom_tool_call")
                    or (isinstance(item_type, str) and item_type.endswith("_call"))
                ):
                    name = sanitize_text(
                        payload.get("name") or item_type, 200
                    )
                    if name:
                        call_id = payload.get("call_id") or f"anon-{len(codex_calls)}"
                        record = codex_calls.setdefault(
                            call_id,
                            {"name": name, "result_bytes": 0, "canonical": False},
                        )
                        if not record["canonical"]:
                            record["name"] = name
                        tool_messages.append(
                            _tool_call_digest(name, _codex_call_args(payload))
                        )
                    agent = _tool_agent_name(payload)
                    if agent:
                        agents[agent] = agents.get(agent, 0) + 1
                elif isinstance(item_type, str) and item_type.endswith("_output"):
                    record = codex_calls.get(payload.get("call_id"))
                    if record is not None:
                        record["result_bytes"] += _tool_result_size(
                            payload.get("output")
                        )
                        tool_messages.append(_tool_result_digest(
                            _tool_result_text(payload.get("output"))
                        ))
                # reasoning, tool outputs, and developer/system messages are
                # intentionally excluded.
            elif (
                entry_type == "event_msg"
                and payload.get("type") == "context_compacted"
            ):
                saw_compaction = True
            elif (
                entry_type == "event_msg"
                and payload.get("type") in ("user_message", "agent_message")
            ):
                text = sanitize_text(payload.get("message", ""))
                if text:
                    role = ("user" if payload["type"] == "user_message"
                            else "assistant")
                    event_messages.append((role, text))
            elif (
                entry_type == "event_msg"
                and payload.get("type") == "token_count"
            ):
                info = payload.get("info") or {}
                totals = info.get("total_token_usage") or {}
                if (totals.get("total_tokens") or 0) > (
                    best_usage.get("total_tokens") or 0
                ):
                    best_usage = totals
                turn = info.get("last_token_usage") or {}
                if (turn.get("total_tokens") or 0) and turn != previous_turn:
                    turns += 1
                    previous_turn = turn
            elif (
                entry_type == "event_msg"
                and payload.get("type") == "mcp_tool_call_end"
            ):
                invocation = payload.get("invocation") or {}
                server = sanitize_text(invocation.get("server", ""), 100)
                tool = sanitize_text(invocation.get("tool", ""), 100)
                name = ".".join(part for part in (server, tool) if part)
                if name:
                    call_id = payload.get("call_id") or f"mcp-{len(codex_calls)}"
                    record = codex_calls.setdefault(
                        call_id,
                        {"name": name, "result_bytes": 0, "canonical": True},
                    )
                    record["name"] = name
                    record["canonical"] = True
                    record["result_bytes"] += _mcp_result_size(
                        payload.get("result")
                    )
                    tool_messages.append(_tool_result_digest(
                        _mcp_result_text(payload.get("result"))
                    ))

        # review/exec subagent rollouts carry their turns only as events
        if not fts_messages:
            for role, text in event_messages:
                if role == "user":
                    text = strip_codex_injected_context(text)
                    if _looks_like_system_prompt(text):
                        continue
                if not text:
                    continue
                exchange_count += 1
                if role == "user":
                    user_prompts.append(text)
                fts_messages.append(text)

        for record in codex_calls.values():
            tools[record["name"]] = tools.get(record["name"], 0) + 1
            tool_tokens.bucket(record["name"])["result_bytes"] += (
                record["result_bytes"]
            )

        fts_content, prose_chars = _combine_fts(fts_messages, tool_messages)
        project = cwd or path.parent.name
        project_name = Path(cwd).name if cwd else path.parent.name
        title = _pick_title(user_prompts)
        topics = [{
            "topic": summary.splitlines()[0][:120],
            "source": "compaction_summary",
            "captured_at": end_time or datetime.now().isoformat(),
            "exchange_number": None,
        } for summary in summaries if summary.strip()]

        return {
            "source": self.source,
            "session_id": session_id,
            "project": project,
            "project_name": project_name,
            "cwd": cwd,
            "title": title,
            "title_display": title,
            "tags": None,
            "client": _detect_client(self.clients, user_prompts, project_name),
            "file_path": str(path),
            "file_size": path.stat().st_size,
            "exchange_count": exchange_count,
            "start_time": start_time,
            "end_time": end_time,
            "duration_minutes": _duration_minutes(start_time, end_time),
            "model": model,
            "has_compaction": int(saw_compaction),
            "metadata_json": json.dumps(metadata, sort_keys=True),
            "tools": tools,
            "tool_tokens": tool_tokens.totals,
            "tool_details": tool_tokens.details,
            "agents": agents,
            "usage": _codex_usage(model, best_usage, turns),
            "fts_content": fts_content,
            "prose_chars": prose_chars,
            "topics": topics,
        }

    def extract_exchanges(self, path: Path, query: str | None = None,
                          limit: int = 10, max_chars: int = 1000) -> list[dict]:
        entries = []
        event_entries = []
        saw_response_message = False
        for entry in _read_jsonl(path):
            payload = entry.get("payload") or {}
            if entry.get("type") == "event_msg":
                event_type = payload.get("type")
                if event_type == "mcp_tool_call_end":
                    invocation = payload.get("invocation") or {}
                    server = sanitize_text(invocation.get("server", ""), 100)
                    tool = sanitize_text(invocation.get("tool", ""), 100)
                    name = ".".join(part for part in (server, tool) if part)
                    if name:
                        entries.append({
                            "type": "assistant",
                            "text": f"[{name}]",
                            "timestamp": entry.get("timestamp", ""),
                        })
                elif event_type in ("user_message", "agent_message"):
                    role = ("user" if event_type == "user_message"
                            else "assistant")
                    text = sanitize_text(payload.get("message", ""))
                    if role == "user":
                        text = strip_codex_injected_context(text)
                        if _looks_like_system_prompt(text):
                            continue
                    if text:
                        event_entries.append({
                            "type": role,
                            "text": text,
                            "timestamp": entry.get("timestamp", ""),
                        })
                continue
            if entry.get("type") != "response_item":
                continue
            item_type = payload.get("type")
            role = payload.get("role")
            if item_type == "message" and role in ("user", "assistant"):
                text = _codex_message_text(payload)
                if role == "user":
                    text = strip_codex_injected_context(text)
                    if _looks_like_system_prompt(text):
                        continue
                if text:
                    saw_response_message = True
                    entries.append({
                        "type": role,
                        "text": text,
                        "timestamp": entry.get("timestamp", ""),
                    })
            elif (
                item_type in ("function_call", "custom_tool_call")
                or (isinstance(item_type, str) and item_type.endswith("_call"))
            ):
                name = sanitize_text(payload.get("name") or item_type, 200)
                if name:
                    entries.append({
                        "type": "assistant",
                        "text": f"[{name}]",
                        "timestamp": entry.get("timestamp", ""),
                    })
        if not saw_response_message and event_entries:
            entries = sorted(
                entries + event_entries, key=lambda item: item["timestamp"]
            )
        return _pair_entries(entries, query, limit, max_chars)


ADAPTER_TYPES = {
    "claude": ClaudeSourceAdapter,
    "codex": CodexSourceAdapter,
}


def build_adapters(source_configs: dict, project_names: dict[str, str] | None = None,
                   clients: list[str] | None = None) -> dict[str, SessionSourceAdapter]:
    adapters = {}
    for source, adapter_type in ADAPTER_TYPES.items():
        settings = source_configs.get(source, {})
        if not settings.get("enabled", True):
            continue
        root = settings.get("root")
        if root:
            adapters[source] = adapter_type(
                Path(root), project_names=project_names, clients=clients
            )
    return adapters


def extract_exchanges_for_source(source: str, path: str | Path,
                                 query: str | None = None, limit: int = 10,
                                 max_chars: int = 1000) -> list[dict]:
    adapter_type = ADAPTER_TYPES.get(source)
    if not adapter_type:
        return []
    path = Path(path)
    return adapter_type(path.parent).extract_exchanges(
        path, query=query, limit=limit, max_chars=max_chars
    )
