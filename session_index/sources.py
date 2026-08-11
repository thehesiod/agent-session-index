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
MAX_FTS_CHARS = 100_000
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


def _build_fts(messages: list[str]) -> str:
    content = "\n".join(part for part in messages if part)
    return content[:MAX_FTS_CHARS]


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
            name = sanitize_text(block.get("name", ""), 200)
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
            for path in project_dir.glob("*.jsonl")
        )

    def parse(self, path: Path) -> Optional[dict]:
        session_id = path.stem
        project = path.parent.name
        project_name = self.project_names.get(project, project)
        user_prompts: list[str] = []
        fts_messages: list[str] = []
        tools: dict[str, int] = {}
        agents: dict[str, int] = {}
        summaries: list[str] = []
        exchange_count = 0
        start_time = end_time = model = cwd = None
        title = title_display = tags = None

        for entry in _read_jsonl(path):
            entry_type = entry.get("type")
            timestamp = entry.get("timestamp")
            if timestamp:
                start_time = start_time or timestamp
                end_time = timestamp
            cwd = cwd or entry.get("cwd")

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
                elif text and not _looks_like_system_prompt(text):
                    user_prompts.append(text)
                if text and not (
                    entry_type == "user" and _looks_like_system_prompt(text)
                ):
                    fts_messages.append(text)
                for tool_name in tool_names:
                    tools[tool_name] = tools.get(tool_name, 0) + 1

                if entry_type == "assistant":
                    for block in message.get("content", []) if isinstance(message.get("content"), list) else []:
                        if not isinstance(block, dict) or block.get("type") != "tool_use":
                            continue
                        if block.get("name") == "Task":
                            agent = sanitize_text(
                                (block.get("input") or {}).get("subagent_type", ""), 200
                            )
                            if agent:
                                agents[agent] = agents.get(agent, 0) + 1

        if not title_display and summaries:
            title_display = summaries[0][:80]
            title = title_display
        elif not title_display:
            title_display = _pick_title(user_prompts)
            title = title_display

        topics = [{
            "topic": summary[:120],
            "source": "compaction_summary",
            "captured_at": end_time or datetime.now().isoformat(),
            "exchange_number": None,
        } for summary in summaries]

        return {
            "source": self.source,
            "session_id": session_id,
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
            "agents": agents,
            "fts_content": _build_fts(fts_messages),
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

    def roots(self) -> Iterable[Path]:
        yield self.root
        archived = self.root.parent / self.ARCHIVED_DIR
        if not archived.is_relative_to(self.root):
            yield archived

    def discover(self) -> Iterable[Path]:
        for root in self.roots():
            if root.exists():
                yield from root.rglob("*.jsonl")

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
        # one MCP call emits both a function_call and an mcp_tool_call_end; key by call_id
        codex_calls: dict[str, str] = {}
        event_messages: list[tuple[str, str]] = []

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
                        codex_calls.setdefault(call_id, name)
                    agent = _tool_agent_name(payload)
                    if agent:
                        agents[agent] = agents.get(agent, 0) + 1
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
                and payload.get("type") == "mcp_tool_call_end"
            ):
                invocation = payload.get("invocation") or {}
                server = sanitize_text(invocation.get("server", ""), 100)
                tool = sanitize_text(invocation.get("tool", ""), 100)
                name = ".".join(part for part in (server, tool) if part)
                if name:
                    call_id = payload.get("call_id") or f"mcp-{len(codex_calls)}"
                    codex_calls[call_id] = name

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

        for name in codex_calls.values():
            tools[name] = tools.get(name, 0) + 1

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
            "agents": agents,
            "fts_content": _build_fts(fts_messages),
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
