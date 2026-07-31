"""Configuration resolution for Agent Session Index.

Priority order:
1. Function arguments (passed directly)
2. Environment variables
3. Config file (~/.session-index/config.json)
4. Sensible defaults
"""

import json
import os
from pathlib import Path
from typing import Optional

DEFAULTS = {
    "projects_dir": str(Path.home() / ".claude" / "projects"),
    "db_path": str(Path.home() / ".session-index" / "sessions.db"),
    "topics_dir": str(Path.home() / ".claude" / "session-topics"),
    "clients": [],
    "project_names": {},
    "embed_model": "minishlab/potion-base-32M",
    "recency_half_life_days": 90,
    # Swept on 225 historical-lookup queries: 0.1 beats both 0 and 0.5, and 0.5 costs 5.7pt of recall@1
    "recency_weight": 0.1,
    "sources": {
        "claude": {
            "enabled": True,
            "root": str(Path.home() / ".claude" / "projects"),
        },
        "codex": {
            "enabled": True,
            "root": str(Path.home() / ".codex" / "sessions"),
        },
    },
}

CONFIG_FILE = Path.home() / ".session-index" / "config.json"

_cached_config: Optional[dict] = None


def _load_config_file() -> dict:
    """Load config from ~/.session-index/config.json if it exists."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def get_config() -> dict:
    """Resolve config from all sources. Result is cached after first call."""
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    # Start with defaults (copy nested source dictionaries too).
    config = {
        **DEFAULTS,
        "sources": {
            name: dict(settings)
            for name, settings in DEFAULTS["sources"].items()
        },
    }

    # Layer on config file
    file_config = _load_config_file()
    for key, value in file_config.items():
        if key == "sources" and isinstance(value, dict):
            for source, settings in value.items():
                if source in config["sources"] and isinstance(settings, dict):
                    config["sources"][source].update(
                        {k: v for k, v in settings.items() if v is not None}
                    )
        elif key in config and value is not None:
            config[key] = value

    # Existing configs used projects_dir for Claude. Keep it authoritative
    # unless the new nested Claude root was explicitly configured.
    if "projects_dir" in file_config and not (
        isinstance(file_config.get("sources"), dict)
        and "claude" in file_config["sources"]
        and "root" in file_config["sources"]["claude"]
    ):
        config["sources"]["claude"]["root"] = file_config["projects_dir"]

    # Layer on environment variables
    env_map = {
        "SESSION_INDEX_PROJECTS": "projects_dir",
        "SESSION_INDEX_DB": "db_path",
        "SESSION_INDEX_TOPICS": "topics_dir",
    }
    for env_key, config_key in env_map.items():
        val = os.environ.get(env_key)
        if val:
            config[config_key] = val

    if os.environ.get("SESSION_INDEX_PROJECTS"):
        config["sources"]["claude"]["root"] = os.environ["SESSION_INDEX_PROJECTS"]

    source_root_env = {
        "SESSION_INDEX_CLAUDE_ROOT": "claude",
        "SESSION_INDEX_CODEX_ROOT": "codex",
    }
    for env_key, source in source_root_env.items():
        if os.environ.get(env_key):
            config["sources"][source]["root"] = os.environ[env_key]

    enabled_sources = os.environ.get("SESSION_INDEX_SOURCES")
    if enabled_sources is not None:
        enabled = {
            item.strip().lower()
            for item in enabled_sources.split(",")
            if item.strip()
        }
        for source in config["sources"]:
            config["sources"][source]["enabled"] = source in enabled

    for source in ("claude", "codex"):
        value = os.environ.get(f"SESSION_INDEX_{source.upper()}_ENABLED")
        if value is not None:
            config["sources"][source]["enabled"] = (
                value.strip().lower() not in {"0", "false", "no", "off"}
            )

    _cached_config = config
    return config


def get_projects_dir(override: str = None) -> Path:
    """Get the Claude projects directory (backwards-compatible helper)."""
    if override:
        return Path(override).expanduser()
    return Path(get_config()["sources"]["claude"]["root"]).expanduser()


def get_codex_sessions_dir(override: str = None) -> Path:
    """Get the Codex rollout sessions directory."""
    if override:
        return Path(override).expanduser()
    return Path(get_config()["sources"]["codex"]["root"]).expanduser()


def get_source_configs(overrides: dict | None = None) -> dict:
    """Return source enablement and roots, with optional runtime overrides."""
    resolved = {
        source: {
            "enabled": bool(settings.get("enabled", True)),
            "root": str(Path(settings["root"]).expanduser()),
        }
        for source, settings in get_config()["sources"].items()
    }
    for source, settings in (overrides or {}).items():
        if source not in resolved:
            continue
        if isinstance(settings, (str, Path)):
            resolved[source]["root"] = str(Path(settings).expanduser())
        elif isinstance(settings, dict):
            if "enabled" in settings:
                resolved[source]["enabled"] = bool(settings["enabled"])
            if settings.get("root"):
                resolved[source]["root"] = str(
                    Path(settings["root"]).expanduser()
                )
    return resolved


def get_db_path(override: str = None) -> Path:
    """Get database path, creating parent directory if needed."""
    if override:
        p = Path(override).expanduser()
    else:
        p = Path(get_config()["db_path"]).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_topics_dir(override: str = None) -> Path:
    """Get topics directory path."""
    if override:
        return Path(override).expanduser()
    return Path(get_config()["topics_dir"]).expanduser()


def get_embed_model() -> str:
    return get_config().get("embed_model") or DEFAULTS["embed_model"]


def get_recency_half_life() -> float:
    value = get_config().get("recency_half_life_days")
    return float(value) if value else float(DEFAULTS["recency_half_life_days"])


def get_recency_weight() -> float:
    value = get_config().get("recency_weight")
    return float(DEFAULTS["recency_weight"]) if value is None else float(value)


def get_clients() -> list[str]:
    """Get list of known client names (optional — used for auto-detection)."""
    return get_config().get("clients", [])


def get_project_names() -> dict[str, str]:
    """Get project directory → friendly name mapping.

    If not configured, auto-generates from directory names:
    '-Users-lee-CC-LFI' → 'LFI'
    '-Users-foo-projects-myapp' → 'myapp'
    """
    configured = get_config().get("project_names", {})
    if configured:
        return configured

    # Auto-generate from directory names
    projects_dir = get_projects_dir()
    mapping = {}
    if projects_dir.exists():
        for d in projects_dir.iterdir():
            if d.is_dir():
                name = d.name
                # Take the last meaningful segment
                parts = [p for p in name.split("-") if p]
                if parts:
                    # Use last 1-2 segments as friendly name
                    friendly = " ".join(parts[-2:]) if len(parts) > 1 else parts[-1]
                    mapping[name] = friendly

    return mapping


def init_config():
    """Create a default config file if one doesn't exist."""
    if CONFIG_FILE.exists():
        return False

    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(DEFAULTS, indent=2) + "\n")
    return True


def ensure_indexed(db_path: Path = None) -> bool:
    """Migrate the database and index each enabled source once."""
    if db_path is None:
        db_path = get_db_path()

    try:
        from session_index.indexer import SessionIndexer
    except ImportError:
        try:
            from .indexer import SessionIndexer
        except ImportError:
            from indexer import SessionIndexer

    indexer = SessionIndexer(db_path=db_path)
    indexer.connect()
    try:
        initialized = {
            row[0] for row in indexer.conn.execute(
                "SELECT source FROM index_state"
            ).fetchall()
        }
        missing = [
            source for source in indexer.adapters
            if source not in initialized
        ]
        if not missing:
            return False
        print(
            "\n  Indexing newly enabled local session sources: "
            + ", ".join(missing)
            + "\n"
        )
        if set(missing) == set(indexer.adapters):
            indexer.backfill_all()
        else:
            for source in missing:
                indexer.backfill_all(source=source)
        return True
    finally:
        indexer.close()
