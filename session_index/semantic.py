#!/usr/bin/env python3
"""Optional dense-vector layer over the session index.

Requires the `semantic` extra: `pip install agent-session-index[semantic]`.
Without it every entry point reports unavailable and callers fall back to FTS
alone, so the base package stays stdlib-only.
"""

from __future__ import annotations

import os
import sqlite3
import struct
from pathlib import Path

try:
    from model2vec import StaticModel
except ImportError:
    StaticModel = None

try:
    import sqlite_vec
except ImportError:
    sqlite_vec = None

try:
    from . import config
except ImportError:
    import config

HF_OFFLINE_VAR = "HF_HUB_OFFLINE"
EMBED_DIMS = 512
CHUNK_CHARS = 1_200
CHUNK_OVERLAP = 200
MIN_CHUNK_CHARS = 60
EMBED_BATCH = 512
# kNN always returns neighbours; on potion-base-32M true matches measure <=0.63 and unrelated >=0.85
MAX_COSINE_DISTANCE = 0.78

# vec0 needs the dimension as a literal, so keep one statement per width
_VEC_TABLE_SQL = {
    256: "CREATE VIRTUAL TABLE IF NOT EXISTS session_vectors USING vec0("
         "chunk_id INTEGER PRIMARY KEY, "
         "embedding FLOAT[256] distance_metric=cosine)",
    384: "CREATE VIRTUAL TABLE IF NOT EXISTS session_vectors USING vec0("
         "chunk_id INTEGER PRIMARY KEY, "
         "embedding FLOAT[384] distance_metric=cosine)",
    512: "CREATE VIRTUAL TABLE IF NOT EXISTS session_vectors USING vec0("
         "chunk_id INTEGER PRIMARY KEY, "
         "embedding FLOAT[512] distance_metric=cosine)",
    768: "CREATE VIRTUAL TABLE IF NOT EXISTS session_vectors USING vec0("
         "chunk_id INTEGER PRIMARY KEY, "
         "embedding FLOAT[768] distance_metric=cosine)",
    1024: "CREATE VIRTUAL TABLE IF NOT EXISTS session_vectors USING vec0("
          "chunk_id INTEGER PRIMARY KEY, "
         "embedding FLOAT[1024] distance_metric=cosine)",
}

_model = None
_model_error: str | None = None


class SemanticUnavailable(RuntimeError):
    pass


def _restore_env(key: str, value: str | None):
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def _load_model():
    """Load the static embedding model once per process, without phoning home.

    from_pretrained revalidates a cached model over the network, costing ~1.8s
    per invocation and telling huggingface.co that this machine is running a
    search - unacceptable for a tool that indexes private transcripts.
    """
    global _model, _model_error
    if _model is not None:
        return _model
    if _model_error is not None:
        raise SemanticUnavailable(_model_error)
    if StaticModel is None:
        _model_error = (
            "model2vec is not installed - "
            "pip install 'agent-session-index[semantic]'"
        )
        raise SemanticUnavailable(_model_error)
    name = config.get_embed_model()
    previous = os.environ.get(HF_OFFLINE_VAR)
    os.environ[HF_OFFLINE_VAR] = "1"
    try:
        _model = StaticModel.from_pretrained(name)
    except Exception:
        # Not cached yet, so allow exactly one download
        _restore_env(HF_OFFLINE_VAR, previous)
        try:
            _model = StaticModel.from_pretrained(name)
        except Exception as exc:
            _model_error = f"could not load {name}: {exc}"
            raise SemanticUnavailable(_model_error) from exc
    finally:
        _restore_env(HF_OFFLINE_VAR, previous)
    return _model


def available() -> bool:
    try:
        _load_model()
    except SemanticUnavailable:
        return False
    return True


def unavailable_reason() -> str | None:
    try:
        _load_model()
    except SemanticUnavailable as exc:
        return str(exc)
    return None


def load_vec_extension(conn: sqlite3.Connection) -> bool:
    """Attach sqlite-vec to a connection, reporting whether it took."""
    if sqlite_vec is None:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (AttributeError, sqlite3.OperationalError):
        return False
    return True


def chunk_prose(text: str) -> list[tuple[int, str]]:
    """Split prose into overlapping windows, preferring line boundaries.

    The boundary search is confined to the tail of each window: allowing it
    anywhere lets a line break near the start shrink the chunk below the
    overlap, which stalls the stride and explodes the chunk count.
    """
    chunks: list[tuple[int, str]] = []
    start = 0
    length = len(text)
    earliest_break = (CHUNK_CHARS * 3) // 4
    while start < length:
        end = min(start + CHUNK_CHARS, length)
        if end < length:
            window = text.rfind("\n", start + earliest_break, end)
            if window > start:
                end = window
        piece = text[start:end].strip()
        if len(piece) >= MIN_CHUNK_CHARS:
            chunks.append((start, piece))
        if end >= length:
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def embed(texts: list[str]) -> list[bytes]:
    """Embed texts into packed float32 blobs, L2-normalized for cosine."""
    model = _load_model()
    vectors = model.encode(texts, batch_size=EMBED_BATCH)
    blobs = []
    for vector in vectors:
        values = [float(value) for value in vector]
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        blobs.append(struct.pack(
            f"{len(values)}f", *[value / norm for value in values]
        ))
    return blobs


def model_dims() -> int:
    model = _load_model()
    dims = getattr(model, "dim", None)
    return int(dims) if dims else EMBED_DIMS


class SemanticIndex:
    """Chunk-level vectors keyed back to whole sessions."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.enabled = load_vec_extension(conn)

    def ensure_schema(self, dims: int) -> bool:
        if not self.enabled:
            return False
        statement = _VEC_TABLE_SQL.get(dims)
        if statement is None:
            raise SemanticUnavailable(
                f"embedding width {dims} is unsupported; "
                f"supported: {sorted(_VEC_TABLE_SQL)}"
            )
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS session_chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                session_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                char_offset INTEGER NOT NULL,
                char_length INTEGER NOT NULL
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_session "
            "ON session_chunks(source, session_id)"
        )
        self.conn.execute(statement)
        return True

    def clear_session(self, source: str, session_id: str):
        if not self.enabled:
            return
        rows = self.conn.execute(
            "SELECT chunk_id FROM session_chunks WHERE source=? AND session_id=?",
            (source, session_id),
        ).fetchall()
        for row in rows:
            self.conn.execute(
                "DELETE FROM session_vectors WHERE chunk_id=?", (row[0],)
            )
        self.conn.execute(
            "DELETE FROM session_chunks WHERE source=? AND session_id=?",
            (source, session_id),
        )

    def index_session(self, source: str, session_id: str, prose: str) -> int:
        """Replace one session's vectors. Returns the chunk count written."""
        if not self.enabled:
            return 0
        self.clear_session(source, session_id)
        chunks = chunk_prose(prose)
        if not chunks:
            return 0
        blobs = embed([piece for _, piece in chunks])
        for ordinal, ((offset, piece), blob) in enumerate(zip(chunks, blobs)):
            cursor = self.conn.execute(
                "INSERT INTO session_chunks "
                "(source, session_id, ordinal, char_offset, char_length) "
                "VALUES (?, ?, ?, ?, ?)",
                (source, session_id, ordinal, offset, len(piece)),
            )
            self.conn.execute(
                "INSERT INTO session_vectors (chunk_id, embedding) VALUES (?, ?)",
                (cursor.lastrowid, blob),
            )
        return len(chunks)

    def search(self, query: str, limit: int = 50) -> list[tuple[str, str, float]]:
        """Nearest chunks collapsed to one best score per session."""
        if not self.enabled:
            return []
        blob = embed([query])[0]
        # Over-fetch: many chunks collapse onto the same session
        rows = self.conn.execute("""
            SELECT c.source, c.session_id, v.distance
            FROM session_vectors v
            JOIN session_chunks c ON c.chunk_id = v.chunk_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
        """, (blob, limit * 8)).fetchall()
        best: dict[tuple[str, str], float] = {}
        for source, session_id, distance in rows:
            if distance > MAX_COSINE_DISTANCE:
                continue
            key = (source, session_id)
            if key not in best or distance < best[key]:
                best[key] = distance
        ranked = sorted(best.items(), key=lambda item: item[1])
        return [
            (source, session_id, distance)
            for (source, session_id), distance in ranked[:limit]
        ]

    def stats(self) -> dict:
        if not self.enabled:
            return {"enabled": False, "chunks": 0, "sessions": 0}
        if not self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='session_chunks'"
        ).fetchone():
            return {"enabled": True, "chunks": 0, "sessions": 0}
        row = self.conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT source || ':' || session_id) "
            "FROM session_chunks"
        ).fetchone()
        return {"enabled": True, "chunks": row[0], "sessions": row[1]}


def open_index(db_path: Path | None = None) -> tuple[sqlite3.Connection, SemanticIndex]:
    conn = sqlite3.connect(str(db_path or config.get_db_path()))
    conn.row_factory = sqlite3.Row
    return conn, SemanticIndex(conn)
