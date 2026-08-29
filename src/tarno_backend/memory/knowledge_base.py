"""CPU-only knowledge base for the TARNO coding assistant.

Stores documents and chunks in SQLite, encodes them with the shared local
embedding provider (tarno.memory.embeddings), and retrieves by cosine
similarity. Falls back to keyword search if embeddings are unavailable.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from tarno_backend.memory.chunking import chunk_text
from tarno_backend.memory.embeddings import (
    blob_to_vector as _blob_to_vector,
    cosine_similarity as _cosine_similarity,
    get_default_provider as _get_default_provider,
    vector_to_blob as _vector_to_blob,
)

log = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    last_modified TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS kb_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES kb_documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_kb_chunks_doc_id ON kb_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_kb_documents_path ON kb_documents(path);
"""


class KnowledgeBase:
    """Local knowledge base backed by SQLite and CPU embeddings."""

    _TEXT_EXTENSIONS = frozenset({
        ".txt", ".md", ".py", ".cs", ".js", ".ts", ".html", ".css",
        ".json", ".yaml", ".yml", ".xml", ".ini", ".cfg", ".toml",
        ".sh", ".ps1", ".bat", ".cpp", ".c", ".h", ".hpp", ".java",
        ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala",
    })

    def __init__(self, db_path: Path | str = "~/.tarno/memory/kb.db") -> None:
        self._path = str(Path(db_path).expanduser()) if db_path != ":memory:" else ":memory:"
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._local = threading.local()
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            self._local.conn = conn
        return conn

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn().executescript(_SCHEMA)

    def ingest_text(self, source: str, text: str, *, chunk_size: int = 1000, overlap: int = 100) -> int:
        """Store a plain-text source (file path or identifier) and its chunks.

        Returns the number of chunks stored.
        """
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return 0

        provider = _get_default_provider()
        vectors = provider.embed(chunks)

        with self._lock:
            conn = self._conn()
            cursor = conn.execute(
                """INSERT INTO kb_documents (path, source, last_modified)
                   VALUES (?, ?, ?)
                   ON CONFLICT(path) DO UPDATE SET
                       source=excluded.source,
                       last_modified=excluded.last_modified
                   RETURNING id""",
                (source, source, datetime.now().isoformat()),
            )
            doc_id = cursor.fetchone()["id"]
            conn.execute("DELETE FROM kb_chunks WHERE doc_id = ?", (doc_id,))
            for idx, chunk in enumerate(chunks):
                embedding_blob = b""
                if vectors is not None:
                    embedding_blob = _vector_to_blob(vectors[idx])
                conn.execute(
                    "INSERT INTO kb_chunks (doc_id, text, embedding, chunk_index) VALUES (?, ?, ?, ?)",
                    (doc_id, chunk, embedding_blob, idx),
                )
            conn.commit()
        log.info("KB: %d Chunks für %s gespeichert", len(chunks), source)
        return len(chunks)

    def ingest_file(self, path: Path | str, *, chunk_size: int = 1000, overlap: int = 100) -> int:
        """Ingest a single text file into the knowledge base."""
        p = Path(path)
        if p.suffix.lower() not in self._TEXT_EXTENSIONS:
            log.debug("KB: %s hat ungültige Erweiterung, wird übersprungen", p)
            return 0
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            log.exception("KB: %s konnte nicht gelesen werden", p)
            return 0
        return self.ingest_text(str(p.resolve()), text, chunk_size=chunk_size, overlap=overlap)

    def delete_document(self, source: str) -> None:
        """Remove a document and its chunks from the knowledge base."""
        with self._lock:
            conn = self._conn()
            conn.execute(
                "DELETE FROM kb_documents WHERE path = ? OR source = ?",
                (source, source),
            )
            conn.commit()

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Return the top-k (chunk_text, score) matches for the query."""
        query_lower = query.lower()
        with self._lock:
            rows = self._conn().execute(
                "SELECT text, embedding FROM kb_chunks"
            ).fetchall()

        if not rows:
            return []

        results: list[tuple[str, float]] = []
        provider = _get_default_provider()
        query_vector = provider.embed_one(query)

        if query_vector is not None:
            query_vector = query_vector.astype(np.float32)
            for text, blob in rows:
                if not blob:
                    continue
                chunk_vector = _blob_to_vector(blob)
                score = _cosine_similarity(query_vector, chunk_vector)
                results.append((text, float(score)))
        else:
            # Keyword fallback: count matching query tokens.
            query_tokens = set(query_lower.split())
            for text, _blob in rows:
                text_tokens = set(text.lower().split())
                matches = len(query_tokens & text_tokens)
                if matches:
                    results.append((text, matches / len(query_tokens)))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

    def list_documents(self) -> list[str]:
        """Return the stored document paths/sources."""
        with self._lock:
            rows = self._conn().execute(
                "SELECT path FROM kb_documents ORDER BY path"
            ).fetchall()
        return [row["path"] for row in rows]
