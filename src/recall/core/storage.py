"""SQLite-backed Storage for Recall.

Implements the schema in ARCHITECTURE.md §2. Single-file SQLite database with
WAL mode for concurrent reads. v2 will swap this for KuzuDB / pgvector.

v0.5 changes:
  * Scope filter is now SUBSET semantics — query scope must be a subset of
    stored scope (every key/value in query must appear in stored). Fixes
    the LongMemEval bug where stored {"qid": q, "session_id": s} would
    not match query {"qid": q}.
  * topk_cosine is vectorized over a cached `(N, embed_dim)` numpy matrix.
    Cache appends on insert, marks on deprecate. Drops p50 latency from
    O(N · python-loop) to O(N · numpy-matmul) — 20-50× at 10K nodes.
  * topk_neighbors_for_gamma takes an optional pool_size cap so write-time
    edge induction doesn't scale O(N).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

from recall.types import (
    AuditEntry,
    Drawer,
    Edge,
    EdgeType,
    Node,
)


SCHEMA_VERSION = 1


def _scope_matches_subset(query: dict | None, stored: dict | None) -> bool:
    """True iff `query` is a subset of `stored`.

    Subset semantics: every (k, v) in `query` must also appear in `stored`.
    Empty/None query matches every stored scope.

    This is what every benchmark and real-world usage actually wants:
    storing with {"qid": q, "session_id": s} and querying with {"qid": q}
    should return the node.
    """
    if not query:
        return True
    if not stored:
        return False
    return all(stored.get(k) == v for k, v in query.items())


def _vec_to_blob(v: np.ndarray | None) -> bytes | None:
    if v is None:
        return None
    return v.astype(np.float32).tobytes()


def _blob_to_vec(b: bytes | None, dim: int) -> np.ndarray | None:
    if b is None:
        return None
    return np.frombuffer(b, dtype=np.float32).reshape(dim).copy()


def _to_iso(t: datetime | None) -> str | None:
    return t.isoformat() if t else None


def _from_iso(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS drawers (
    id TEXT PRIMARY KEY,
    tenant TEXT NOT NULL,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    scope_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    transaction_time TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS drawers_tenant_scope ON drawers (tenant, scope_json);
CREATE INDEX IF NOT EXISTS drawers_source ON drawers (source);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    tenant TEXT NOT NULL,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    drawer_ids TEXT NOT NULL,
    f_embedding BLOB,
    b_embedding BLOB,
    s_embedding BLOB,        -- v0.6: raw symmetric retrieval embedding
    embed_dim INTEGER,
    role TEXT,
    quality_score REAL NOT NULL,
    quality_status TEXT NOT NULL,
    scope_json TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    parent_node_id TEXT,
    deprecated_at TEXT,
    deprecated_reason TEXT,
    created_at TEXT NOT NULL,
    transaction_time TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS nodes_tenant_scope ON nodes (tenant, scope_json);
CREATE INDEX IF NOT EXISTS nodes_quality ON nodes (quality_status);
CREATE INDEX IF NOT EXISTS nodes_deprecated ON nodes (deprecated_at);
CREATE INDEX IF NOT EXISTS nodes_text_hash ON nodes (text_hash, scope_json);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    tenant TEXT NOT NULL,
    src_node_id TEXT NOT NULL,
    dst_node_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL NOT NULL,
    gamma_score REAL NOT NULL,
    gamma_anti REAL,
    s_squared REAL NOT NULL,
    bmrs_log_ratio REAL,
    deprecated_at TEXT,
    deprecated_reason TEXT,
    created_at TEXT NOT NULL,
    last_validated_at TEXT,
    FOREIGN KEY (src_node_id) REFERENCES nodes(id),
    FOREIGN KEY (dst_node_id) REFERENCES nodes(id)
);
CREATE INDEX IF NOT EXISTS edges_src ON edges (src_node_id);
CREATE INDEX IF NOT EXISTS edges_dst ON edges (dst_node_id);
CREATE INDEX IF NOT EXISTS edges_type ON edges (edge_type);
CREATE INDEX IF NOT EXISTS edges_tenant ON edges (tenant);

CREATE TABLE IF NOT EXISTS motifs (
    id TEXT PRIMARY KEY,
    tenant TEXT NOT NULL,
    pattern_json TEXT NOT NULL,
    instances_json TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL,
    parameter_summary TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    operation TEXT NOT NULL,
    actor TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS audit_log_target ON audit_log (target_type, target_id);
CREATE INDEX IF NOT EXISTS audit_log_tenant_time ON audit_log (tenant, timestamp);
"""


class Storage(Protocol):
    """Storage protocol — see ARCHITECTURE.md §4.3."""

    tenant: str
    embed_dim: int

    def insert_drawer(self, d: Drawer) -> None: ...
    def has_drawer(self, drawer_id: str) -> bool: ...
    def get_drawer(self, drawer_id: str) -> Drawer | None: ...

    def insert_node(self, n: Node) -> None: ...
    def get_node(self, node_id: str) -> Node | None: ...
    def has_node_with_text_hash(self, text_hash: str, scope: dict) -> bool: ...
    def deprecate_node(self, node_id: str, reason: str) -> None: ...

    def insert_edge(self, e: Edge) -> None: ...
    def get_edge(self, edge_id: str) -> Edge | None: ...
    def get_edges_from(self, node_id: str, edge_type: str | None = None) -> list[Edge]: ...
    def get_edges_to(self, node_id: str, edge_type: str | None = None) -> list[Edge]: ...
    def deprecate_edge(self, edge_id: str, reason: str) -> None: ...

    def topk_cosine(self, query_emb: np.ndarray, scope: dict, k: int) -> list[Node]: ...
    def topk_neighbors_for_gamma(self, node: Node, scope: dict, k: int) -> list[Node]: ...

    def append_audit(self, entry: AuditEntry) -> None: ...
    def query_audit(
        self, target_id: str | None = None, since: datetime | None = None
    ) -> list[AuditEntry]: ...

    def all_active_nodes(self, scope: dict | None = None) -> list[Node]: ...
    def all_active_edges(self) -> list[Edge]: ...

    def close(self) -> None: ...


class _EmbeddingCache:
    """In-memory cache of the symmetric (s = (f+b)/2) embedding matrix.

    Used by topk_cosine to do vectorized cosine over all active nodes in a
    single numpy matmul instead of a per-row Python loop.

    v0.5: pre-allocated chunked storage. Matrix capacity grows by doubling
    so insert is amortized O(1), not O(N) per write. This is what unblocked
    the scale-stress benchmark above 5K nodes.
    """

    _INITIAL_CAPACITY = 256

    def __init__(self, dim: int):
        self.dim = dim
        self._initialized = False
        self._ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._matrix: np.ndarray | None = None      # shape (capacity, dim), L2-normalized
        self._scopes: list[dict] = []
        self._active: np.ndarray | None = None      # bool mask, shape (capacity,)
        self._size: int = 0                          # number of valid rows ≤ capacity

    def is_initialized(self) -> bool:
        return self._initialized

    def reset(self) -> None:
        self._initialized = False
        self._ids.clear()
        self._id_to_idx.clear()
        self._matrix = None
        self._scopes.clear()
        self._active = None
        self._size = 0

    def _alloc(self, capacity: int) -> None:
        self._matrix = np.zeros((capacity, self.dim), dtype=np.float32)
        self._active = np.zeros((capacity,), dtype=bool)

    def _grow(self, needed: int) -> None:
        if self._matrix is None:
            self._alloc(max(self._INITIAL_CAPACITY, needed))
            return
        cap = self._matrix.shape[0]
        if needed <= cap:
            return
        new_cap = cap
        while new_cap < needed:
            new_cap *= 2
        new_mat = np.zeros((new_cap, self.dim), dtype=np.float32)
        new_mat[:cap] = self._matrix
        new_act = np.zeros((new_cap,), dtype=bool)
        new_act[:cap] = self._active
        self._matrix = new_mat
        self._active = new_act

    def bulk_load(self, rows: list[tuple[str, np.ndarray, dict]]) -> None:
        """Load (id, s_vec, scope) tuples into the cache.

        s_vec must be a 1D float32 array; L2-normalized on the way in so
        the matmul yields a true cosine score.
        """
        self._initialized = True
        n = len(rows)
        # Allocate at least 2× the bulk-load size so steady-state inserts are
        # amortized O(1).
        cap = max(self._INITIAL_CAPACITY, n * 2)
        self._alloc(cap)
        if n == 0:
            self._size = 0
            return
        ids = [r[0] for r in rows]
        vecs = np.stack([r[1] for r in rows]).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        vecs = vecs / norms
        self._ids = ids
        self._id_to_idx = {nid: i for i, nid in enumerate(ids)}
        self._matrix[:n] = vecs
        self._scopes = [r[2] for r in rows]
        self._active[:n] = True
        self._size = n

    def add(self, node_id: str, s_vec: np.ndarray, scope: dict) -> None:
        if not self._initialized:
            return  # caller will refill on first query
        # In-place update if already known
        if node_id in self._id_to_idx:
            idx = self._id_to_idx[node_id]
            v = s_vec.astype(np.float32)
            nrm = float(np.linalg.norm(v))
            if nrm > 1e-12:
                v = v / nrm
            self._matrix[idx] = v
            self._scopes[idx] = scope
            self._active[idx] = True
            return
        # Append — amortized O(1) thanks to chunked allocation
        idx = self._size
        self._grow(idx + 1)
        v = s_vec.astype(np.float32)
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-12:
            v = v / nrm
        self._matrix[idx] = v
        self._active[idx] = True
        self._ids.append(node_id)
        self._id_to_idx[node_id] = idx
        self._scopes.append(scope)
        self._size += 1

    def deprecate(self, node_id: str) -> None:
        if not self._initialized:
            return
        idx = self._id_to_idx.get(node_id)
        if idx is not None and self._active is not None:
            self._active[idx] = False

    def topk(
        self,
        query_emb: np.ndarray,
        scope_filter: dict | None,
        k: int,
    ) -> list[tuple[str, float]]:
        """Return [(node_id, score)] sorted by descending cosine score.

        Applies (active mask) ∧ (scope subset filter), then a single matmul
        over the live prefix of the matrix.
        """
        if not self._initialized or self._matrix is None or self._size == 0:
            return []
        q = np.asarray(query_emb, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn < 1e-12:
            return []
        q = q / qn
        # Only use the populated prefix
        live = self._matrix[: self._size]
        scores = live @ q  # (size,)
        mask = self._active[: self._size].copy()
        if scope_filter:
            for i in range(self._size):
                if mask[i] and not _scope_matches_subset(scope_filter, self._scopes[i]):
                    mask[i] = False
        scores = np.where(mask, scores, -np.inf)
        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            partial = np.argpartition(-scores, k)[:k]
            order = partial[np.argsort(-scores[partial])]
        out: list[tuple[str, float]] = []
        for i in order:
            s = float(scores[i])
            if s == float("-inf"):
                break
            out.append((self._ids[i], s))
        return out


class SQLiteStorage:
    """SQLite-backed Storage implementation."""

    def __init__(self, tenant: str, db_path: str | Path = ":memory:", embed_dim: int = 256):
        self.tenant = tenant
        self.embed_dim = embed_dim
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if self._db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        # In-memory embedding index for vectorized cosine retrieval.
        # Lazily initialized on first topk_cosine / topk_neighbors_for_gamma call.
        self._index = _EmbeddingCache(dim=embed_dim)
        # v0.6: in-memory adjacency cache for the auto-router. Maps
        # node_id → set of neighbor ids over active edges. Lazily built
        # on first read; invalidated on edge insert/deprecate. Drops
        # router cost from O(E load + python construct) to O(1) read.
        self._adj_cache: dict[str, set[str]] | None = None
        self._n_active_edges_cached: int = 0

    def _init_schema(self) -> None:
        # executescript handles multi-statement SQL correctly
        self._conn.executescript(SCHEMA_SQL)
        # v0.6 migration: add s_embedding column if upgrading from v0.5
        # PRAGMA table_info returns rows for each column.
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(nodes)").fetchall()
        }
        if "s_embedding" not in cols:
            self._conn.execute("ALTER TABLE nodes ADD COLUMN s_embedding BLOB")
        self._conn.execute(
            "INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    # --- Drawers ---

    def insert_drawer(self, d: Drawer) -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO drawers
                (id, tenant, text, source, scope_json, created_at, valid_from, valid_to,
                 transaction_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                d.id,
                d.tenant,
                d.text,
                d.source,
                d.scope_json(),
                _to_iso(d.created_at),
                _to_iso(d.valid_from),
                _to_iso(d.valid_to),
                _to_iso(d.transaction_time),
            ),
        )
        self._conn.commit()

    def has_drawer(self, drawer_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM drawers WHERE id = ?", (drawer_id,)
        ).fetchone()
        return row is not None

    def get_drawer(self, drawer_id: str) -> Drawer | None:
        row = self._conn.execute(
            "SELECT * FROM drawers WHERE id = ?", (drawer_id,)
        ).fetchone()
        if not row:
            return None
        return Drawer(
            id=row["id"],
            tenant=row["tenant"],
            text=row["text"],
            source=row["source"],
            scope=json.loads(row["scope_json"]),
            created_at=_from_iso(row["created_at"]),
            valid_from=_from_iso(row["valid_from"]),
            valid_to=_from_iso(row["valid_to"]),
            transaction_time=_from_iso(row["transaction_time"]),
        )

    # --- Nodes ---

    def insert_node(self, n: Node) -> None:
        import hashlib

        text_hash = hashlib.sha256(n.text.encode("utf-8")).hexdigest()
        self._conn.execute(
            """INSERT OR REPLACE INTO nodes
                (id, tenant, text, text_hash, drawer_ids, f_embedding, b_embedding,
                 s_embedding, embed_dim, role, quality_score, quality_status,
                 scope_json, version, parent_node_id, deprecated_at,
                 deprecated_reason, created_at, transaction_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                n.id,
                n.tenant,
                n.text,
                text_hash,
                json.dumps(n.drawer_ids),
                _vec_to_blob(n.f_embedding),
                _vec_to_blob(n.b_embedding),
                _vec_to_blob(n.s_embedding),  # v0.6
                self.embed_dim,
                n.role,
                n.quality_score,
                n.quality_status,
                n.scope_json(),
                n.version,
                n.parent_node_id,
                _to_iso(n.deprecated_at),
                n.deprecated_reason,
                _to_iso(n.created_at),
                _to_iso(n.transaction_time),
            ),
        )
        self._conn.commit()
        # Update in-memory index for vectorized topk. v0.6: prefer
        # s_embedding (raw symmetric); fall back to (f+b)/2 if absent.
        if n.tenant == self.tenant and n.quality_status == "promoted" and n.deprecated_at is None:
            if self._index.is_initialized():
                if n.s_embedding is not None:
                    self._index.add(n.id, n.s_embedding, n.scope or {})
                elif n.f_embedding is not None and n.b_embedding is not None:
                    self._index.add(n.id, (n.f_embedding + n.b_embedding) / 2, n.scope or {})

    def _row_to_node(self, row: sqlite3.Row) -> Node:
        dim = row["embed_dim"] or self.embed_dim
        # v0.6: s_embedding is optional (column added by migration; pre-v0.6
        # rows have NULL). sqlite3.Row supports column-by-name access.
        s_blob = None
        try:
            s_blob = row["s_embedding"]
        except (IndexError, KeyError):
            pass
        return Node(
            id=row["id"],
            tenant=row["tenant"],
            text=row["text"],
            drawer_ids=json.loads(row["drawer_ids"]),
            f_embedding=_blob_to_vec(row["f_embedding"], dim),
            b_embedding=_blob_to_vec(row["b_embedding"], dim),
            s_embedding=_blob_to_vec(s_blob, dim),
            role=row["role"],
            quality_score=row["quality_score"],
            quality_status=row["quality_status"],
            scope=json.loads(row["scope_json"]),
            version=row["version"],
            parent_node_id=row["parent_node_id"],
            deprecated_at=_from_iso(row["deprecated_at"]),
            deprecated_reason=row["deprecated_reason"],
            created_at=_from_iso(row["created_at"]) or datetime.now(),
            transaction_time=_from_iso(row["transaction_time"]) or datetime.now(),
        )

    def get_node(self, node_id: str) -> Node | None:
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def has_node_with_text_hash(self, text_hash: str, scope: dict) -> bool:
        scope_str = json.dumps(scope, sort_keys=True, default=str)
        row = self._conn.execute(
            "SELECT 1 FROM nodes WHERE text_hash = ? AND scope_json = ? "
            "AND deprecated_at IS NULL",
            (text_hash, scope_str),
        ).fetchone()
        return row is not None

    def deprecate_node(self, node_id: str, reason: str) -> None:
        self._conn.execute(
            "UPDATE nodes SET deprecated_at = ?, deprecated_reason = ? WHERE id = ?",
            (datetime.now().isoformat(), reason, node_id),
        )
        self._conn.commit()
        self._index.deprecate(node_id)

    # --- Edges ---

    def insert_edge(self, e: Edge) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO edges
                (id, tenant, src_node_id, dst_node_id, edge_type, weight, gamma_score,
                 gamma_anti, s_squared, bmrs_log_ratio, deprecated_at, deprecated_reason,
                 created_at, last_validated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                e.id,
                e.tenant,
                e.src_node_id,
                e.dst_node_id,
                e.edge_type.value if isinstance(e.edge_type, EdgeType) else e.edge_type,
                e.weight,
                e.gamma_score,
                e.gamma_anti,
                e.s_squared,
                e.bmrs_log_ratio,
                _to_iso(e.deprecated_at),
                e.deprecated_reason,
                _to_iso(e.created_at),
                _to_iso(e.last_validated_at),
            ),
        )
        self._conn.commit()
        # v0.6: keep adjacency cache in sync if it's been built
        if self._adj_cache is not None and e.deprecated_at is None:
            self._adj_cache.setdefault(e.src_node_id, set()).add(e.dst_node_id)
            self._adj_cache.setdefault(e.dst_node_id, set()).add(e.src_node_id)
            self._n_active_edges_cached += 1

    def _row_to_edge(self, row: sqlite3.Row) -> Edge:
        try:
            etype = EdgeType(row["edge_type"])
        except ValueError:
            etype = EdgeType.PENDING
        return Edge(
            id=row["id"],
            tenant=row["tenant"],
            src_node_id=row["src_node_id"],
            dst_node_id=row["dst_node_id"],
            edge_type=etype,
            weight=row["weight"],
            gamma_score=row["gamma_score"],
            gamma_anti=row["gamma_anti"],
            s_squared=row["s_squared"],
            bmrs_log_ratio=row["bmrs_log_ratio"],
            deprecated_at=_from_iso(row["deprecated_at"]),
            deprecated_reason=row["deprecated_reason"],
            created_at=_from_iso(row["created_at"]) or datetime.now(),
            last_validated_at=_from_iso(row["last_validated_at"]),
        )

    def get_edge(self, edge_id: str) -> Edge | None:
        row = self._conn.execute("SELECT * FROM edges WHERE id = ?", (edge_id,)).fetchone()
        return self._row_to_edge(row) if row else None

    def get_edges_from(self, node_id: str, edge_type: str | None = None) -> list[Edge]:
        q = "SELECT * FROM edges WHERE src_node_id = ? AND deprecated_at IS NULL"
        params: list[Any] = [node_id]
        if edge_type:
            q += " AND edge_type = ?"
            params.append(edge_type)
        rows = self._conn.execute(q, params).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_edges_to(self, node_id: str, edge_type: str | None = None) -> list[Edge]:
        q = "SELECT * FROM edges WHERE dst_node_id = ? AND deprecated_at IS NULL"
        params: list[Any] = [node_id]
        if edge_type:
            q += " AND edge_type = ?"
            params.append(edge_type)
        rows = self._conn.execute(q, params).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def deprecate_edge(self, edge_id: str, reason: str) -> None:
        # Look up the (src, dst) pair before deprecation so we can update
        # the adjacency cache without a second SELECT.
        prior = None
        if self._adj_cache is not None:
            row = self._conn.execute(
                "SELECT src_node_id, dst_node_id FROM edges WHERE id = ?", (edge_id,)
            ).fetchone()
            if row is not None:
                prior = (row["src_node_id"], row["dst_node_id"])
        self._conn.execute(
            "UPDATE edges SET deprecated_at = ?, deprecated_reason = ? WHERE id = ?",
            (datetime.now().isoformat(), reason, edge_id),
        )
        self._conn.commit()
        if prior is not None and self._adj_cache is not None:
            # Invalidate the cache rather than precisely mutate — the
            # graph may have multiple edges between the same pair, and
            # we don't want to remove a still-active one. Cheap rebuild.
            self._adj_cache = None
            self._n_active_edges_cached = 0

    # --- ANN-style retrieval ---

    def _ensure_index_loaded(self) -> None:
        """Lazily load all promoted-active nodes into the in-memory matrix.

        v0.6: prefers `s_embedding` (raw symmetric, no prompt prefix) when
        present; falls back to `(f + b) / 2` for legacy rows.
        """
        if self._index.is_initialized():
            return
        rows = self._conn.execute(
            "SELECT id, f_embedding, b_embedding, s_embedding, embed_dim, scope_json "
            "FROM nodes WHERE tenant = ? AND deprecated_at IS NULL "
            "AND quality_status = 'promoted'",
            (self.tenant,),
        ).fetchall()
        triples: list[tuple[str, np.ndarray, dict]] = []
        for r in rows:
            dim = r["embed_dim"] or self.embed_dim
            s_blob = r["s_embedding"]
            s_vec = _blob_to_vec(s_blob, dim)
            if s_vec is None:
                f = _blob_to_vec(r["f_embedding"], dim)
                b = _blob_to_vec(r["b_embedding"], dim)
                if f is None or b is None:
                    continue
                s_vec = (f + b) / 2.0
            scope = json.loads(r["scope_json"]) if r["scope_json"] else {}
            triples.append((r["id"], s_vec, scope))
        # If embed_dim disagrees with the cache, prefer the stored dim
        if triples and triples[0][1].shape[0] != self._index.dim:
            self._index.dim = triples[0][1].shape[0]
        self._index.bulk_load(triples)

    def topk_cosine(self, query_emb: np.ndarray, scope: dict, k: int) -> list[Node]:
        """Vectorized cosine over the symmetric component s = (f+b)/2.

        v0.5: scope filter is SUBSET semantics (query subset-of stored).
        Single numpy matmul over the cached embedding matrix.
        """
        self._ensure_index_loaded()
        hits = self._index.topk(query_emb, scope or None, k)
        if not hits:
            return []
        # Fetch full node rows for the matched ids
        ids = [h[0] for h in hits]
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT * FROM nodes WHERE id IN ({placeholders})", ids
        ).fetchall()
        by_id = {r["id"]: self._row_to_node(r) for r in rows}
        # Preserve scored order
        return [by_id[i] for i in ids if i in by_id]

    def topk_neighbors_for_gamma(
        self,
        node: Node,
        scope: dict,
        k: int,
        pool_size: int | None = None,  # noqa: ARG002 — kept for API stability
    ) -> list[Node]:
        """Return k closest existing nodes for Γ-edge induction.

        v0.5: same vectorized path as topk_cosine — write-time edge induction
        is now a single matmul rather than a Python loop.
        """
        if node.f_embedding is None or node.b_embedding is None:
            return []
        s_query = node.s()
        cand = self.topk_cosine(s_query, scope, k=k + 1)
        return [n for n in cand if n.id != node.id][:k]

    # --- Audit ---

    def append_audit(self, entry: AuditEntry) -> None:
        self._conn.execute(
            """INSERT INTO audit_log
                (tenant, timestamp, operation, actor, target_type, target_id,
                 payload_json, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.tenant,
                _to_iso(entry.timestamp),
                entry.operation,
                entry.actor,
                entry.target_type,
                entry.target_id,
                json.dumps(entry.payload, default=str),
                entry.reason,
            ),
        )
        self._conn.commit()

    def query_audit(
        self, target_id: str | None = None, since: datetime | None = None
    ) -> list[AuditEntry]:
        q = "SELECT * FROM audit_log WHERE tenant = ?"
        params: list[Any] = [self.tenant]
        if target_id:
            q += " AND target_id = ?"
            params.append(target_id)
        if since:
            q += " AND timestamp >= ?"
            params.append(_to_iso(since))
        q += " ORDER BY seq ASC"
        rows = self._conn.execute(q, params).fetchall()
        out: list[AuditEntry] = []
        for r in rows:
            out.append(
                AuditEntry(
                    seq=r["seq"],
                    tenant=r["tenant"],
                    timestamp=_from_iso(r["timestamp"]),
                    operation=r["operation"],
                    actor=r["actor"],
                    target_type=r["target_type"],
                    target_id=r["target_id"],
                    payload=json.loads(r["payload_json"]),
                    reason=r["reason"],
                )
            )
        return out

    # --- Bulk ---

    def all_active_nodes(
        self, scope: dict | None = None, limit: int | None = None,
    ) -> list[Node]:
        """v0.5: scope filter is subset semantics (query subset-of stored).

        `limit` (when provided) caps the number of rows returned by the SQL
        query — useful for the conversation-buffer query that only needs the
        last 20 active nodes. Without scope filter the limit is exact; with a
        scope filter we may need to over-fetch to satisfy the limit after
        filtering, so we add a 4× cushion.
        """
        if limit is None:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE tenant = ? AND deprecated_at IS NULL "
                "AND quality_status = 'promoted'",
                (self.tenant,),
            ).fetchall()
        else:
            sql_limit = limit if not scope else limit * 4
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE tenant = ? AND deprecated_at IS NULL "
                "AND quality_status = 'promoted' "
                "ORDER BY rowid DESC LIMIT ?",
                (self.tenant, sql_limit),
            ).fetchall()
            rows = list(reversed(rows))  # restore creation order
        nodes = [self._row_to_node(r) for r in rows]
        if scope:
            nodes = [n for n in nodes if _scope_matches_subset(scope, n.scope or {})]
        if limit is not None:
            nodes = nodes[-limit:]
        return nodes

    def all_active_edges(self) -> list[Edge]:
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE tenant = ? AND deprecated_at IS NULL", (self.tenant,)
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def adjacency(self) -> dict[str, set[str]]:
        """v0.6: return cached undirected adjacency over active edges.

        Built on first call; kept in sync on insert_edge; invalidated on
        deprecate_edge (lazy rebuild on next call). Used by the auto-router
        and any other graph-traversal code that needs O(1) neighbor lookup.
        """
        if self._adj_cache is not None:
            return self._adj_cache
        rows = self._conn.execute(
            "SELECT src_node_id, dst_node_id FROM edges "
            "WHERE tenant = ? AND deprecated_at IS NULL",
            (self.tenant,),
        ).fetchall()
        adj: dict[str, set[str]] = {}
        for r in rows:
            adj.setdefault(r["src_node_id"], set()).add(r["dst_node_id"])
            adj.setdefault(r["dst_node_id"], set()).add(r["src_node_id"])
        self._adj_cache = adj
        self._n_active_edges_cached = len(rows)
        return adj

    def n_active_edges(self) -> int:
        """Cheap count of active edges, served from cache when possible."""
        if self._adj_cache is not None:
            return self._n_active_edges_cached
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM edges "
            "WHERE tenant = ? AND deprecated_at IS NULL",
            (self.tenant,),
        ).fetchone()
        return int(row["n"]) if row else 0

    def close(self) -> None:
        self._conn.close()
