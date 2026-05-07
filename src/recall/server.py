"""FastAPI HTTP server — drop-in for Letta / LangGraph / Mastra.

Exposes the public Memory API as REST endpoints:

  POST /v1/memory/observe          — write a turn
  POST /v1/memory/recall           — read a path
  POST /v1/memory/bounded_generate — bounded generation
  POST /v1/memory/forget           — surgical forget
  GET  /v1/memory/trace/:gen_id    — provenance
  GET  /v1/memory/audit            — audit log
  GET  /v1/memory/stats            — health snapshot

Run:
    uvicorn recall.server:app --port 8765
"""
from __future__ import annotations

import os
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Server requires `pip install fastapi pydantic uvicorn`.")

from recall import Memory


# Single global memory per process, tenant-keyed
_memories: dict[str, Memory] = {}


def get_memory(tenant: str) -> Memory:
    if tenant not in _memories:
        db_dir = os.environ.get("RECALL_DB_DIR", "./recall_data")
        os.makedirs(db_dir, exist_ok=True)
        _memories[tenant] = Memory(
            tenant=tenant,
            storage=f"sqlite://{db_dir}/{tenant}.db",
        )
    return _memories[tenant]


class ObserveRequest(BaseModel):
    tenant: str
    user_msg: str
    agent_msg: str = ""
    scope: dict[str, Any] = {}
    source: str = "conversation"


class RecallRequest(BaseModel):
    tenant: str
    query: str
    scope: dict[str, Any] = {}
    mode: str = "path"
    k: int = 10
    depth: int = 4


class BoundedGenerateRequest(BaseModel):
    tenant: str
    query: str
    scope: dict[str, Any] = {}
    bound: str = "soft"
    k: int = 10
    depth: int = 4


class ForgetRequest(BaseModel):
    tenant: str
    node_id: str
    reason: str = "user_request"
    actor: str = "user"


app = FastAPI(title="Recall", version="0.1.0",
              description="Memory layer for AI agents")


@app.get("/")
def root():
    return {
        "name": "Recall",
        "version": "0.1.0",
        "tenants": list(_memories.keys()),
    }


@app.post("/v1/memory/observe")
def observe(req: ObserveRequest):
    mem = get_memory(req.tenant)
    result = mem.observe(req.user_msg, req.agent_msg, scope=req.scope, source=req.source)
    return {
        "drawer_id": result.drawer_id,
        "nodes_written": result.nodes_written,
        "nodes_rejected": [{"id": nid, "reason": r} for nid, r in result.nodes_rejected],
        "nodes_skipped_duplicate": result.nodes_skipped_duplicate,
        "edges_written": result.edges_written,
        "drawer_was_duplicate": result.drawer_was_duplicate,
        "skipped_recall_loop": result.skipped_recall_loop,
    }


@app.post("/v1/memory/recall")
def recall(req: RecallRequest):
    mem = get_memory(req.tenant)
    result = mem.recall(req.query, scope=req.scope, mode=req.mode, k=req.k, depth=req.depth)
    return {
        "query": req.query,
        "mode": result.mode,
        "nodes": [
            {"id": n.id, "text": n.text, "role": n.role, "quality": n.quality_score}
            for n in result.subgraph_nodes
        ],
        "edges": [
            {
                "src": e.src_node_id, "dst": e.dst_node_id,
                "type": e.edge_type.value if hasattr(e.edge_type, "value") else str(e.edge_type),
                "weight": e.weight,
                "gamma": e.gamma_score,
            }
            for e in result.subgraph_edges
        ],
    }


@app.post("/v1/memory/bounded_generate")
def bounded_generate(req: BoundedGenerateRequest):
    mem = get_memory(req.tenant)
    try:
        result = mem.bounded_generate(
            req.query, scope=req.scope, bound=req.bound, k=req.k, depth=req.depth
        )
    except Exception as e:
        raise HTTPException(status_code=409, detail=f"hallucination_blocked: {e}")
    return {
        "text": result.text,
        "flagged_claims": result.flagged_claims,
        "bound_value": result.bound_value,
        "blocked": result.blocked,
        "retrieved_nodes": [n.id for n in result.retrieved.subgraph_nodes],
        "retrieved_edges": [e.id for e in result.retrieved.subgraph_edges],
    }


@app.post("/v1/memory/forget")
def forget(req: ForgetRequest):
    mem = get_memory(req.tenant)
    result = mem.forget(req.node_id, reason=req.reason, actor=req.actor)
    if result.error:
        raise HTTPException(status_code=404, detail=result.error)
    return {
        "deprecated_node_id": result.deprecated_node_id,
        "deprecated_edge_ids": result.deprecated_edge_ids,
    }


@app.get("/v1/memory/audit")
def audit(tenant: str, target: str | None = None, limit: int = 100):
    mem = get_memory(tenant)
    if target:
        entries = mem.audit.for_target(target)
    else:
        entries = mem.audit.since(mem._epoch_start())
    return {
        "entries": [
            {
                "seq": e.seq,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                "operation": e.operation,
                "actor": e.actor,
                "target_type": e.target_type,
                "target_id": e.target_id,
                "reason": e.reason,
                "payload": e.payload,
            }
            for e in entries[-limit:]
        ]
    }


@app.get("/v1/memory/stats")
def stats(tenant: str):
    mem = get_memory(tenant)
    return mem.stats()


@app.post("/v1/memory/consolidate")
def consolidate(tenant: str, budget: int = 10):
    mem = get_memory(tenant)
    stats = mem.consolidate(budget=budget)
    return {
        "regions_processed": stats.regions_processed,
        "edges_pruned": stats.edges_pruned,
        "edges_refined": stats.edges_refined,
        "motifs_found": stats.motifs_found,
    }
