"""Recall — memory layer for AI agents.

Public API:

    from recall import Memory, Embedder, LLMClient

    mem = Memory(tenant="my_app")
    mem.observe(user_msg, agent_msg)
    answer = mem.bounded_generate(query)
    trace = mem.trace(answer)
    mem.forget(node_id, reason="outdated")
"""

from recall.api import HallucinationBlocked, Memory
from recall.consolidate.scheduler import ConsolidationStats, Consolidator
from recall.embeddings import (
    BGEEmbedder,
    Embedder,
    HashEmbedder,
    TfidfEmbedder,
    auto_embedder,
)
from recall.geometry.identifiability import (
    CanonicalEmbedder,
    ParaphraseEnsembleEmbedder,
    WhiteningProjector,
)
from recall.llm import LLMClient, MockLLMClient, OpenAIClient
from recall.llm_quality import LLMQualityClassifier
from recall.llm_router import RouterClient
from recall.types import (
    AuditEntry,
    Drawer,
    Edge,
    EdgeType,
    GenerationResult,
    Node,
    RetrievalResult,
    Trace,
    WriteResult,
)

__version__ = "0.1.0"

__all__ = [
    "Memory",
    "HallucinationBlocked",
    "Embedder",
    "HashEmbedder",
    "TfidfEmbedder",
    "BGEEmbedder",
    "auto_embedder",
    "ParaphraseEnsembleEmbedder",
    "WhiteningProjector",
    "CanonicalEmbedder",
    "LLMClient",
    "MockLLMClient",
    "OpenAIClient",
    "RouterClient",
    "LLMQualityClassifier",
    "Consolidator",
    "ConsolidationStats",
    "Drawer",
    "Node",
    "Edge",
    "EdgeType",
    "AuditEntry",
    "WriteResult",
    "RetrievalResult",
    "GenerationResult",
    "Trace",
    "__version__",
]
