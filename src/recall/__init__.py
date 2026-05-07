"""Recall — memory layer for AI agents.

Public API:

    from recall import Memory, Embedder, LLMClient

    mem = Memory(tenant="my_app")
    mem.observe(user_msg, agent_msg)
    answer = mem.bounded_generate(query)
    trace = mem.trace(answer)
    mem.forget(node_id, reason="outdated")
"""

# v0.2: auto-load `~/.recall/.env` so OPENAI_API_KEY / RECALL_DB_DIR /
# etc. set by `recall-setup` are picked up by the library, CLI, and MCP
# server without requiring the user to manually export them.
def _autoload_env() -> None:
    import os
    from pathlib import Path
    env_path = Path.home() / ".recall" / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(env_path, override=False)
    except ImportError:
        # Manual fallback — don't fail import just because dotenv is missing
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if v and v[0] in "\"'" and v[-1] == v[0]:
                v = v[1:-1]
            os.environ.setdefault(k, v)


_autoload_env()
del _autoload_env

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

__version__ = "0.2.0"

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
