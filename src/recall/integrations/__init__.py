"""Integration adapters for popular agent frameworks.

  - LettaMemoryBackend  — drop-in replacement for Letta's archival memory
  - LangGraphMemoryNode — LangGraph node wrapper

These are thin shims; the substrate is Recall, the framework just calls our
public API.
"""
from recall.integrations.letta_adapter import LettaMemoryBackend

__all__ = ["LettaMemoryBackend"]
