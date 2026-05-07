"""Quality classifier for write-time gating.

The mem0 #4573 audit showed 97.8% junk after 32 days. The pattern is:
  - Boilerplate LLM responses ("As an AI language model...")
  - Re-extraction of system prompts
  - Hallucinated profile claims with no support in the conversation
  - Trivial repeats / acknowledgments

v0.7: bio-fingerprint anchor-checking is now wired in by default. The previous
implementation defined `is_fabricated_bio()` but never called it from the
write path — `WritePipeline` was passing `conversation=...` to a `classify()`
that didn't accept it, hitting a TypeError fallback that silently disabled
the gate. This module now accepts (and uses) the conversation context.
"""
from __future__ import annotations

from recall.config import Config, DEFAULT
from recall.types import Node
from recall.write.bio_fingerprint import is_fabricated_bio


class QualityClassifier:
    def __init__(self, config: Config | None = None):
        self._cfg = config or DEFAULT

    def score(self, node: Node, conversation: list[str] | None = None) -> float:
        """Returns quality score in [0.0, 1.0]. Lower = more likely junk.

        v0.7: when ``conversation`` is supplied, the bio-fingerprint detector
        runs against that conversation buffer. A fabricated profile claim
        with 3+ specific personal attributes and no anchor in conversation
        is hard-rejected (score 0.05) regardless of surface plausibility.
        """
        text = (node.text or "").strip()
        if not text:
            return 0.0

        # Length penalty for trivially short utterances
        if len(text) < 10:
            return 0.05

        text_lc = text.lower()

        # Negative-template hit → strong rejection signal
        for pattern in self._cfg.QUALITY_NEGATIVE_TEMPLATES:
            if pattern.lower() in text_lc:
                return 0.1

        # v0.7: bio-fingerprint hard-reject for fabricated profile claims.
        # Only fires when a conversation buffer is provided — without one
        # we can't tell if the claim is anchored, so we skip the check.
        if conversation:
            try:
                if is_fabricated_bio(text, conversation):
                    return 0.05
            except Exception:
                pass  # fail-open: don't break ingestion on a fingerprint bug

        # Specificity heuristic: ratio of unique tokens to total tokens
        tokens = text_lc.split()
        if len(tokens) < 3:
            return 0.15
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio < 0.4:
            return 0.2  # very repetitive → junk

        # Length boost for substantive content
        length_bonus = min(1.0, len(tokens) / 30.0)

        # Default high quality if it survives the negative patterns
        return max(0.5, min(1.0, 0.4 + 0.5 * length_bonus + 0.2 * unique_ratio))

    def classify(
        self, node: Node, conversation: list[str] | None = None,
    ) -> tuple[float, str]:
        """Returns (score, status) where status ∈ {'rejected', 'pending'}.

        v0.7: now accepts the conversation buffer so the write pipeline's
        ``classify(node, conversation=recent)`` call no longer hits TypeError
        and silently falls back to the no-anchor path.
        """
        score = self.score(node, conversation=conversation)
        status = "rejected" if score < self._cfg.THRESH_QUALITY else "pending"
        return score, status
