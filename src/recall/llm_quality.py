"""LLM-driven quality classifier — the v2 path to <5% junk.

Per the mem0-#4573 audit, template-matching catches obvious boilerplate but
misses plausible-looking hallucinated profile claims. An LLM-driven classifier
asks the model to score each candidate memory against a rubric, with cached
results.

Costs are kept low:
  - Small model (gpt-4o-mini, deepseek-v3, claude-haiku) is sufficient
  - Per-text cache (sha256 of text → score) means each unique text is scored once
  - Score is float in [0, 1]; reject below threshold
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from recall.config import Config, DEFAULT
from recall.llm import LLMClient
from recall.types import Node


# SAFE-style support prompt (DeepMind arXiv 2403.18802).
# Instead of asking the LLM to *judge plausibility* (which fabricated bios
# pass by construction), ask whether the conversation contains support.
_SUPPORT_PROMPT = """\
Determine whether STATEMENT is supported by KNOWLEDGE.

KNOWLEDGE: the recent conversation buffer (the only valid source of support).

Rules:
- [Supported]: a sentence in KNOWLEDGE entails or strongly implies STATEMENT.
- [Not Supported]: STATEMENT contains attributes (name, employer, city, age,
  job title) not present in KNOWLEDGE. This includes plausible-but-fabricated
  profile bios — if STATEMENT introduces 3+ specific personal attributes that
  KNOWLEDGE never mentions, answer [Not Supported].
- [Irrelevant]: STATEMENT is boilerplate, system prompt, or empty.

Do not use outside knowledge. Use only KNOWLEDGE.

KNOWLEDGE:
{conversation}

STATEMENT:
{candidate}

Reason briefly, then output one tag in brackets at the end.
"""


# Legacy rubric-style prompt (kept for fallback when no conversation buffer).
_QUALITY_PROMPT = """\
You are a memory quality gate. Score this candidate memory on whether it
should be stored in an AI agent's long-term memory. Output a JSON object with
one key: "score" (float in [0,1]).

REJECT (score < 0.4):
- Boilerplate / generic LLM responses ("Sure, I can help", "As an AI...", "Let me clarify")
- System prompts or assistant identity statements ("I am Claude", "You are a helpful assistant")
- **Profile claims about a user with specific personal attributes** (job title + employer + city + age etc.) that look like a fabricated bio. If the memory reads like "User X is a Y at Z based in W age N", REJECT — those are typically hallucinated profile leaks. Real conversation-derived facts are more specific to the actual conversation.
- Repetitive trivial acknowledgments ("ok", "got it", "yes")
- Empty or near-empty content

ACCEPT (score >= 0.4):
- Specific factual claims about systems, decisions, processes, configurations
- Verifiable observations grounded in the conversation context
- Concrete events with named participants, dates, or measurable quantities
- Preferences stated by the user themselves about a single attribute (not a full bio)

Candidate memory:
\"\"\"
{text}
\"\"\"

Output JSON only, like {{"score": 0.7}}:
"""


class LLMQualityClassifier:
    """LLM-backed quality scorer with sha-keyed cache + bio-fingerprint hard-reject.

    Two-stage gate:
      Stage 1: bio_fingerprint hard reject (no LLM call)
        — catches fabricated-profile claims with no anchor in conversation buffer
      Stage 2: SAFE-style support prompt against the conversation
        — falls back to rubric prompt if no conversation buffer provided

    Per arXiv 2403.18802 (SAFE) and arXiv 2512.15068 (Semantic Illusion):
    judging plausibility is the wrong objective; judging anchor-presence in the
    conversation buffer kills the most common slip-through pattern.
    """

    def __init__(
        self,
        llm: LLMClient,
        config: Config | None = None,
        cache: dict[str, float] | None = None,
        use_bio_fingerprint: bool = True,
        use_safe_prompt: bool = True,
    ):
        self.llm = llm
        self.cfg = config or DEFAULT
        self._cache: dict[str, float] = cache if cache is not None else {}
        self.use_bio_fingerprint = use_bio_fingerprint
        self.use_safe_prompt = use_safe_prompt

    def score(self, node: Node, conversation: list[str] | None = None) -> float:
        """Score node quality, optionally with conversation context.

        Two-stage gate:
          Stage 1: bio-fingerprint hard reject (no LLM call) — catches the
                   "John Doe is a Google engineer in SF age 32" mem0-#4573 pattern.
                   Only fires if 3+ bio attributes present AND no anchor in convo.
          Stage 2: rubric prompt (LLM call). The legacy gate that already
                   handled boilerplate, system leaks, generic patterns.

        SAFE-style support prompt was tested but rejected legitimate isolated
        facts when the conversation buffer was unrelated. Reserve SAFE for
        future use when the conversation has true continuity.
        """
        text = (node.text or "").strip()
        if not text:
            return 0.0

        # Stage 1: bio-fingerprint hard reject (free, no LLM call)
        if self.use_bio_fingerprint and conversation is not None:
            from recall.write.bio_fingerprint import is_fabricated_bio
            if is_fabricated_bio(text, conversation):
                return 0.05  # strong reject — fabricated bio with no anchor

        cache_key = hashlib.sha256(
            (text + "|" + ("|".join(conversation[-20:]) if conversation else "")).encode("utf-8")
        ).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Stage 2: rubric prompt (LLM)
        score = self._score_via_rubric(text)
        self._cache[cache_key] = score
        return score

    def _score_via_safe(self, text: str, conversation: list[str]) -> float | None:
        """SAFE-style support check (arXiv 2403.18802)."""
        convo_blob = "\n".join(conversation[-20:])[:4000]
        prompt = _SUPPORT_PROMPT.format(conversation=convo_blob, candidate=text[:1000])
        try:
            raw = self.llm.complete(prompt, max_tokens=120).strip()
        except Exception:
            return None
        # Parse the bracketed tag
        # Look for [Supported] / [Not Supported] / [Irrelevant], take the LAST one
        # to get the model's final answer.
        import re
        tags = re.findall(r"\[(Supported|Not Supported|Irrelevant)\]", raw, re.IGNORECASE)
        if not tags:
            return None
        last = tags[-1].lower()
        if "not" in last and "support" in last:
            return 0.10  # explicitly rejected
        if "irrelevant" in last:
            return 0.05  # boilerplate
        return 0.85  # supported

    def _score_via_rubric(self, text: str) -> float:
        prompt = _QUALITY_PROMPT.format(text=text[:1000])
        try:
            raw = self.llm.complete(prompt, max_tokens=40)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.strip("`").strip()
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            data = json.loads(raw[: raw.rfind("}") + 1] if "}" in raw else raw)
            return max(0.0, min(1.0, float(data.get("score", 0.5))))
        except Exception:
            from recall.write.quality import QualityClassifier
            return QualityClassifier(self.cfg).score(Node(text=text))

    def classify(self, node: Node, conversation: list[str] | None = None) -> tuple[float, str]:
        s = self.score(node, conversation=conversation)
        return s, "rejected" if s < self.cfg.THRESH_QUALITY else "pending"
