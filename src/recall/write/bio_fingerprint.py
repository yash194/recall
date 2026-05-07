"""Bio-fingerprint detector for fabricated profile claims.

The mem0 #4573 audit slip-through pattern: fabricated profile claims with
many specific attributes ("User John Doe is a Google engineer based in San
Francisco, age 32"). They look plausible because they're specific.

The fingerprint: 3+ attribute-types co-occurring with no anchor in the
recent conversation. If the conversation never mentioned "John", "Google",
"engineer", we know the claim is hallucinated regardless of how plausible
it sounds.

Per arXiv 2512.15068 ("The Semantic Illusion"): cosine-distance hallucination
detection has 100% FPR because hallucinations preserve semantic entailment.
The fix is structural — anchor-presence in the conversation buffer.
"""
from __future__ import annotations

import re
from typing import Sequence

# Attribute-type fingerprints that co-occur in fabricated bios
_BIO_PATTERNS = [
    # Job title patterns
    re.compile(
        r"\bis\s+(?:a|an|the)\s+\w+\s+(engineer|developer|manager|designer|"
        r"scientist|researcher|founder|CEO|CTO|CFO|VP|director|"
        r"analyst|consultant|architect|programmer|lead|head)\b",
        re.IGNORECASE,
    ),
    # Tech-employer patterns
    re.compile(
        r"\b(?:at|works\s+(?:at|for)|employed\s+(?:at|by))\s+"
        r"(Google|Meta|Apple|Microsoft|Amazon|OpenAI|Anthropic|Tesla|"
        r"Netflix|Stripe|Uber|Airbnb|Spotify|Twitter|X|LinkedIn|Salesforce|"
        r"Oracle|IBM|Intel|NVIDIA|AMD)\b",
        re.IGNORECASE,
    ),
    # Geographic patterns
    re.compile(
        r"\b(?:based\s+in|lives?\s+in|located\s+in|from|in)\s+"
        r"(San\s+Francisco|New\s+York|London|Seattle|Boston|Austin|Berlin|"
        r"Paris|Tokyo|Beijing|Shanghai|Singapore|Mumbai|Bangalore|Toronto|"
        r"Dublin|Amsterdam|Stockholm|Helsinki|Sydney|Melbourne)\b",
        re.IGNORECASE,
    ),
    # Age patterns
    re.compile(r"\bage\s+\d{1,3}\b|\b\d{2}\s+years?\s+old\b", re.IGNORECASE),
    # Bio opener patterns: "User X is", "[Name] is a"
    re.compile(
        r"^User\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+is\b|"
        r"^[A-Z][a-z]+\s+[A-Z][a-z]+\s+is\s+(?:a|an|the)\b",
        re.IGNORECASE,
    ),
]


def count_attribute_types(text: str) -> int:
    """Count how many distinct bio-attribute pattern types match the text."""
    return sum(1 for p in _BIO_PATTERNS if p.search(text or ""))


def conversation_anchors(text: str, conversation: Sequence[str]) -> int:
    """Count how many capitalized identifiers from `text` appear in `conversation`."""
    if not conversation:
        return 0
    convo_blob = "\n".join(conversation[-20:])
    # Capitalized words length 3+ — proper nouns / entities
    candidates = set(re.findall(r"\b[A-Z][a-z]{2,}\b", text or ""))
    return sum(1 for c in candidates if c in convo_blob)


def is_fabricated_bio(
    text: str,
    conversation: Sequence[str],
    attribute_threshold: int = 3,
    anchor_threshold: int = 2,
) -> bool:
    """Hard-reject signal for the mem0-#4573 fabricated-bio failure mode.

    Returns True iff the candidate text has attribute_threshold+ bio attribute
    types AND fewer than anchor_threshold capitalized tokens appear in the
    recent conversation buffer.

    With defaults (3 attributes, 2 anchors) this catches "User John Doe is a
    Google engineer based in San Francisco, age 32" when neither "John" nor
    "Google" nor "engineer" appears in the conversation.
    """
    n_attrs = count_attribute_types(text)
    if n_attrs < attribute_threshold:
        return False
    n_anchors = conversation_anchors(text, conversation)
    return n_anchors < anchor_threshold
