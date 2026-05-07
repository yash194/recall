"""Edge-type classification.

Classifies a directed Γ-edge `(src, dst)` into one of:
  supports / contradicts / corrects / agrees / pivots / temporal_next / superseded

V1: rule-based classifier using keyword patterns + Γ_anti sign.
V2: small fine-tuned classifier (optional LLM fallback).
"""
from __future__ import annotations

import re

from recall.types import EdgeType, Node


_PATTERNS = {
    EdgeType.CONTRADICTS: [
        r"\b(but|however|on the contrary|actually|in fact)\b",
        r"\b(no|not|never|n't)\b",
        r"\b(wrong|incorrect|false)\b",
    ],
    EdgeType.CORRECTS: [
        r"\b(correction|to clarify|let me correct|actually)\b",
        r"\b(should be|meant to say|i meant)\b",
    ],
    EdgeType.PIVOTS: [
        r"\b(switched to|moved to|migrated to|changed to|instead of)\b",
        r"\b(pivot|shift|transition)\b",
    ],
    EdgeType.SUPERSEDED: [
        r"\b(deprecated|outdated|superseded|no longer|replaced)\b",
    ],
    EdgeType.TEMPORAL_NEXT: [
        r"\b(then|after that|next|subsequently|following|later)\b",
        r"\b(\d+ (?:days?|weeks?|months?) (?:later|after))\b",
    ],
    EdgeType.AGREES: [
        r"\b(yes|exactly|correct|right|agreed|confirmed|same)\b",
    ],
    EdgeType.SUPPORTS: [
        r"\b(because|since|as|due to|results in|leads to|causes)\b",
        r"\b(supports|reinforces|implies|entails)\b",
    ],
}


def _matches(patterns: list[str], text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text, flags=re.IGNORECASE))


def classify_edge_type(
    src: Node, dst: Node, gamma: float, gamma_anti: float | None = None
) -> EdgeType:
    """Classify the type of a directed edge (src → dst).

    Strategy:
      1. Strong-signal patterns in dst.text take priority (CONTRADICTS, CORRECTS,
         SUPERSEDED, PIVOTS).
      2. Otherwise, use Γ-anti sign + magnitude:
         - High |Γ_anti|, positive → SUPPORTS or TEMPORAL_NEXT
         - High |Γ_anti|, negative gamma → CONTRADICTS
         - Low Γ_anti, similar → AGREES
      3. Fall back to PENDING.
    """
    src_text = src.text or ""
    dst_text = dst.text or ""
    combined = src_text + " " + dst_text

    # Priority order: more specific patterns first
    if _matches(_PATTERNS[EdgeType.CORRECTS], dst_text):
        return EdgeType.CORRECTS
    if _matches(_PATTERNS[EdgeType.SUPERSEDED], combined) and gamma < 0:
        return EdgeType.SUPERSEDED
    if _matches(_PATTERNS[EdgeType.CONTRADICTS], combined) and gamma < 0.05:
        return EdgeType.CONTRADICTS
    if _matches(_PATTERNS[EdgeType.PIVOTS], combined):
        return EdgeType.PIVOTS
    if _matches(_PATTERNS[EdgeType.TEMPORAL_NEXT], combined):
        return EdgeType.TEMPORAL_NEXT

    # Γ-driven defaults
    if gamma_anti is not None and abs(gamma_anti) > 0.1:
        if gamma > 0:
            return EdgeType.SUPPORTS
        else:
            return EdgeType.CONTRADICTS

    if _matches(_PATTERNS[EdgeType.AGREES], combined) and gamma > 0:
        return EdgeType.AGREES

    if _matches(_PATTERNS[EdgeType.SUPPORTS], combined) and gamma > 0:
        return EdgeType.SUPPORTS

    # Symmetric or weak signal: default to AGREES if positive, CONTRADICTS if negative
    if gamma > 0.1:
        return EdgeType.AGREES
    if gamma < -0.05:
        return EdgeType.CONTRADICTS
    return EdgeType.PENDING


def signed_weight_for_type(edge_type: EdgeType, gamma: float) -> float:
    """Convert raw Γ score to a signed edge weight by edge type.

    Negative-typed edges (CONTRADICTS, SUPERSEDED) get negative weight even if
    Γ was positive; positive types preserve the magnitude.
    """
    magnitude = abs(gamma)
    if EdgeType.is_negative(edge_type):
        return -magnitude
    return gamma if gamma > 0 else magnitude * 0.5  # mild positive for weakly-correlated supportive
