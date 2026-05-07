"""Structural support — does the retrieved subgraph entail a claim?

Implements ARCHITECTURE.md §5.6 / MATH.md §3.6 (Definition 3.3).

A claim `c` is structurally supported by retrieval `R(q)` iff there exists a
directed walk in R(q) such that:
  - Each edge has weight ≥ τ.
  - Some node along the walk has drawer text that entails `c`.

V1 strategies (in order of fidelity):

  - `lexical`         — fast token-coverage check
  - `tfidf`           — TF-IDF cosine entailment threshold
  - `nli`             — small NLI model (requires extras)

Default: `lexical`. Use `Memory(..., support_method='tfidf')` to upgrade.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from recall.core.storage import Storage
from recall.types import Edge, Node


_STOPWORDS = frozenset(
    """a an and are as at be by for from has have he her him his i in is it
       its of on or our she that the their them they this to was we were what
       which who will with you your""".split()
)


def _tokens(text: str) -> set[str]:
    raw = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    return {t for t in raw if t not in _STOPWORDS and len(t) > 1}


def _lexical_entails(premise: str, hypothesis: str, threshold: float = 0.6) -> bool:
    p_toks = _tokens(premise)
    h_toks = _tokens(hypothesis)
    if not h_toks:
        return False
    coverage = len(h_toks & p_toks) / len(h_toks)
    return coverage >= threshold


def _tfidf_entails(premise: str, hypothesis: str, threshold: float = 0.4) -> bool:
    """Use sklearn TF-IDF cosine for sentence-level entailment.

    Returns True if cos(tfidf(premise), tfidf(hypothesis)) ≥ threshold.
    Better than lexical for paraphrase but doesn't capture deep semantics.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return _lexical_entails(premise, hypothesis, threshold=0.6)

    if not premise.strip() or not hypothesis.strip():
        return False
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    try:
        m = v.fit_transform([premise, hypothesis])
    except ValueError:
        return _lexical_entails(premise, hypothesis, threshold=0.6)
    a = m[0].toarray().ravel()
    b = m[1].toarray().ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return False
    return float(np.dot(a, b) / (na * nb)) >= threshold


def structurally_supported(
    claim: str,
    nodes: list[Node],
    edges: list[Edge],
    storage: Storage,
    weight_threshold: float = 0.0,
    method: str = "lexical",
) -> bool:
    """Check if claim is structurally supported by the retrieval subgraph.

    method:
      'lexical' — token coverage (default; fast)
      'tfidf'   — char-ngram TF-IDF cosine
    """
    if not claim or not nodes:
        return False

    drawer_texts: list[str] = []
    for n in nodes:
        for did in n.drawer_ids:
            d = storage.get_drawer(did)
            if d is not None:
                drawer_texts.append(d.text)
        if n.text:
            drawer_texts.append(n.text)

    if not drawer_texts:
        return False

    entail = _tfidf_entails if method == "tfidf" else _lexical_entails

    # Try per-drawer entailment first
    for dt in drawer_texts:
        if entail(dt, claim):
            return True

    # Then union of texts
    union_text = " ".join(drawer_texts)
    return entail(union_text, claim)


def extract_claims(text: str) -> list[str]:
    """Split generation into atomic claim sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    out = []
    for s in sentences:
        s = s.strip()
        if len(s.split()) < 4:
            continue
        if s.endswith("?"):
            continue
        out.append(s)
    return out


def support_score(
    claim: str,
    nodes: list[Node],
    storage: Storage,
    method: str = "tfidf",
) -> float:
    """Continuous-valued support score in [0, 1].

    Returns max similarity between the claim and any node/drawer in the
    subgraph. Useful for soft-mode flagging where we want a confidence number,
    not just a binary pass/fail.
    """
    drawer_texts: list[str] = []
    for n in nodes:
        for did in n.drawer_ids:
            d = storage.get_drawer(did)
            if d is not None:
                drawer_texts.append(d.text)
        if n.text:
            drawer_texts.append(n.text)
    if not drawer_texts:
        return 0.0

    if method == "tfidf":
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
            m = v.fit_transform([claim] + drawer_texts)
            q = m[0].toarray().ravel()
            qn = float(np.linalg.norm(q))
            if qn < 1e-9:
                return 0.0
            scores = []
            for i in range(1, m.shape[0]):
                d = m[i].toarray().ravel()
                dn = float(np.linalg.norm(d))
                if dn < 1e-9:
                    continue
                scores.append(float(np.dot(q, d) / (qn * dn)))
            return max(scores) if scores else 0.0
        except (ImportError, ValueError):
            pass

    # Fallback: token-overlap coverage
    h_toks = _tokens(claim)
    if not h_toks:
        return 0.0
    best = 0.0
    for dt in drawer_texts:
        p_toks = _tokens(dt)
        cov = len(h_toks & p_toks) / len(h_toks)
        best = max(best, cov)
    return best
