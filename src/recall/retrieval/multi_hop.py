"""Multi-hop retrieval — entity-aware iterative expansion.

Inspired by HippoRAG (Gutiérrez et al. 2024, NeurIPS) and GraphRAG (Microsoft 2024).
The idea: when a question chains across N facts (e.g., "country of citizenship of
the spouse of the author of X"), the gold answer node is rarely cosine-similar to
the question itself. It IS cosine-similar to entities mentioned in intermediate
facts.

Algorithm:
    1. Get k_init seeds by cosine on the raw query.
    2. For each seed, extract capitalized-phrase entities (a cheap NER proxy).
    3. For each entity, do another cosine top-k as if the entity were a query.
       This surfaces facts that mention the entity but aren't surface-similar
       to the original question.
    4. Aggregate. Rank candidates by max similarity to either (a) the original
       query OR (b) any of the extracted entities. Return top-k_final.

This bypasses the limitation that single-hop cosine retrieval can't traverse
compositional questions when intermediate concepts (entity names) aren't in
the question text.

References:
  Gutiérrez et al. 2024. HippoRAG: Neurobiologically Inspired Long-Term Memory
    for Large Language Models. NeurIPS. arXiv:2405.14831
  Edge et al. 2024. From Local to Global: A Graph RAG Approach to
    Query-Focused Summarization. arXiv:2404.16130
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from recall.types import Node


# Match capitalized phrases that are 1–4 tokens long, allowing internal
# apostrophes and hyphens but starting with a capital letter. This catches
# proper-noun entities (people, places, organizations) without needing spaCy.
_ENTITY_RE = re.compile(
    r"\b[A-Z][a-zA-Z'-]+(?:\s+(?:of|and|the|de|von|van|la|le|du|der)\s+[A-Z][a-zA-Z'-]+)*"
    r"(?:\s+[A-Z][a-zA-Z'-]+){0,3}\b"
)

# Phrases to strip from question-extracted entities — function words that
# shouldn't seed a new sub-query on their own.
_BAD_ENTITY_PREFIXES = ("The ", "A ", "An ", "This ", "That ", "Which ", "What ", "Who ", "Where ", "When ", "Why ", "How ")
_STOPWORDS = frozenset(
    "the a an of in on at to for with by from is was were are be been "
    "this that these those which what who where when why how".split()
)


def extract_entities(text: str, *, min_chars: int = 3, max_per_text: int = 12) -> list[str]:
    """Cheap entity extractor: capitalized phrases.

    Returns a list of unique entity strings, sorted by length (longest first)
    so multi-token names take precedence over their parts ("New York" before
    "New").
    """
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _ENTITY_RE.finditer(text):
        ent = m.group(0).strip()
        for prefix in _BAD_ENTITY_PREFIXES:
            if ent.startswith(prefix):
                ent = ent[len(prefix):].strip()
                break
        # Skip pure stopwords or single-char entities
        toks = ent.split()
        if all(t.lower() in _STOPWORDS for t in toks):
            continue
        if len(ent) < min_chars:
            continue
        # Skip generic capitalized words at sentence start (e.g., "The" caught
        # without prefix-stripping)
        if len(toks) == 1 and ent.lower() in _STOPWORDS:
            continue
        key = ent.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(ent)
    # Sort by length desc so "Charles Dickens" precedes "Charles"
    out.sort(key=lambda s: -len(s))
    return out[:max_per_text]


def multi_hop_recall(
    storage,
    embedder,
    query: str,
    *,
    scope: dict | None = None,
    k_init: int = 5,
    k_per_entity: int = 3,
    k_final: int = 10,
    hops: int = 2,
    entity_budget: int = 8,
) -> list[Node]:
    """Multi-hop entity-expanded retrieval (HippoRAG-inspired).

    Algorithm:
        1. Initial cosine retrieval with the raw query → seed nodes.
        2. Extract entities from the question + each retrieved seed.
        3. For each entity, cosine-retrieve top-k_per_entity additional
           candidates.
        4. Score each candidate by max(sim_to_query, 0.6 · sim_to_entity)
           — anchored to the question but able to surface entity-related
           nodes that aren't directly cosine-similar to the question.
        5. Repeat for ``hops`` rounds.
        6. Return top ``k_final`` candidates by score.

    For compositional multi-hop questions, the gold-answer node may have
    low cosine similarity to the question itself but high similarity to
    an intermediate entity. The 0.6 weight on entity similarity is tuned
    to prevent entity-only matches from dominating the cosine seeds.
    """

    def s_emb(text: str) -> np.ndarray:
        if hasattr(embedder, "embed_symmetric"):
            return embedder.embed_symmetric(text)
        f, b = embedder.embed_dual(text)
        return (f + b) / 2.0

    scope = scope or {}
    query_vec = s_emb(query)

    # Round 0: cosine seeds from the raw query
    initial_seeds = storage.topk_cosine(query_vec, scope, k=k_init)
    candidates: dict[str, tuple[Node, float]] = {}
    for n in initial_seeds:
        node_vec = n.s_embedding if n.s_embedding is not None else None
        sim = (
            float(np.dot(query_vec, node_vec)
                  / (np.linalg.norm(query_vec) * np.linalg.norm(node_vec) + 1e-12))
            if node_vec is not None else 0.0
        )
        candidates[n.id] = (n, sim)

    # Question-level entities — useful for bootstrap when seeds are weak
    seeded_entities: list[str] = list(extract_entities(query, max_per_text=4))

    # Iterative expansion
    frontier: list[Node] = list(initial_seeds)
    for hop in range(hops):
        new_entities: list[str] = []
        # Always try the question's own entities at hop 0
        if hop == 0:
            new_entities.extend(seeded_entities)
        for n in frontier:
            new_entities.extend(extract_entities(n.text, max_per_text=4))
            if len(new_entities) >= entity_budget * (hop + 1):
                break

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for e in new_entities:
            key = e.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(e)
        deduped = deduped[: entity_budget]

        new_frontier: list[Node] = []
        for ent in deduped:
            ent_vec = s_emb(ent)
            hits = storage.topk_cosine(ent_vec, scope, k=k_per_entity)
            for n in hits:
                node_vec = n.s_embedding if n.s_embedding is not None else None
                if node_vec is None:
                    continue
                sim_to_query = float(
                    np.dot(query_vec, node_vec)
                    / (np.linalg.norm(query_vec) * np.linalg.norm(node_vec) + 1e-12)
                )
                sim_to_entity = float(
                    np.dot(ent_vec, node_vec)
                    / (np.linalg.norm(ent_vec) * np.linalg.norm(node_vec) + 1e-12)
                )
                score = max(sim_to_query, 0.6 * sim_to_entity)
                if n.id not in candidates or candidates[n.id][1] < score:
                    candidates[n.id] = (n, score)
                    new_frontier.append(n)

        frontier = new_frontier
        if not frontier:
            break

    # Rank and return top-k_final
    ranked = sorted(candidates.values(), key=lambda x: -x[1])
    return [n for n, _ in ranked[:k_final]]
