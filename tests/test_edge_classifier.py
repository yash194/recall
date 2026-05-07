"""Tests for edge-type classification."""
from __future__ import annotations

import numpy as np

from recall.types import EdgeType, Node
from recall.write.edge_classifier import classify_edge_type, signed_weight_for_type


def _n(text):
    return Node(
        text=text, tenant="t",
        f_embedding=np.ones(8, dtype=np.float32),
        b_embedding=np.ones(8, dtype=np.float32),
    )


def test_classify_contradicts():
    src = _n("We use Postgres.")
    dst = _n("That is wrong; we use Redis.")
    et = classify_edge_type(src, dst, gamma=-0.2, gamma_anti=-0.1)
    assert et == EdgeType.CONTRADICTS


def test_classify_pivots():
    src = _n("We tried Postgres LISTEN/NOTIFY.")
    dst = _n("We switched to Redis Streams.")
    et = classify_edge_type(src, dst, gamma=0.3, gamma_anti=0.2)
    assert et == EdgeType.PIVOTS


def test_classify_temporal_next():
    src = _n("We deployed the change.")
    dst = _n("Then the queue stabilized two days later.")
    et = classify_edge_type(src, dst, gamma=0.3, gamma_anti=0.2)
    assert et == EdgeType.TEMPORAL_NEXT


def test_classify_supports():
    src = _n("Database is overloaded.")
    dst = _n("This causes slow queries.")
    et = classify_edge_type(src, dst, gamma=0.3, gamma_anti=0.2)
    assert et in (EdgeType.SUPPORTS, EdgeType.TEMPORAL_NEXT)


def test_signed_weight_for_negative_type():
    assert signed_weight_for_type(EdgeType.CONTRADICTS, 0.5) < 0
    assert signed_weight_for_type(EdgeType.SUPERSEDED, 0.5) < 0


def test_signed_weight_for_positive_type():
    assert signed_weight_for_type(EdgeType.SUPPORTS, 0.5) > 0
    assert signed_weight_for_type(EdgeType.AGREES, 0.5) > 0
