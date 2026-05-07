"""Tests for the write-time quality classifier."""
from __future__ import annotations

import numpy as np

from recall.types import Node
from recall.write.quality import QualityClassifier


def _node(text):
    return Node(
        tenant="t1", text=text,
        f_embedding=np.ones(8, dtype=np.float32),
        b_embedding=np.ones(8, dtype=np.float32),
    )


def test_quality_rejects_short_text():
    qc = QualityClassifier()
    score, status = qc.classify(_node("ok"))
    assert status == "rejected"


def test_quality_rejects_negative_template():
    qc = QualityClassifier()
    score, status = qc.classify(_node("As an AI language model, I cannot help with that."))
    assert status == "rejected"


def test_quality_accepts_substantive_text():
    qc = QualityClassifier()
    text = (
        "We made the architectural decision to switch from Postgres LISTEN/NOTIFY "
        "to Redis Streams after the queue began losing messages under load."
    )
    score, status = qc.classify(_node(text))
    assert status == "pending"
    assert score > 0.4


def test_quality_rejects_very_repetitive():
    qc = QualityClassifier()
    text = "yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes"
    score, status = qc.classify(_node(text))
    assert status == "rejected"
