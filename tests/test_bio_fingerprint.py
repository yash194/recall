"""Tests for the bio-fingerprint hard-reject signal (mem0 #4573 fix)."""
from __future__ import annotations

from recall.write.bio_fingerprint import (
    conversation_anchors,
    count_attribute_types,
    is_fabricated_bio,
)


# The exact slip-through from mem0 #4573: 4 attribute types, no anchors
JOHN_DOE = "User John Doe is a Google engineer based in San Francisco, age 32."

# Real fact (no bio fingerprint)
REAL = "We migrated our queue from Postgres LISTEN/NOTIFY to Redis Streams in March."


def test_count_attributes_on_fabricated_bio():
    n = count_attribute_types(JOHN_DOE)
    assert n >= 3  # job + employer + city + age + bio-opener


def test_count_attributes_on_real_fact():
    assert count_attribute_types(REAL) <= 1


def test_is_fabricated_bio_no_anchors():
    """John Doe pattern with NO conversation context → reject."""
    convo = ["We're discussing the queue infrastructure.", "Need to fix message loss."]
    assert is_fabricated_bio(JOHN_DOE, convo) is True


def test_is_fabricated_bio_with_anchors():
    """Same pattern but conversation MENTIONS John & Google → don't reject."""
    convo = [
        "John joined the team last month.",
        "He's coming from Google where he led the queues team.",
    ]
    # 'John' and 'Google' appear → shouldn't reject as fabricated
    assert is_fabricated_bio(JOHN_DOE, convo) is False


def test_real_facts_pass_through():
    convo = ["Discussing queue infrastructure."]
    assert is_fabricated_bio(REAL, convo) is False


def test_no_conversation_no_reject():
    """If there's no conversation buffer at all, can't anchor — pass through."""
    # When conversation is empty, anchor count is 0 — depending on attribute count
    # we may or may not reject. Implementation rejects if attrs >= 3 and anchors < 2.
    # With empty convo, anchors = 0, attrs = 4, → rejected.
    assert is_fabricated_bio(JOHN_DOE, []) is True


def test_threshold_tuning():
    # Lower attribute threshold should still reject the John Doe pattern
    assert is_fabricated_bio(JOHN_DOE, [], attribute_threshold=2) is True
    # Higher threshold should NOT reject (only 4 attrs, not 5)
    assert is_fabricated_bio(JOHN_DOE, [], attribute_threshold=10) is False
