"""Tests for Qdrant library document chunk stitching."""

from app.services.library_document_service import (
    LibraryDocumentService,
    stitch_chunk_texts,
    _append_with_overlap,
)


def test_append_overlap_mid_word_person_erson():
    left = (
        "Now if a poor person of domestic\n"
        "consumption up to a one kilowatt is sent with a bill of unreasonable amount for\n"
        "instance of Rs.1,00,000 for six months and he is charged again for the same\n"
        "amount"
    )
    right = (
        "erson of domestic\n"
        "consumption up to a one kilowatt is sent with a bill of unreasonable amount for\n"
        "instance of Rs.1,00,000 for six months and he is charged again for the same\n"
        "amount in the next month will face the situation of disconnection"
    )
    merged = _append_with_overlap(left, right)
    assert merged.startswith("Now if a poor person of domestic")
    assert "amount\n\nerson" not in merged
    assert merged.count("one kilowatt") == 1
    assert "disconnection" in merged


def test_stitch_prefers_document_start_when_order_is_wrong():
    head = (
        "CLOG ON DISCRETION\nBy\nAdvocate\n"
        "Now if a poor person of domestic\nconsumption up to a one kilowatt is sent "
        "with a bill of unreasonable amount for\ninstance of Rs.1,00,000 for six months "
        "and he is charged again for the same\namount"
    )
    tail = (
        "erson of domestic\nconsumption up to a one kilowatt is sent with a bill of "
        "unreasonable amount for\ninstance of Rs.1,00,000 for six months and he is "
        "charged again for the same\namount in the next month will face disconnection "
        "to remedy the blind adherence to section 54‑C of the Act."
    )
    # Deliberately pass mid-word chunk first (server-style bad order).
    merged = stitch_chunk_texts([tail, head])
    assert merged.startswith("CLOG ON DISCRETION")
    assert not merged.lstrip().startswith("erson")
    assert "section 54‑C of the Act." in merged
    assert merged.count("one kilowatt") == 1


def test_merge_keeps_all_children_even_with_same_chunk_index():
    class FakePoint:
        def __init__(self, payload):
            self.payload = payload

    head = (
        "CLOG ON DISCRETION\nBy\nAdvocate\n"
        "Now if a poor person of domestic\nconsumption up to a one kilowatt is sent "
        "with a bill of unreasonable amount for\ninstance of Rs.1,00,000 for six months "
        "and he is charged again for the same\namount"
    )
    tail = (
        "erson of domestic\nconsumption up to a one kilowatt is sent with a bill of "
        "unreasonable amount for\ninstance of Rs.1,00,000 for six months and he is "
        "charged again for the same\namount in the next month will face disconnection "
        "to remedy the blind adherence to section 54‑C of the Act."
    )
    # Server-style: both children tagged chunk_index=0 (previously dropped one → half file).
    points = [
        FakePoint(
            {
                "chunk_id": "c0",
                "chunk_index": 0,
                "chunk_role": "child",
                "is_parent": False,
                "chunk_text": head,
            }
        ),
        FakePoint(
            {
                "chunk_id": "c1",
                "chunk_index": 0,
                "chunk_role": "child",
                "is_parent": False,
                "chunk_text": tail,
            }
        ),
    ]
    text, count = LibraryDocumentService._merge_points(points)
    assert count == 2
    assert text.startswith("CLOG ON DISCRETION")
    assert "section 54‑C of the Act." in text
    assert text.count("one kilowatt") == 1


def test_merge_points_uses_parent_or_stitched_full_text():
    class FakePoint:
        def __init__(self, payload):
            self.payload = payload

    parent = (
        "CLOG ON DISCRETION\nBy\nFull parent body with section 54‑C ending here."
    )
    child0 = parent[:40] + " mid " + parent[40:80]
    child1 = "ody with section 54‑C ending here."

    points = [
        FakePoint(
            {
                "chunk_index": 0,
                "chunk_role": "parent",
                "is_parent": True,
                "chunk_text": parent,
            }
        ),
        FakePoint(
            {
                "chunk_index": 0,
                "chunk_role": "child",
                "is_parent": False,
                "chunk_text": child0,
            }
        ),
        FakePoint(
            {
                "chunk_index": 1,
                "chunk_role": "child",
                "is_parent": False,
                "chunk_text": child1,
            }
        ),
    ]
    text, count = LibraryDocumentService._merge_points(points)
    assert text.startswith("CLOG ON DISCRETION")
    assert "54‑C ending here." in text
    assert not text.lstrip().startswith("ody")
    assert count >= 1
