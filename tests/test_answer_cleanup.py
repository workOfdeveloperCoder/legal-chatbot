"""Tests for answer text cleanup before API response."""

from app.rag.answer_cleanup import clean_answer_for_display


def test_removes_trailing_sources_block():
    answer = (
        "Section 54-C has a problem with discretion.\n\n"
        "Sources:\n[Source 1]\n[Source 2]"
    )
    cleaned = clean_answer_for_display(answer)
    assert "[Source" not in cleaned
    assert "Sources:" not in cleaned
    assert "Section 54-C" in cleaned


def test_removes_inline_source_markers():
    answer = "The court held this [Source 1] and also noted [Source 2]."
    cleaned = clean_answer_for_display(answer)
    assert cleaned == "The court held this and also noted."


def test_removes_combined_source_marker():
    answer = (
        "According to the article, section 54-C creates a clog "
        "[Source 1, Source 2]."
    )
    cleaned = clean_answer_for_display(answer)
    assert "[Source" not in cleaned
    assert "clog" in cleaned


def test_removes_empty_sources_footer_with_bullets():
    answer = (
        "The article identifies a clog on discretion.\n\n"
        "Sources:\n-\n-"
    )
    cleaned = clean_answer_for_display(answer)
    assert "Sources:" not in cleaned
    assert "-" not in cleaned.split("\n")[-1] if cleaned else True
    assert "clog" in cleaned


def test_removes_trailing_bare_sources_label():
    answer = (
        "Section 54-C restricts judicial discretion.\n\n"
        "Sources:"
    )
    cleaned = clean_answer_for_display(answer)
    assert "Sources:" not in cleaned
    assert "Section 54-C" in cleaned


def test_preserves_normal_answer():
    answer = "The problem is that section 54-C lacks clear guidance."
    assert clean_answer_for_display(answer) == answer


def test_repairs_fragmented_words_from_stream_artifacts():
    raw = (
        "A nswe r:\n\n"
        "Close rel ati ve s of a murder victi m ha ve the opt ion "
        "t o file a pe titi on for le ave to app eal."
    )
    cleaned = clean_answer_for_display(raw)
    assert "Answer" in cleaned
    assert "relatives" in cleaned
    assert "rel ati ve" not in cleaned
    assert "victim" in cleaned
    assert "have" in cleaned
    assert "option" in cleaned
    assert "to file a" in cleaned
    assert "petition" in cleaned
    assert "leave" in cleaned
    assert "appeal" in cleaned


def test_preserves_normal_short_words():
    answer = "Close relatives of a murder victim have the option to file a petition."
    assert clean_answer_for_display(answer) == answer
