"""Evaluation harness for LegalQueryPlanner against regression cases."""

from __future__ import annotations

import pytest

from app.rag.legal_query_planner import LegalQueryPlanner
from tests.evaluation.legal_regression_cases import LEGAL_REGRESSION_CASES


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [c for c in LEGAL_REGRESSION_CASES if c.expected_answer_mode],
    ids=lambda case: case.question[:48],
)
async def test_planner_matches_expected_answer_mode(case):
    planner = LegalQueryPlanner()
    plan = await planner.plan(
        question=case.question,
        has_uploaded_documents=case.has_uploaded_documents,
    )
    assert plan.answer_mode.value == case.expected_answer_mode
    assert plan.requires_legal_authority == case.requires_legal_authority
