"""DeepEval guardrail, pytest style: every criterion's test cases must be grounded, complete and correct.

    deepeval test run evals/test_deepeval.py     (or: pytest evals)

Needs a judge LLM (EVAL_JUDGE or an API key); skipped otherwise.
"""

import json
from pathlib import Path

import pytest
from deepeval import assert_test

from evals.review import OWN_MODEL, by_criterion, cases_for, judge, metrics
from testcase_ai import TestCaseModel, UserStory

JUDGE = judge()
pytestmark = pytest.mark.skipif(JUDGE is None, reason="no judge LLM configured")

STORIES = [UserStory(**s) for s in json.loads(
    (Path(__file__).resolve().parent.parent / "samples" / "sample_stories.json").read_text(encoding="utf-8"))]
MODEL = TestCaseModel.load_portable() if JUDGE else None
CASES = [c for s in STORIES for c in cases_for(s, by_criterion(MODEL.generate(s, 40)), OWN_MODEL)] if MODEL else []


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_test_cases_match_the_criterion(case):
    assert_test(case, metrics(JUDGE))
