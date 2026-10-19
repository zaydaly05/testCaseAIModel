"""The DeepEval review pipeline, run end to end with a fake judge (no API key, no network)."""

import typing

import pytest

pytest.importorskip("deepeval")

from deepeval.models.base_model import DeepEvalBaseLLM  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from evals.compare import markdown, quality, summary  # noqa: E402
from evals.review import OWN_MODEL, by_criterion, llm, review  # noqa: E402
from testcase_ai import TestCaseModel, UserStory  # noqa: E402


def _fill(tp):
    """A valid value of any type DeepEval asks for: agreeing verdicts and a high score."""
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        return _fill(next(a for a in args if a is not type(None)))
    if origin is list:
        return [_fill(args[0])]
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return tp(**{n: _fill(f.annotation) for n, f in tp.model_fields.items()})
    if tp in (int, float):
        return 9
    return "yes"


class FakeJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.prompts = []
        super().__init__("fake-judge")

    def load_model(self):
        return self

    def generate(self, prompt, schema=None):
        self.prompts.append(prompt)
        return _fill(schema) if schema else "Test: Fake baseline test\n  Step 1: Do it -> expected: It works"

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "fake-judge"


STORY = UserStory(key="T-1", summary="Leave a community review",
                  acceptance_criteria="- Given a rating outside 1-5, then it is clamped to the range.\n"
                                      "- Given name and comment are provided, when I submit, then the review is saved.")


@pytest.fixture(scope="module")
def model():
    return TestCaseModel.load_portable()


def test_each_criterion_is_judged_with_its_own_tests(model):
    tests = by_criterion(model.generate(STORY, 40))
    assert len(tests) == 2
    assert all(t.startswith("Test: ") and "-> expected:" in t for t in tests.values())


def test_review_scores_both_generators(model):
    fake = FakeJudge()
    reviews = review([STORY], fake, baseline=fake, own=model)
    assert {r.generator for r in reviews} == {OWN_MODEL, "baseline (fake-judge)"}
    assert all({s.metric for s in r.scores} == {"Grounded in the criteria", "Covers every side", "Logically correct",
                                                "Hallucination", "Answer Relevancy"} for r in reviews)
    assert any("clamped" in p for p in fake.prompts)  # the criterion reaches the judge
    table = summary(reviews)
    assert table[OWN_MODEL]["pass rate"] == 1.0 and quality(table[OWN_MODEL]) > 0.7
    assert "| **quality** |" in markdown(reviews, "fake-judge")


def test_unknown_provider_is_refused():
    with pytest.raises(ValueError):
        llm("nope:model")
