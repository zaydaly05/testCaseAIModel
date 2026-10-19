"""Score test cases against the story they were written for, with DeepEval's LLM-as-a-judge metrics.

Each acceptance criterion is one DeepEval test case:

    input           the story (summary, description) and the criterion
    actual_output   the test cases written for that criterion, as plain text
    context         the story and the criterion: the only facts the tests may rely on

The same metrics score this project's model and, optionally, a baseline LLM ("Model X") that writes
its own test cases for the same criterion, so the two can be compared side by side.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.metrics import AnswerRelevancyMetric, GEval, HallucinationMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

from testcase_ai import TestCaseModel, TestSuite, UserStory
from testcase_ai.models import TestCase

OWN_MODEL = "own model"

# provider -> (DeepEval class name, default model, API key variables, OpenAI-compatible endpoint)
_PROVIDERS = {
    "groq": ("OpenAIModel", "openai/gpt-oss-120b", ("GROQ_API_KEY",), "https://api.groq.com/openai/v1"),
    "gemini": ("GeminiModel", "gemini-flash-latest", ("GEMINI_API_KEY", "GOOGLE_API_KEY"), None),
}


def _api_key(provider: str) -> str | None:
    return next((os.environ[k].strip() for k in _PROVIDERS[provider][2] if os.getenv(k, "").strip()), None)


def llm(spec: str | None = None) -> DeepEvalBaseLLM | None:
    """An LLM from "provider:model" (e.g. "groq:openai/gpt-oss-120b", "gemini:gemini-flash-latest").

    Without a spec the first provider with an API key in the environment is used; None if there is none.
    """
    if not spec:
        spec = next((p for p in _PROVIDERS if _api_key(p)), None)
        if spec is None:
            return None
    provider, _, name = spec.partition(":")
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown LLM provider {provider!r}; use one of {', '.join(_PROVIDERS)}")
    class_name, default, keys, base_url = _PROVIDERS[provider]
    import deepeval.models

    kwargs = {"api_key": _api_key(provider)} if keys else {}
    if base_url:
        kwargs["base_url"] = base_url
    return Patient(getattr(deepeval.models, class_name)(model=name or default, **kwargs))


_BUSY = ("429", "503", "rate limit", "ratelimit", "resource_exhausted", "unavailable", "overloaded")


def _wait(error: Exception, attempt: int) -> float | None:
    """Seconds to wait before retrying a "busy" answer (as the provider asks, if it says); None = not busy."""
    text = f"{type(error).__name__} {error}".lower()
    if not any(b in text for b in _BUSY):
        return None
    asked = re.search(r"(?:try again|retry) in ([\d.]+)\s*(ms|s)", text)
    if asked:
        return float(asked[1]) / (1000 if asked[2] == "ms" else 1) + 1
    return min(10 * 2 ** attempt, 120)


class Patient(DeepEvalBaseLLM):
    """Wraps an LLM so rate limits and outages are waited out, with at most 2 calls in flight.

    Free tiers (Groq: 8K tokens a minute) reject most calls when DeepEval sends them all at once.
    """

    def __init__(self, inner: DeepEvalBaseLLM, attempts: int = 10):
        self.inner, self.attempts, self._slots = inner, attempts, None
        super().__init__(inner.get_model_name())

    def load_model(self):
        return self.inner

    def generate(self, prompt, schema=None):
        for attempt in range(self.attempts):
            try:
                return self.inner.generate(prompt, schema=schema)
            except Exception as e:  # noqa: BLE001 - every SDK has its own error classes
                wait = _wait(e, attempt)
                if wait is None or attempt == self.attempts - 1:
                    raise
                time.sleep(wait)

    async def a_generate(self, prompt, schema=None):
        self._slots = self._slots or asyncio.Semaphore(2)
        async with self._slots:
            for attempt in range(self.attempts):
                try:
                    return await self.inner.a_generate(prompt, schema=schema)
                except Exception as e:  # noqa: BLE001
                    wait = _wait(e, attempt)
                    if wait is None or attempt == self.attempts - 1:
                        raise
                    await asyncio.sleep(wait)

    def get_model_name(self):
        return self.inner.get_model_name()

    def supports_log_probs(self):
        return self.inner.supports_log_probs()

    def supports_json_mode(self):
        return self.inner.supports_json_mode()

    def supports_structured_outputs(self):
        return self.inner.supports_structured_outputs()

    def supports_temperature(self):
        return self.inner.supports_temperature()

    def supports_multimodal(self):
        return self.inner.supports_multimodal()


def judge() -> DeepEvalBaseLLM | None:
    """The judge: EVAL_JUDGE, else the first provider with an API key."""
    return llm(os.getenv("EVAL_JUDGE"))


# ------------------------------------------------------------------ what is judged

def story_text(story: UserStory) -> str:
    return "\n".join(filter(None, [f"User story {story.key}: {story.summary}", story.description]))


def render(tc: TestCase) -> str:
    """A test case as a tester would read it."""
    lines = [f"Test: {tc.title}"]
    lines += [f"  Precondition: {p}" for p in tc.preconditions]
    lines += [f"  Test data: {d}" for d in tc.test_data]
    lines += [f"  Step {i}: {s.action} -> expected: {s.expected_result}" for i, s in enumerate(tc.steps, 1)]
    return "\n".join(lines)


def by_criterion(suite: TestSuite) -> dict[str, str]:
    """The criterion -> its test cases as text, in the story's order."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for tc in suite.test_cases:
        grouped[tc.acceptance_criterion].append(render(tc))
    order = [i.text for i in suite.insights] or list(grouped)
    return {c: "\n\n".join(grouped[c]) for c in order if grouped.get(c)}


BASELINE_PROMPT = """You are a senior QA engineer. Write the test cases for ONE acceptance criterion of a Jira user story.
Test the functionality and logic from every side the criterion implies (main behaviour, each condition reversed,
each listed option, limits on both sides, errors), and nothing the story does not state.

{story}
Acceptance criterion: {criterion}

Answer in plain text only, in exactly this format, one block per test case:
Test: <title>
  Precondition: <precondition>
  Test data: <value>
  Step 1: <action> -> expected: <expected result>
"""


def baseline_tests(model: DeepEvalBaseLLM, story: UserStory, criterion: str) -> str:
    """Model X: the test cases a general-purpose LLM writes for the criterion."""
    out = model.generate(BASELINE_PROMPT.format(story=story_text(story), criterion=criterion))
    return (out[0] if isinstance(out, tuple) else out).strip()


def cases_for(story: UserStory, tests: dict[str, str], generator: str) -> list[LLMTestCase]:
    return [
        LLMTestCase(
            name=f"{generator} | {story.key} | {criterion[:60]}",
            input=f"{story_text(story)}\nAcceptance criterion: {criterion}",
            actual_output=text,
            context=[story_text(story), f"Acceptance criterion: {criterion}"],
            metadata={"generator": generator, "story": story.key, "criterion": criterion},
        )
        for criterion, text in tests.items()
    ]


# ------------------------------------------------------------------ the metrics

def metrics(model: DeepEvalBaseLLM | None, threshold: float = 0.7) -> list:
    """Hallucination and relevancy from DeepEval, plus three G-Eval rubrics written for test cases."""
    io = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]
    rubric = dict(evaluation_params=io, model=model, threshold=threshold, async_mode=True)
    return [
        GEval(name="Grounded in the criteria", **rubric, evaluation_steps=[
            "List every feature, field, value, endpoint, role, message and rule the test cases rely on.",
            "Check each one against the user story and the acceptance criterion in the input.",
            "Accept what is stated or directly implied: the reverse of a stated condition or input (an unregistered or "
            "malformed value where the story names a registered or valid one), and generic negative checks on what "
            "the criterion names (unauthenticated request, empty input, unknown id).",
            "Accept boundary values of a limit the story states as grounded, not invented: for 'at least 12 "
            "characters', 11/12/13 characters; for 'expires after 24 hours', just before and just after 24 hours. "
            "They test the stated limit from both sides.",
            "Penalise heavily every invented feature, field, rule, error message or behaviour the story does not mention.",
        ]),
        GEval(name="Covers every side", **rubric, evaluation_steps=[
            "Identify what the acceptance criterion requires: its main behaviour, each condition, each listed "
            "option or value, every limit and every error or edge case it implies.",
            "Check that the main behaviour is tested with its outcome verified, not only input validation.",
            "Check that each condition is also tested reversed, each option has a test, and each limit is tested "
            "on both sides.",
            "Score by the share of these items that are tested; missing negative paths lower the score.",
        ]),
        GEval(name="Logically correct", **rubric, evaluation_steps=[
            "For each test, check that the expected results follow from the preconditions, test data and steps "
            "according to the acceptance criterion.",
            "Penalise expected results that contradict the criterion, tests whose data cannot trigger the case "
            "they claim to test, and steps a tester could not carry out.",
        ]),
        HallucinationMetric(threshold=threshold, model=model, async_mode=True),  # 1 = nothing contradicts the story
        AnswerRelevancyMetric(threshold=threshold, model=model, async_mode=True),
    ]


# ------------------------------------------------------------------ running a review

@dataclass
class Score:
    metric: str
    score: float
    passed: bool
    reason: str = ""


@dataclass
class CriterionReview:
    generator: str
    story: str
    criterion: str
    tests: str
    scores: list[Score] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # judge calls that failed (quota, outage): no score

    @property
    def passed(self) -> bool:
        return bool(self.scores) and not self.errors and all(s.passed for s in self.scores)


def review(stories: list[UserStory], judge_model: DeepEvalBaseLLM | None = None,
           baseline: DeepEvalBaseLLM | None = None, own: TestCaseModel | None = None,
           threshold: float = 0.7, max_cases: int = 40, quiet: bool = True,
           throttle: float = 0) -> list[CriterionReview]:
    """Score the own model's test cases for each story, and the baseline LLM's when one is given.

    throttle: seconds between criteria, judged one at a time (for rate-limited free tiers).
    """
    own = own or TestCaseModel.load_portable()
    judge_model = judge_model or judge()
    if judge_model is None:
        raise RuntimeError("No judge LLM: set GROQ_API_KEY or GEMINI_API_KEY (EVAL_JUDGE picks the model)")
    cases: list[LLMTestCase] = []
    for story in stories:
        ours = by_criterion(own.generate(story, max_cases))
        cases += cases_for(story, ours, OWN_MODEL)
        if baseline is not None:
            theirs = {c: baseline_tests(baseline, story, c) for c in ours}
            cases += cases_for(story, theirs, f"baseline ({baseline.get_model_name()})")
    if not cases:
        return []
    result = evaluate(
        cases, metrics(judge_model, threshold),
        async_config=AsyncConfig(max_concurrent=1, throttle_value=throttle) if throttle else AsyncConfig(max_concurrent=4),
        display_config=DisplayConfig(print_results=not quiet, show_indicator=not quiet),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        error_config=ErrorConfig(ignore_errors=True),
    )
    reviews = []
    for r in result.test_results:
        m = r.metadata or {}
        reviews.append(CriterionReview(
            m.get("generator", ""), m.get("story", ""), m.get("criterion", ""), r.actual_output or "",
            [Score(d.name.removesuffix(" [GEval]"), d.score or 0.0, bool(d.success), d.reason or "")
             for d in r.metrics_data or [] if not d.error],  # a judge call that failed is not a score of 0
            [f"{d.name.removesuffix(' [GEval]')}: {d.error.splitlines()[0][:200]}"
             for d in r.metrics_data or [] if d.error],
        ))
    judged = {(r.generator, r.story, r.criterion) for r in reviews}
    for c in cases:  # DeepEval drops cases it cancelled; report them instead of losing them
        m = c.metadata
        if (m["generator"], m["story"], m["criterion"]) not in judged:
            reviews.append(CriterionReview(m["generator"], m["story"], m["criterion"], c.actual_output,
                                           errors=["not evaluated (cancelled or timed out)"]))
    return sorted(reviews, key=lambda r: (r.story, r.criterion, r.generator != OWN_MODEL))
