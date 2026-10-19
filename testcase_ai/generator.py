"""Test case generators.

`ClaudeTestCaseGenerator` asks Claude for structured test cases (validated
against the pydantic schema in models.py). `RuleBasedGenerator` is an offline
fallback that derives test cases directly from the acceptance criteria, so the
app still works without an API key.
"""

from __future__ import annotations

import re
from typing import Protocol

import anthropic

from .config import AISettings
from .models import (
    GeneratedTestCases,
    Priority,
    TestCase,
    TestStep,
    TestSuite,
    TestType,
    UserStory,
)

SYSTEM_PROMPT = """You are a senior QA engineer. Given a Jira user story, write \
manual test cases a tester can execute without further clarification.

Guidelines:
- Cover every acceptance criterion with at least one positive test, and add \
negative and edge-case tests where inputs, permissions, limits or states can vary.
- Add security, performance or accessibility tests only when the story makes \
them relevant.
- Steps are concrete user or system actions, each with an observable expected result.
- Use realistic, specific test data values.
- Number ids as <STORY-KEY>-TC01, <STORY-KEY>-TC02, ...
- If the story is ambiguous, state your assumptions in coverage_notes rather \
than inventing requirements."""


class GenerationError(RuntimeError):
    pass


class TestCaseGenerator(Protocol):
    name: str

    def generate(self, story: UserStory, max_cases: int = 10) -> TestSuite: ...


class ClaudeTestCaseGenerator:
    def __init__(
        self,
        settings: AISettings | None = None,
        client: anthropic.Anthropic | None = None,
        api_key: str | None = None,
    ):
        self.settings = settings or AISettings.from_env()
        self.client = client or anthropic.Anthropic(api_key=api_key or None)
        self.name = f"Claude ({self.settings.model})"

    def generate(self, story: UserStory, max_cases: int = 10) -> TestSuite:
        prompt = (
            f"Write up to {max_cases} test cases for this user story.\n\n"
            f"<user_story>\n{story.as_prompt_text()}\n</user_story>"
        )
        try:
            response = self.client.beta.messages.parse(
                model=self.settings.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self.settings.effort},
                output_format=GeneratedTestCases,
                # Re-run on a fallback model if the request is declined by a safety classifier.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as exc:
            raise GenerationError("Anthropic API key is missing or invalid.") from exc
        except anthropic.RateLimitError as exc:
            raise GenerationError("Rate limited by the Anthropic API, try again shortly.") from exc
        except anthropic.APIStatusError as exc:
            raise GenerationError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise GenerationError("Could not reach the Anthropic API.") from exc

        if response.stop_reason == "refusal":
            raise GenerationError("The model declined to generate test cases for this story.")
        if response.stop_reason == "max_tokens":
            raise GenerationError("Output was cut off; lower the number of test cases.")
        parsed = response.parsed_output
        if parsed is None:
            raise GenerationError("The model returned no parsable test cases.")

        return TestSuite(
            story=story,
            test_cases=parsed.test_cases[:max_cases],
            coverage_notes=parsed.coverage_notes,
            generator=self.name,
        )


_AC_HEADING = re.compile(r"^\s*(?:#+\s*)?acceptance\s+criteria\s*:?\s*$", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)]|AC\s*\d+\s*[:.-])\s*", re.I)
_GWT = re.compile(r"^\s*(given|when|then|and|but)\b\s*(.*)$", re.I)
_NEGATIVE_HINTS = re.compile(
    r"\b(invalid|error|fail|not|cannot|can't|reject|denied|unauthori[sz]ed|wrong|missing|empty)\b",
    re.I,
)
_LIMIT_HINTS = re.compile(r"\b(\d+|max(?:imum)?|min(?:imum)?|limit|at least|at most|up to)\b", re.I)


def extract_acceptance_criteria(story: UserStory) -> list[str]:
    """Split acceptance criteria into individual criteria.

    Uses the dedicated field when present, otherwise looks for an
    "Acceptance Criteria" section in the description. Given/When/Then blocks
    are kept together as one criterion.
    """
    text = story.acceptance_criteria
    if not text.strip():
        lines = story.description.splitlines()
        start = next((i for i, l in enumerate(lines) if _AC_HEADING.match(l)), None)
        text = "\n".join(lines[start + 1 :]) if start is not None else ""

    criteria: list[str] = []
    current_gwt: list[str] = []
    for raw in text.splitlines():
        line = _BULLET.sub("", raw).strip()
        if not line:
            continue
        gwt = _GWT.match(line)
        if gwt:
            keyword = gwt.group(1).lower()
            if keyword == "given" and current_gwt:
                criteria.append(" ".join(current_gwt))
                current_gwt = []
            current_gwt.append(line)
            continue
        if current_gwt:
            criteria.append(" ".join(current_gwt))
            current_gwt = []
        criteria.append(line)
    if current_gwt:
        criteria.append(" ".join(current_gwt))
    return criteria


def _split_gwt(criterion: str) -> tuple[list[str], list[str], list[str]]:
    given: list[str] = []
    when: list[str] = []
    then: list[str] = []
    bucket = None
    for part in re.split(r"\s+(?=(?:Given|When|Then|And|But)\b)", criterion):
        m = _GWT.match(part)
        if not m:
            continue
        keyword, body = m.group(1).lower(), m.group(2).strip()
        if keyword in ("given", "when", "then"):
            bucket = {"given": given, "when": when, "then": then}[keyword]
        if bucket is not None and body:
            bucket.append(body)
    return given, when, then


class RuleBasedGenerator:
    """Deterministic, offline generator driven by the acceptance criteria."""

    name = "Rule-based (offline)"

    def generate(self, story: UserStory, max_cases: int = 10) -> TestSuite:
        criteria = extract_acceptance_criteria(story)
        cases: list[TestCase] = []

        def add(**kwargs) -> None:
            if len(cases) < max_cases:
                cases.append(TestCase(id=f"{story.key}-TC{len(cases) + 1:02d}", **kwargs))

        if not criteria:
            criteria = [story.summary]

        for criterion in criteria:
            given, when, then = _split_gwt(criterion)
            if when or then:
                steps = [
                    TestStep(action=w, expected_result=then[0] if i == len(when) - 1 and then else "Action is accepted")
                    for i, w in enumerate(when)
                ] or [TestStep(action="Perform the described action", expected_result=then[0])]
                expected = "; ".join(then) or "Behaviour matches the acceptance criterion"
            else:
                steps = [
                    TestStep(action=f"Set up the scenario: {criterion}", expected_result="Scenario is ready"),
                    TestStep(action="Execute the behaviour described", expected_result=criterion),
                ]
                expected = criterion

            is_negative = bool(_NEGATIVE_HINTS.search(criterion))
            short = (then[0] if then else criterion)[:80]
            add(
                title=f"Verify: {short}",
                type=TestType.NEGATIVE if is_negative else TestType.POSITIVE,
                priority=Priority.HIGH,
                preconditions=given or ["User has access to the feature"],
                test_data=["Valid data as described in the story"],
                steps=steps,
                expected_result=expected,
                acceptance_criterion=criterion,
            )

            if not is_negative:
                add(
                    title=f"Invalid input or state rejected: {short}",
                    type=TestType.NEGATIVE,
                    priority=Priority.MEDIUM,
                    preconditions=given or ["User has access to the feature"],
                    test_data=["Invalid, missing or malformed data"],
                    steps=[
                        TestStep(
                            action="Repeat the scenario with invalid or missing input",
                            expected_result="The system blocks the action",
                        ),
                        TestStep(
                            action="Observe the feedback shown",
                            expected_result="A clear validation or error message is shown and no data is changed",
                        ),
                    ],
                    expected_result="The system rejects the request gracefully",
                    acceptance_criterion=criterion,
                )

            if _LIMIT_HINTS.search(criterion):
                add(
                    title=f"Boundary values: {short}",
                    type=TestType.EDGE,
                    priority=Priority.MEDIUM,
                    preconditions=given or ["User has access to the feature"],
                    test_data=["Value exactly at the limit", "Value one below the limit", "Value one above the limit"],
                    steps=[
                        TestStep(action="Use a value exactly at the limit", expected_result="Accepted"),
                        TestStep(action="Use a value just inside the limit", expected_result="Accepted"),
                        TestStep(action="Use a value just outside the limit", expected_result="Rejected with a clear message"),
                    ],
                    expected_result="Limits are enforced exactly as specified",
                    acceptance_criterion=criterion,
                )

        notes = (
            f"Derived {len(cases)} test case(s) from {len(criteria)} acceptance criterion/criteria "
            "using rules. Configure an Anthropic API key for richer, AI-generated test cases."
        )
        return TestSuite(story=story, test_cases=cases, coverage_notes=notes, generator=self.name)
