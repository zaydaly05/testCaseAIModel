"""Export generated test suites as CSV, JSON or Gherkin."""

from __future__ import annotations

import csv
import io
import json

from .models import TestSuite

CSV_COLUMNS = [
    "Story",
    "ID",
    "Title",
    "Type",
    "Priority",
    "Acceptance Criterion",
    "Preconditions",
    "Test Data",
    "Steps",
    "Expected Result",
]


def to_csv(suites: list[TestSuite]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(CSV_COLUMNS)
    for suite in suites:
        for tc in suite.test_cases:
            steps = "\n".join(
                f"{i}. {s.action} -> {s.expected_result}" for i, s in enumerate(tc.steps, 1)
            )
            writer.writerow(
                [
                    suite.story.key,
                    tc.id,
                    tc.title,
                    tc.type.value,
                    tc.priority.value,
                    tc.acceptance_criterion,
                    "\n".join(tc.preconditions),
                    "\n".join(tc.test_data),
                    steps,
                    tc.expected_result,
                ]
            )
    return buf.getvalue()


def to_json(suites: list[TestSuite]) -> str:
    return json.dumps([s.model_dump(mode="json") for s in suites], indent=2)


def to_gherkin(suites: list[TestSuite]) -> str:
    out: list[str] = []
    for suite in suites:
        out.append(f"@{suite.story.key}")
        out.append(f"Feature: {suite.story.summary}")
        out.append("")
        for tc in suite.test_cases:
            out.append(f"  @{tc.id} @{tc.type.value} @{tc.priority.value}")
            out.append(f"  Scenario: {tc.title}")
            for i, pre in enumerate(tc.preconditions):
                out.append(f"    {'Given' if i == 0 else 'And'} {pre}")
            for i, step in enumerate(tc.steps):
                out.append(f"    {'When' if i == 0 else 'And'} {step.action}")
                out.append(f"    Then {step.expected_result}")
            out.append("")
    return "\n".join(out)
