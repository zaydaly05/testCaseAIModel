import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import responses

from testcase_ai import ClaudeTestCaseGenerator, GenerationError, JiraClient, RuleBasedGenerator, UserStory
from testcase_ai.config import AISettings, JiraSettings
from testcase_ai.exporters import to_csv, to_gherkin, to_json
from testcase_ai.generator import extract_acceptance_criteria
from testcase_ai.jira_client import adf_to_text
from testcase_ai.models import GeneratedTestCases

SAMPLES = json.loads((Path(__file__).parent.parent / "samples" / "sample_stories.json").read_text())
BASE = "https://example.atlassian.net"


def story(i=0):
    return UserStory(**SAMPLES[i])


def test_adf_to_text_flattens_paragraphs_and_lists():
    doc = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "Hello"}]},
            {"type": "bulletList", "content": [
                {"type": "listItem", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "one"}]}]},
                {"type": "listItem", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "two"}]}]},
            ]},
        ],
    }
    assert adf_to_text(doc) == "Hello\n- one\n- two\n"


def test_extract_criteria_groups_given_when_then():
    criteria = extract_acceptance_criteria(story(0))
    assert criteria[0].startswith("Given I am on the login page")
    assert "Then I receive a reset link" in criteria[0]
    assert criteria[1].startswith("Given I have a reset link")
    assert len(criteria) == 4


def test_extract_criteria_from_description_section():
    criteria = extract_acceptance_criteria(story(1))
    assert criteria[0] == 'Clicking "Add to cart" adds the product with quantity 1'
    assert len(criteria) == 5


def test_rule_based_generator_covers_every_criterion():
    s = story(1)
    suite = RuleBasedGenerator().generate(s, max_cases=25)
    covered = {tc.acceptance_criterion for tc in suite.test_cases}
    assert covered == set(extract_acceptance_criteria(s))
    assert suite.test_cases[0].id == "DEMO-102-TC01"
    assert any(tc.type.value == "edge" for tc in suite.test_cases)  # "at most 50"


def test_rule_based_generator_respects_max_cases():
    assert len(RuleBasedGenerator().generate(story(1), max_cases=3).test_cases) == 3


def test_exporters_produce_output():
    suites = [RuleBasedGenerator().generate(story(0))]
    assert to_csv(suites).splitlines()[0].startswith("Story,ID,Title")
    assert json.loads(to_json(suites))[0]["story"]["key"] == "DEMO-101"
    assert "Feature:" in to_gherkin(suites) and "Scenario:" in to_gherkin(suites)


@responses.activate
def test_jira_get_story_cloud():
    responses.get(
        f"{BASE}/rest/api/3/issue/PROJ-1",
        json={
            "key": "PROJ-1",
            "fields": {
                "summary": "Login",
                "description": {"type": "doc", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Desc"}]}]},
                "issuetype": {"name": "Story"},
                "priority": {"name": "High"},
                "labels": ["auth"],
                "components": [{"name": "Web"}],
                "customfield_100": "Given x When y Then z",
            },
        },
    )
    client = JiraClient(JiraSettings(BASE, "a@b.c", "tok", "3", "customfield_100"))
    s = client.get_story("PROJ-1")
    assert (s.summary, s.description, s.priority, s.components) == ("Login", "Desc", "High", ["Web"])
    assert s.acceptance_criteria == "Given x When y Then z"
    assert s.url == f"{BASE}/browse/PROJ-1"


@responses.activate
def test_jira_search_paginates_with_tokens():
    issue = lambda k: {"key": k, "fields": {"summary": k}}
    responses.get(f"{BASE}/rest/api/3/search/jql", json={"issues": [issue("A-1")], "nextPageToken": "t", "isLast": False})
    responses.get(f"{BASE}/rest/api/3/search/jql", json={"issues": [issue("A-2")], "isLast": True})
    stories = JiraClient(JiraSettings(BASE, "a@b.c", "tok")).search_stories("project = A")
    assert [s.key for s in stories] == ["A-1", "A-2"]
    assert "nextPageToken=t" in responses.calls[1].request.url


def test_claude_generator_returns_parsed_suite():
    parsed = GeneratedTestCases.model_validate(
        {
            "coverage_notes": "ok",
            "test_cases": [
                {
                    "id": "DEMO-101-TC01", "title": "Reset link sent", "type": "positive", "priority": "high",
                    "preconditions": ["User exists"], "test_data": ["user@example.com"],
                    "steps": [{"action": "Submit email", "expected_result": "Confirmation shown"}],
                    "expected_result": "Email received", "acceptance_criterion": "AC1",
                }
            ],
        }
    )
    client = MagicMock()
    client.beta.messages.parse.return_value = SimpleNamespace(stop_reason="end_turn", parsed_output=parsed)
    suite = ClaudeTestCaseGenerator(AISettings(), client=client).generate(story(0))
    assert suite.test_cases[0].title == "Reset link sent"
    kwargs = client.beta.messages.parse.call_args.kwargs
    assert kwargs["output_format"] is GeneratedTestCases
    assert "DEMO-101" in kwargs["messages"][0]["content"]


def test_claude_generator_raises_on_refusal():
    client = MagicMock()
    client.beta.messages.parse.return_value = SimpleNamespace(stop_reason="refusal", parsed_output=None)
    with pytest.raises(GenerationError):
        ClaudeTestCaseGenerator(AISettings(), client=client).generate(story(0))
