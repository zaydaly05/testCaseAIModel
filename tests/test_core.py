import json
from pathlib import Path

import pytest
import responses

from testcase_ai import JiraClient, TestCaseModel, UserStory
from testcase_ai.config import JiraSettings
from testcase_ai.exporters import to_csv, to_gherkin, to_json
from testcase_ai.jira_client import adf_to_text
from testcase_ai.model.datasets import load_any, load_training_data, map_label
from testcase_ai.model.extractor import analyze_criterion, extract_acceptance_criteria, extract_limits
from testcase_ai.model.taxonomy import CATEGORIES

ROOT = Path(__file__).parent.parent
SAMPLES = json.loads((ROOT / "samples" / "sample_stories.json").read_text())
REAL = json.loads((ROOT / "samples" / "real_stories.json").read_text())
BASE = "https://example.atlassian.net"


def story(i=0):
    return UserStory(**SAMPLES[i])


@pytest.fixture(scope="session")
def model():
    return TestCaseModel.train(feedback=None, extra_data=[], evaluate=False)


# ------------------------------------------------------------------ extraction

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


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The password must be at least 12 characters long", "≥ 12 characters"),
        ("A cart can contain at most 50 distinct products", "≤ 50 distinct products"),
        ("The quantity must be between 1 and 99", "1–99"),
        ("The page loads in under 2 seconds", "≤ 2 seconds"),
        ("Orders above 200 qualify for free shipping", "> 200"),
    ],
)
def test_extract_limits(text, expected):
    assert extract_limits(text)[0].describe() == expected


def test_analyze_criterion_finds_fields_roles_states():
    facts = analyze_criterion("Only admins can change the order status from Pending to Shipped when the email is valid")
    assert "admin" in facts.roles
    assert facts.states[:2] == ["Pending", "Shipped"]
    assert "email" in facts.fields


# ------------------------------------------------------------------ datasets

def test_promise_labels_are_mapped():
    assert map_label("SE") == "security"
    assert map_label("LF") == "usability"
    assert map_label("boundary") == "boundary"
    assert map_label("NFR") is None


def test_training_data_covers_all_categories():
    data = load_training_data()
    assert set(data["category"]) == set(CATEGORIES)
    assert len(data) > 1000
    assert not data["text"].str.lower().duplicated().any()


def test_load_user_csv(tmp_path):
    f = tmp_path / "mine.csv"
    f.write_text("Requirement,Type\nThe system shall encrypt passwords,SE\nUsers can export reports,F\n")
    df = load_any(f)
    assert list(df["category"]) == ["security", "functional"]


# ------------------------------------------------------------------ model

@pytest.mark.parametrize(
    "text, category",
    [
        ("The password must be at least 12 characters long", "boundary"),
        ("A confirmation email is sent after the order is placed", "notification"),
        ("If the card is declined an error message is shown", "error_handling"),
        ("The page loads in under 2 seconds", "performance"),
        ("Only administrators can delete user accounts", "security"),
        ("The email address must be in a valid format", "validation"),
        ("The order status changes from Pending to Shipped when dispatched", "workflow"),
    ],
)
def test_classifier_predicts_categories(model, text, category):
    assert model.classifier.predict([text])[0].category == category


def test_generate_covers_every_criterion(model):
    s = story(1)
    suite = model.generate(s, max_cases=40)
    assert {tc.acceptance_criterion for tc in suite.test_cases} == set(extract_acceptance_criteria(s))
    assert suite.test_cases[0].id == "DEMO-102-TC01"
    assert len(suite.insights) == 5


def test_boundary_tests_use_extracted_values(model):
    s = UserStory(key="X-1", summary="Password rules", acceptance_criteria="The password must be at least 12 characters long")
    suite = model.generate(s, max_cases=10)
    data = " ".join(d for tc in suite.test_cases for d in tc.test_data)
    assert "11 characters" in data and "12 characters" in data


def test_max_cases_prioritises_breadth(model):
    suite = model.generate(story(1), max_cases=5)
    assert len(suite.test_cases) == 5
    assert len({tc.acceptance_criterion for tc in suite.test_cases}) == 5


def test_real_stories_fully_covered(model):
    for raw in REAL[:15]:
        s = UserStory(**raw)
        suite = model.generate(s, max_cases=10)
        assert len({tc.acceptance_criterion for tc in suite.test_cases}) == len(suite.insights)


def test_save_load_and_feedback(model, tmp_path):
    path = model.save(tmp_path / "m.joblib")
    loaded = TestCaseModel.load(path)
    assert loaded.classifier.predict(["The page loads in under 2 seconds"])[0].category == "performance"

    fb = tmp_path / "feedback.csv"
    TestCaseModel.add_feedback([("The banner is purple and friendly", "compliance")], fb)
    retrained = TestCaseModel.train(feedback=fb, extra_data=[], evaluate=False)
    assert retrained.metadata["n_samples"] == model.metadata["n_samples"] + 1
    assert "feedback.csv" in retrained.metadata["sources"]


def test_exporters_produce_output(model):
    suites = [model.generate(story(0))]
    assert to_csv(suites).splitlines()[0].startswith("Story,ID,Title,Type,Priority,Technique")
    assert json.loads(to_json(suites))[0]["story"]["key"] == "DEMO-101"
    assert "Feature:" in to_gherkin(suites) and "Scenario:" in to_gherkin(suites)


# ------------------------------------------------------------------ jira

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


@responses.activate
def test_jira_search_paginates_with_tokens():
    issue = lambda k: {"key": k, "fields": {"summary": k}}
    responses.get(f"{BASE}/rest/api/3/search/jql", json={"issues": [issue("A-1")], "nextPageToken": "t", "isLast": False})
    responses.get(f"{BASE}/rest/api/3/search/jql", json={"issues": [issue("A-2")], "isLast": True})
    stories = JiraClient(JiraSettings(BASE, "a@b.c", "tok")).search_stories("project = A")
    assert [s.key for s in stories] == ["A-1", "A-2"]
    assert "nextPageToken=t" in responses.calls[1].request.url
