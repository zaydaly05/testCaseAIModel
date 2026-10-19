"""Reading the logic of a criterion, and the test cases derived from it."""

import pytest

from testcase_ai import TestCaseModel, UserStory
from testcase_ai.model.behaviour import analyze_behaviour, flip_condition, negate


@pytest.fixture(scope="module")
def model():
    return TestCaseModel.load_portable()


def cases(model, criteria: str, summary: str = "Leave a community review"):
    return model.generate(UserStory(key="T-1", summary=summary, acceptance_criteria=criteria), 40).test_cases


def steps(tc):
    return [(s.action, s.expected_result) for s in tc.steps]


# ------------------------------------------------------------------ reading the behaviour

def test_given_then_with_an_endpoint_and_a_second_scenario():
    b = analyze_behaviour("Given GET /api/logs as admin, then logs are returned; without admin, 401.", "Audit logs")
    first, second = b.scenarios
    assert first.request == ("GET", "/api/logs") and first.outcomes == ["logs are returned"]
    assert second.conditions == ["without admin"] and second.action == "GET /api/logs"  # "as admin" dropped
    assert second.outcomes == ["the response status is 401"] and second.negative


def test_capability_with_listed_options():
    b = analyze_behaviour("Users must be able to filter search results by date, source, or relevance.", "Search news")
    assert b.main.capability and b.main.action == "filter search results by date, source, or relevance"
    assert b.options == ["date", "source", "relevance"]


def test_clamp_and_default_are_read():
    b = analyze_behaviour("Given a rating outside 1-5, then it is clamped to the range; a missing rating defaults to 5.",
                          "Leave a review")
    assert b.clamp and b.default == ("rating", "5")
    assert b.main.action == "Leave a review with a rating outside 1-5"


@pytest.mark.parametrize("condition, opposite", [
    ("a valid key", "an invalid key"),
    ("the database is down", "the database is up"),
    ("the build key is missing or wrong", "the build key is present and correct"),
    ("I am admin", "I am not admin"),
    ("a pending change", "no pending change"),
    ("any field is empty", "every field is filled"),
    ("an unknown or already-resolved code", None),  # no clean opposite
    ("I am on the hero", None),  # a place, not a state
])
def test_flip_condition(condition, opposite):
    assert flip_condition(condition) == opposite


@pytest.mark.parametrize("clause, negated", [
    ("the entry is stored", "the entry is not stored"),
    ("the review is saved and appears in the list", "the review is not saved and does not appear in the list"),
    ("no automatic reply is sent to me", "an automatic reply is sent to me"),
    ("Out-of-stock products cannot be added", "Out-of-stock products can be added"),
    ("GET /api/star returns the current count", "GET /api/star does not return the current count"),
    ("I see an error", "I do not see an error"),
])
def test_negate(clause, negated):
    assert negate(clause) == negated


# ------------------------------------------------------------------ the test cases

def test_endpoint_story_checks_status_codes_and_both_sides(model):
    tcs = cases(model, "- Given GET /api/logs as admin, then logs are returned; without admin, 401.\n"
                       "- Given DELETE /api/logs as admin, then logs are cleared.", "Record and view audit logs")
    by_title = {tc.title: tc for tc in tcs}
    assert steps(by_title["GET /api/logs: logs are returned"])[0] == ("Send GET /api/logs as admin", "Response status is 2xx")
    denied = by_title["Without admin: GET /api/logs: returns 401"]
    assert denied.preconditions == ["Signed in as a user who is not an admin"]
    assert steps(denied) == [("Send GET /api/logs", "Response status is 401")]
    assert any(steps(tc)[0] == ("Send DELETE /api/logs without credentials", "Response status is 401") for tc in tcs)
    cleared = next(tc for tc in tcs if tc.technique == "Read-back check")
    assert steps(cleared)[-1] == ("Inspect the response", "The list is empty")  # clearing a collection, not a 404


def test_rules_are_tested_not_templated(model):
    tcs = cases(model, "- Given a rating outside 1-5, then it is clamped to the range; a missing rating defaults to 5.\n"
                       "- Given the database is down, then the review is stored in the local JSON fallback.")
    data = {d: tc.title for tc in tcs for d in tc.test_data}
    clamped = next(tc for tc in tcs if tc.title.startswith("Values outside the range are clamped"))
    assert steps(clamped) == [("Use 0", "Stored as 1"), ("Use 6", "Stored as 5")]
    assert any(tc.title == "Missing rating defaults to 5" for tc in tcs)
    up = next(tc for tc in tcs if tc.preconditions == ["The database is up"])
    assert "works normally" in up.expected_result  # the other side of a failure is the normal path
    assert "Break a precondition" not in str([steps(tc) for tc in tcs]) and data


def test_each_listed_option_is_tested(model):
    tcs = cases(model, "- Users must be able to filter search results by date, source, or relevance.", "Search news")
    titles = {tc.title for tc in tcs}
    assert {"Filter search results by date", "Filter search results by source", "Filter search results by relevance"} <= titles
    assert "Nothing matches: an empty state is shown, not an error" in titles


def test_negative_outcome_is_not_turned_into_a_message_test(model):
    tcs = cases(model, "- Given I reply STOP, unsubscribe or an Arabic equivalent, then no automatic reply is sent to me.",
                "Respect STOP requests")
    assert all("arrives exactly once" not in tc.title for tc in tcs)
    assert {"Reply STOP", "Reply unsubscribe", "Reply an Arabic equivalent"} <= {tc.title for tc in tcs}


def test_no_invented_format_rules(model):
    tcs = cases(model, "- Given name and comment are provided, when I submit, then the review is saved and appears in the list.")
    assert not any("Invalid name" in tc.title for tc in tcs)
    other_side = next(tc for tc in tcs if tc.technique == "Decision table (condition reversed)")
    assert other_side.preconditions == ["Name and comment are missing"]
    assert steps(other_side) == [("Submit", "The review is not saved and does not appear in the list")]


def test_email_input_is_tested_from_the_other_side(model):
    tcs = cases(model, '- Given I am on the login page When I click "Forgot password" and submit my registered email '
                       "Then I receive a reset link by email within 5 minutes", "Reset my password")
    by_title = {tc.title: tc for tc in tcs}
    unregistered = by_title["An unregistered email: I do not receive a reset link by email"]
    assert steps(unregistered)[-1][0].endswith("submit my unregistered email")
    malformed = by_title["A malformed email address: I do not receive a reset link by email"]
    assert "jane.doe@" in steps(malformed)[-1][0]
    assert "The email contains a working reset link" in by_title
    assert not any("exactly once" in t for t in by_title)  # the story does not ask for it


def test_checks_only_what_the_story_asks_for(model):
    titles = {tc.title for tc in cases(model, "- The new password must be at least 12 characters long\n"
                                              "- Submitting an unregistered email shows the same confirmation message",
                                       "Reset my password")}
    assert not {"Invalid password formats are rejected", "Required input left empty is refused",
                "Malicious input is neutralised", "No sensitive information is disclosed"} & titles
    assert any("(≥ 12 characters)" in t for t in titles)  # the length rule is tested by its boundaries
    titles = {tc.title for tc in cases(model, "- Comments are sanitized so HTML or script input is not executed\n"
                                              "- The password must contain a digit")}
    assert "Malicious input is neutralised" in titles and "Invalid password formats are rejected" in titles


def test_only_roles_that_act_get_access_tests(model):
    tcs = cases(model, "- Given an opt-out, then the owner is told so the number can be removed from the broadcast list.",
                "Respect STOP requests")
    assert not any(tc.type.value == "security" for tc in tcs)
    tcs = cases(model, "- Only admins can delete users", "Manage users")
    assert any(tc.title == "Users who are not an admin cannot delete users" for tc in tcs)
