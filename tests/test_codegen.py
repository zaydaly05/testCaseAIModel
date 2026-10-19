"""Generated test cases as pytest code."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from testcase_ai import TestCaseModel, UserStory
from testcase_ai.codegen import identifier, literal, to_pytest

ROOT = Path(__file__).parent.parent
REAL = json.loads((ROOT / "samples" / "drive_stories.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model():
    return TestCaseModel.load_portable()


@pytest.mark.parametrize("text, expected", [
    ("Enter email: jane@ and submit", "enter_email_jane_and_submit"),
    ('I click "Forgot password"', "i_click_forgot_password"),
    ("2 factor login", "step_2_factor_login"),
    ("", "step"),
    ("class", "class_"),
])
def test_identifier(text, expected):
    assert identifier(text) == expected


@pytest.mark.parametrize("value, expected", [
    ("12 characters ('Abcdefghij12')", '"a" * 12'),
    ("49 distinct products", "49"),
    ("1000", "1000"),
    ("(empty)", '""'),
    ("'   '", "'   '"),
    ("<script>alert(1)</script>", "'<script>alert(1)</script>'"),
])
def test_literal(value, expected):
    assert literal(value) == expected


def test_each_test_case_has_code(model):
    story = UserStory(key="SHOP-1", summary="Reset password",
                      acceptance_criteria="- The password must be at least 12 characters long")
    suite = model.generate(story, 10)
    for tc in suite.test_cases:
        assert tc.code.startswith("def test_shop_1_")
        compile(tc.code, tc.id, "exec")
    assert any('app.use("a" * 11)' in tc.code for tc in suite.test_cases)


def test_pytest_file_is_valid_python_for_real_stories(model):
    suites = [model.generate(UserStory(**s), 40) for s in REAL[::25]]
    code = to_pytest(suites)
    compile(code, "test_generated.py", "exec")
    names = [line for line in code.splitlines() if line.startswith("def test_")]
    assert len(names) == len(set(names)) == sum(len(s.test_cases) for s in suites)


def test_generated_file_runs_under_pytest(model, tmp_path):
    suite = model.generate(UserStory(key="SHOP-1", summary="Cart", acceptance_criteria="- A cart holds at most 50 items"), 5)
    (tmp_path / "test_generated.py").write_text(to_pytest([suite]), encoding="utf-8")
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
                         capture_output=True, text=True)
    # Every test is collected and fails only because the App adapter is not implemented yet.
    assert f"{len(suite.test_cases)} failed" in out.stdout
    assert "NotImplementedError" in out.stdout and "SyntaxError" not in out.stdout
