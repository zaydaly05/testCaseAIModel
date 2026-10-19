"""Turn generated test cases into runnable Python code (pytest).

Each test case becomes a pytest function: preconditions and steps become
method calls on an `app` object, test data becomes real Python values, and
expected results become `assert` statements. The full file also contains an
`App` class with one stub per action/check, which the team implements once
for their system (e.g. with Playwright for a web UI or requests for an API).
"""

from __future__ import annotations

import keyword
import re
import unicodedata
from dataclasses import dataclass, field

from .models import TestCase, TestSuite

_SKIP_WORDS = {"the", "a", "an", "is", "are", "be", "of", "to"}
_MAX_WORDS = 7


def identifier(text: str, max_words: int = _MAX_WORDS, prefix: str = "step") -> str:
    """'Enter email: jane@ and submit' -> 'enter_email_and_submit'."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    words = [w for w in re.findall(r"[a-z0-9]+", ascii_text) if w not in _SKIP_WORDS][:max_words]
    name = "_".join(words) or prefix
    if name[0].isdigit():
        name = f"{prefix}_{name}"
    return f"{name}_" if keyword.iskeyword(name) else name


def literal(value: str) -> str:
    """A test-data description -> a Python expression for that value."""
    value = value.strip()
    m = re.match(r"^(\d+) characters\b", value)
    if m:  # "12 characters ('Abcdefghij12')" -> a 12-character string
        return f'"a" * {int(m.group(1))}'
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)(?:\s+[A-Za-z %]+)?", value)
    if m:  # "49 distinct products", "1000" -> the number
        return m.group(1)
    if value in ("(empty)", ""):
        return '""'
    if value.startswith("'") and value.endswith("'") and len(value) >= 2:
        return repr(value[1:-1])
    return repr(value)


@dataclass
class _Method:
    name: str
    takes_value: bool = False
    meaning: set[str] = field(default_factory=set)


class _Registry:
    """Collects every App method the tests call, so the App class can be written."""

    def __init__(self):
        self.methods: dict[str, _Method] = {}

    def use(self, name: str, meaning: str, takes_value: bool = False) -> str:
        m = self.methods.setdefault(name, _Method(name))
        m.takes_value = m.takes_value or takes_value
        m.meaning.add(meaning)
        return name


def _call(text: str, test_data: list[str], reg: _Registry) -> str:
    """One step -> `app.method(value)`; the value is taken from the test data when it appears in the step."""
    hits = [d for d in test_data if d and d in text]
    if hits:
        value = max(hits, key=len)
        rest = text.replace(value, " ")
        name = reg.use(identifier(rest, prefix="do"), text, takes_value=True)
        return f"app.{name}({literal(value)})"
    return f"app.{reg.use(identifier(text, prefix='do'), text)}()"


def _check(text: str, reg: _Registry) -> str:
    return f"assert app.{reg.use(identifier(text, prefix='check'), text)}()"


def test_function(tc: TestCase, reg: _Registry | None = None) -> str:
    reg = reg or _Registry()
    name = f"test_{identifier(tc.id, max_words=4, prefix='tc')}_{identifier(tc.title, max_words=6, prefix='case')}"
    covers = " ".join(tc.acceptance_criterion.split())[:100]
    lines = [f"def {name}(app):",
             f"    # {tc.id} [{tc.type.value}, {tc.priority.value}] covers: {covers}"]
    if tc.preconditions:
        lines.append("    # arrange")
        lines += [f"    {_call(p, tc.test_data, reg)}" for p in tc.preconditions]
    lines.append("    # act and assert")
    last_check = None
    for step in tc.steps:
        lines.append(f"    {_call(step.action, tc.test_data, reg)}")
        last_check = _check(step.expected_result, reg)
        lines.append(f"    {last_check}")
    final = _check(tc.expected_result, reg)
    if final != last_check:
        lines.append(f"    {final}")
    return "\n".join(lines) + "\n"


def _app_class(reg: _Registry) -> str:
    out = ["class App:",
           '    """Connect the tests to your system: implement each method (Playwright, requests, ...).',
           "",
           "    Actions perform a step; checks return True when the expected result is observed.",
           '    """',
           ""]
    for m in sorted(reg.methods.values(), key=lambda m: m.name):
        args = "self, value" if m.takes_value else "self"
        meaning = " ".join(sorted(m.meaning)[0].split())[:110]
        out.append(f"    def {m.name}({args}):")
        out.append(f"        # {meaning}")
        out.append(f'        raise NotImplementedError("App.{m.name}")')
        out.append("")
    return "\n".join(out)


def to_pytest(suites: list[TestSuite]) -> str:
    """One runnable pytest file for all suites: the App adapter, an `app` fixture and every test."""
    reg = _Registry()
    tests = []
    for suite in suites:
        tests.append(f"# ----- {suite.story.key}: {suite.story.summary[:90]}\n")
        tests += [test_function(tc, reg) + "\n" for tc in suite.test_cases]
    header = ('"""Generated test cases. Run with:  pytest test_generated.py\n\n'
              'Implement the methods of App below once for your system; every test then runs against it.\n"""\n\n'
              "import pytest\n\n\n")
    fixture = "\n\n@pytest.fixture\ndef app():\n    return App()\n\n\n"
    return header + _app_class(reg) + fixture + "\n".join(tests)
