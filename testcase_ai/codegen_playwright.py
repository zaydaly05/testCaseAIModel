"""Turn generated test cases into Playwright web tests (pytest-playwright).

Each test case becomes a test that drives a real browser through the `page`
fixture of pytest-playwright. Steps the model recognises (sign in, open the
page, enter a value, submit, reload, keyboard navigation, mobile viewport,
response-time measurement ...) become Playwright calls, and expected results
become `expect(...)` checks. Story-specific steps that cannot be guessed
become small typed helper functions at the top of the file for the team to
fill in once.
"""

from __future__ import annotations

import re

from .codegen import _Registry, identifier, literal
from .models import TestCase, TestSuite

_HELPERS = {"open_page", "login", "submit", "fill", "expect_error", "expect_no_error", "expect_denied",
            "expect_signed_in", "expect_sign_in_page", "expect_loaded", "account"}
_ENVIRONMENT = re.compile(r"account (?:without .*)?exists|environment|performance tool|monitoring tool"
                          r"|inbox|device is available|reset link", re.I)
_LIMIT = re.compile(r"(?:≤|<=|<|within|under|at most)\s*(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|sec|seconds?)\b", re.I)


class _State:
    """What earlier lines of one test defined, so later lines only use what exists."""

    def __init__(self, page_key: str):
        self.page_key = page_key
        self.response = False
        self.timings = False
        self.percentile = ""


def _page(state: _State) -> str:
    return f"PAGES[{state.page_key!r}]"


def _value(text: str, test_data: list[str]) -> str | None:
    hits = [d for d in test_data if d and d in text]
    return literal(max(hits, key=len)) if hits else None


def _stub(text: str, reg: _Registry, is_check: bool, value: str | None = None) -> str:
    text = re.sub(r"^(?:Perform the behaviour|Trigger the event|Create the situation):\s*", "", text)
    name = identifier(text, prefix="step")
    name = f"expect_{name}" if is_check else name
    name = f"{name}_step" if name in _HELPERS else name
    name = f"step_{name}" if name.startswith("test") else name  # pytest would collect it as a test
    reg.use(name, text, takes_value=value is not None, is_check=is_check)
    return f"{name}(page, {value})" if value is not None else f"{name}(page)"


def _action(text: str, test_data: list[str], reg: _Registry, state: _State, roles: set[str]) -> list[str]:
    """One precondition or step -> Playwright lines."""
    t = text.strip()
    if m := re.match(r"^(?:I am )?sign(?:ed)? in as an? (.+?)(?: who is not an? (\w+))?$", t, re.I):
        role = f"not_{identifier(m.group(2))}" if m.group(2) else identifier(m.group(1), max_words=3, prefix="role")
        roles.add(role)
        return [f"login(page, {role!r})"]
    if _ENVIRONMENT.search(t):
        return [f"# needs: {t}"]
    if re.search(r"login page|sign-in page", t, re.I):
        return ["open_page(page, LOGIN_PATH)"]
    if re.search(r"sign out|clear cookies|log out", t, re.I):
        return ["page.context.clear_cookies()"]
    if re.match(r"^reload", t, re.I):
        return ["page.reload()"]
    if m := re.search(r"\b(\d{3,4})\s*px\b", t):
        return [f'page.set_viewport_size({{"width": {m.group(1)}, "height": 800}})', f"open_page(page, {_page(state)})"]
    if m := re.match(r"^measure .* (\d+) times$", t, re.I):
        state.timings = True
        return ["timings: list[float] = []",
                f"for _ in range({m.group(1)}):",
                "    start = time.perf_counter()",
                f"    open_page(page, {_page(state)})",
                "    timings.append(time.perf_counter() - start)"]
    if (m := re.search(r"(\d+)(?:st|nd|rd|th) percentile", t)) and state.timings:
        state.percentile = f"p{m.group(1)}"
        return [f"p{m.group(1)} = sorted(timings)[max(0, round(len(timings) * {int(m.group(1)) / 100}) - 1)]"]
    if re.search(r"(?:url|endpoint) directly", t, re.I) and re.match(r"^call", t, re.I):
        state.response = True
        return [f"response = page.request.get(BASE_URL + {_page(state)})"]
    if re.match(r"^(?:open|go to|navigate to|visit)\b", t, re.I):
        return [f"open_page(page, {_page(state)})"]
    if re.search(r"\btab\b.*\bonly\b|keyboard only", t, re.I):
        return ["for _ in range(10):", '    page.keyboard.press("Tab")']
    if re.match(r"^enter only spaces", t, re.I):
        return ['fill(page, "   ")', "submit(page)"]
    if re.match(r"^leave .*empty", t, re.I):
        return ['fill(page, "")']
    if m := re.match(r"^enter (?:the )?([A-Za-z][\w ]{0,30}?): (.+?)( and submit)?$", t, re.I):
        value = _value(m.group(2), test_data) or literal(m.group(2))
        return [f"fill(page, {value}, {m.group(1).strip()!r})"] + (["submit(page)"] if m.group(3) else [])
    if m := re.match(r"^(?:enter|type|use) (.+?)(?: and submit)?$", t, re.I):
        if value := _value(m.group(1), test_data):
            return [f"fill(page, {value})", "submit(page)"]
    if m := re.search(r'(?:click|press|tap|select)\b.*?"([^"]+)"', t, re.I):
        return [f'page.get_by_role("button", name={m.group(1)!r}).click()']
    if re.match(r"^(?:submit|perform the action$|retry\b|save\b|confirm\b)", t, re.I):
        return ["submit(page)"]
    return [_stub(t, reg, is_check=False, value=_value(t, test_data))]


def _expect(text: str, reg: _Registry, state: _State) -> list[str]:
    """One expected result -> Playwright checks."""
    t = text.strip()
    low = t.lower()
    if m := re.search(r'"([^"]+)"', t):
        return [f"expect(page.get_by_text({m.group(1)!r})).to_be_visible()"]
    if "horizontal scroll" in low:
        return ['assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")']
    if state.percentile and (limit := _LIMIT.search(t)):
        p = state.percentile
        seconds = float(limit.group(1)) / (1000 if limit.group(2).lower().startswith("m") else 1)
        return [f'assert {p} <= {seconds:g}, f"{p} was {{{p}:.2f}} s"']
    if state.response and re.search(r"reject|denied|no data", low):
        return ["assert response.status in (401, 403, 404)"]
    if re.search(r"redirected to sign-in|without authentication", low):
        return ["expect_sign_in_page(page)"]
    if "escaped" in low:
        return ['assert "<script>alert(" not in page.content()']
    if re.search(r"\bno (?:validation )?errors?\b|passes validation|^accepted|accepted and saved|access is granted"
                 r"|can perform the action|now succeeds|are accepted", low):
        return ["expect_no_error(page)"]
    if re.search(r"denied|forbidden|\b40[13]\b|without permission cannot", low):
        return ["expect_denied(page)"]
    if re.search(r"reject|refuse|blocked|error is shown|'required' message|treated as empty|not accepted|invalid", low):
        return ["expect_error(page)"]
    if re.match(r"^(?:signed in|session is restored)$", low):
        return ["expect_signed_in(page)"]
    if low == "session ended":
        return ["assert not page.context.cookies()"]
    if low == "screen loads":
        return ["expect_loaded(page)"]
    if "focus is visible" in low:
        return ['expect(page.locator(":focus")).to_be_visible()']
    if low == "field is empty":
        return ['expect(page.locator("input:visible, textarea:visible").first).to_have_value("")']
    if low == "timings recorded" and state.timings:
        return ["assert len(timings) > 0"]
    return [_stub(t, reg, is_check=True)]


def playwright_function(tc: TestCase, page_key: str, reg: _Registry | None = None,
                        roles: set[str] | None = None) -> str:
    """One test case -> one Playwright test function (uses the helpers of `to_playwright`)."""
    reg = reg if reg is not None else _Registry()
    roles = roles if roles is not None else set()
    state = _State(page_key)
    name = f"test_{identifier(tc.id, max_words=4, prefix='tc')}_{identifier(tc.title, max_words=6, prefix='case')}"
    covers = " ".join(tc.acceptance_criterion.split())[:100]
    body = [f"# {tc.id} [{tc.type.value}, {tc.priority.value}] covers: {covers}"]
    signed_in = False
    if tc.preconditions:
        body.append("# arrange")
        for p in tc.preconditions:
            lines = _action(p, tc.test_data, reg, state, roles)
            signed_in = signed_in or lines[0].startswith("login(")
            body += lines
    if not signed_in and not any(re.match(r"^(?:sign|open|go to|navigate)", s.action, re.I) for s in tc.steps):
        body.append(f"open_page(page, {_page(state)})")
    body.append("# act and assert")
    last: list[str] = []
    for step in tc.steps:
        body += _action(step.action, tc.test_data, reg, state, roles)
        last = _expect(step.expected_result, reg, state)
        body += last
    final = _expect(tc.expected_result, reg, state)
    if final != last:
        body += final
    return f"def {name}(page: Page) -> None:\n" + "".join(f"    {line}\n" for line in body)


_HEADER = '''"""Generated Playwright web tests.

Run with:
    pip install pytest pytest-playwright
    playwright install chromium
    APP_URL=https://your-app.example pytest test_generated_playwright.py --headed

Set BASE_URL, PAGES and USERS below for your site, then fill in the step helpers
(each one names the story step it stands for). Everything else runs as generated.
"""

from __future__ import annotations

import os
import re
{time}
from playwright.sync_api import Page, expect

BASE_URL = os.getenv("APP_URL", "http://localhost:3000").rstrip("/")
LOGIN_PATH = "/login"
SIGN_IN_URL = re.compile(r"log-?in|sign-?in", re.I)
ERROR = "[role=alert], .error, .invalid-feedback, [aria-invalid=true]"
'''

_HELPER_CODE = '''

def account(role: str) -> tuple[str, str]:
    email, password = USERS[role]
    prefix = role.upper()
    return os.getenv(f"{prefix}_EMAIL", email), os.getenv(f"{prefix}_PASSWORD", password)


def open_page(page: Page, path: str = "/") -> None:
    page.goto(BASE_URL + path)


def submit(page: Page) -> None:
    pattern = re.compile(r"submit|save|sign in|log in|continue|send|confirm|ok", re.I)
    page.get_by_role("button", name=pattern).first.click()


def login(page: Page, role: str) -> None:
    email, password = account(role)
    open_page(page, LOGIN_PATH)
    page.get_by_label(re.compile(r"e-?mail|user", re.I)).first.fill(email)
    page.get_by_label(re.compile(r"password", re.I)).first.fill(password)
    submit(page)
    expect_signed_in(page)


def fill(page: Page, value: object, field: str | None = None) -> None:
    if field:
        box = page.get_by_label(re.compile(re.escape(field), re.I)).first
    else:
        box = page.locator("input:visible, textarea:visible").first
    box.fill(str(value))


def expect_error(page: Page) -> None:
    expect(page.locator(ERROR).first).to_be_visible()


def expect_no_error(page: Page) -> None:
    expect(page.locator(ERROR)).to_have_count(0)


def expect_denied(page: Page) -> None:
    denied = re.compile(r"denied|forbidden|not authori[sz]ed|no permission|403", re.I)
    expect(page.get_by_text(denied).first).to_be_visible()


def expect_signed_in(page: Page) -> None:
    expect(page).not_to_have_url(SIGN_IN_URL)


def expect_sign_in_page(page: Page) -> None:
    expect(page).to_have_url(SIGN_IN_URL)


def expect_loaded(page: Page) -> None:
    expect(page.locator("body")).to_be_visible()
    expect_no_error(page)
'''


def _stubs(reg: _Registry) -> str:
    if not reg.methods:
        return ""
    out = ["", "", "# ----- Story-specific steps: implement each one with Playwright calls on `page`."]
    for m in sorted(reg.methods.values(), key=lambda m: m.name):
        args = "page: Page, value: object" if m.takes_value else "page: Page"
        meaning = " ".join(sorted(m.meaning)[0].split())[:110]
        out += ["", "", f"def {m.name}({args}) -> None:", f"    # {meaning}",
                f'    raise NotImplementedError("{m.name}")']
    return "\n".join(out) + "\n"


def to_playwright(suites: list[TestSuite]) -> str:
    """One Playwright test file for all suites: settings, helpers, step stubs and every test."""
    reg = _Registry()
    roles: set[str] = set()
    tests = []
    for suite in suites:
        tests.append(f"# ----- {suite.story.key}: {suite.story.summary[:90]}\n")
        tests += [playwright_function(tc, suite.story.key, reg, roles) + "\n" for tc in suite.test_cases]
    pages = "".join(f"    {s.story.key!r}: '/',\n" for s in suites)
    users = "".join(f"    {r!r}: ({r + '@example.com'!r}, 'change-me'),\n" for r in sorted(roles))
    header = _HEADER.replace("{time}", "import time\n" if any("time.perf_counter" in t for t in tests) else "")
    settings = ("\n# The page each story is tested on (path after BASE_URL).\n"
                f"PAGES: dict[str, str] = {{\n{pages}}}\n\n"
                "# One account per role; <ROLE>_EMAIL / <ROLE>_PASSWORD environment variables override them.\n"
                f"USERS: dict[str, tuple[str, str]] = {{\n{users}}}\n")
    return header + settings + _HELPER_CODE + _stubs(reg) + "\n\n" + "\n".join(tests)
