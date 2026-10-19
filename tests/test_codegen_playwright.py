"""Generated test cases as Playwright web tests."""

import importlib.util
import json
import os
import subprocess
import sys
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from testcase_ai import TestCaseModel, UserStory
from testcase_ai.codegen_playwright import to_playwright

ROOT = Path(__file__).parent.parent
REAL = json.loads((ROOT / "samples" / "drive_stories.json").read_text(encoding="utf-8"))
STORY = UserStory(key="SHOP-1", summary="Register", acceptance_criteria=(
    "- Email must be valid\n- On mobile (375px) there is no horizontal scrolling\n"
    "- The page should load within 2 seconds\n- Only admins can delete users"))


@pytest.fixture(scope="module")
def model():
    return TestCaseModel.load_portable()


def test_each_test_case_has_a_playwright_test(model):
    suite = model.generate(STORY, 40)
    for tc in suite.test_cases:
        assert tc.playwright.startswith("def test_shop_1_") and "(page: Page) -> None:" in tc.playwright
        compile(tc.playwright, tc.id, "exec")
    code = "\n".join(tc.playwright for tc in suite.test_cases)
    assert "fill(page, 'jane.doe@', 'email')" in code and "expect_error(page)" in code
    assert 'page.set_viewport_size({"width": 375, "height": 800})' in code
    assert "assert p95 <= 2" in code
    assert "login(page, 'admin')" in code and "login(page, 'not_admin')" in code


def test_playwright_file_is_valid_python_for_real_stories(model):
    suites = [model.generate(UserStory(**s), 40) for s in REAL[::25]]
    code = to_playwright(suites)
    compile(code, "test_generated_playwright.py", "exec")
    assert "from playwright.sync_api import Page, expect" in code
    names = [line for line in code.splitlines() if line.startswith("def test_")]
    assert len(names) == len(set(names)) == sum(len(s.test_cases) for s in suites)
    helpers = [line.split("(")[0] for line in code.splitlines() if line.startswith("def ") and not line.startswith("def test_")]
    assert len(helpers) == len(set(helpers))
    for s in suites:
        assert f"    {s.story.key!r}: '/'," in code


SITE = {
    "login/index.html": """<!doctype html><title>Sign in</title>
<form onsubmit="event.preventDefault(); location.href='/'">
<label>Email <input></label><label>Password <input type=password></label><button>Sign in</button></form>""",
    "index.html": """<!doctype html><title>Shop</title><meta name=viewport content="width=device-width">
<form id=f><label>Email <input id=email></label><button>Submit</button><div id=msg></div></form>
<script>f.onsubmit = (e) => { e.preventDefault();
  msg.innerHTML = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(email.value) ? '<p role=status>Saved</p>'
                                                                  : '<p role=alert>Enter a valid email</p>'; }</script>""",
}
CONFTEST = """import glob
import pytest

@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args):
    found = glob.glob("/opt/pw-browsers/chromium-*/chrome-linux*/chrome")
    return {**browser_type_launch_args, **({"executable_path": found[0]} if found else {})}
"""


@pytest.mark.skipif(importlib.util.find_spec("pytest_playwright") is None, reason="pytest-playwright not installed")
def test_generated_tests_run_in_a_browser(model, tmp_path):
    site = tmp_path / "site"
    for name, html in SITE.items():
        (site / name).parent.mkdir(parents=True, exist_ok=True)
        (site / name).write_text(html, encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    story = UserStory(key="SHOP-1", summary="Register", acceptance_criteria=(
        "- Email must be valid\n- On mobile (375px) there is no horizontal scrolling"))
    (tmp_path / "test_generated_playwright.py").write_text(to_playwright([model.generate(story, 40)]), encoding="utf-8")
    (tmp_path / "conftest.py").write_text(CONFTEST, encoding="utf-8")
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
                             capture_output=True, text=True, timeout=300, cwd=tmp_path,
                             env={**os.environ, "APP_URL": f"http://127.0.0.1:{server.server_port}"})
    finally:
        server.shutdown()
    if "Executable doesn't exist" in out.stdout:
        pytest.skip("no Playwright browser installed")
    # Valid email, invalid emails and the mobile layout run as generated; the rest stop at story-specific stubs.
    # (No empty-input test: the story does not say the email is required.)
    assert "3 passed" in out.stdout, out.stdout[-3000:]
    assert "SyntaxError" not in out.stdout and "NameError" not in out.stdout


class _ItemsApi(BaseHTTPRequestHandler):
    """A tiny JSON API: POST /api/items needs a JSON body (400 otherwise); /api/items/<id> only knows id 1."""

    def _send(self, status: int, body: object = None) -> None:
        data = json.dumps(body if body is not None else {}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            item = json.loads(raw)
        except ValueError:
            return self._send(400, {"error": "invalid JSON"})
        self._send(201, item)

    def do_GET(self) -> None:
        self._send(200 if self.path.rstrip("/").endswith(("/api/items", "/1")) else 404, [])

    def do_DELETE(self) -> None:
        self._send(200 if self.path.endswith("/1") else 404)

    def log_message(self, *args) -> None:
        pass


@pytest.mark.skipif(importlib.util.find_spec("pytest_playwright") is None, reason="pytest-playwright not installed")
def test_generated_api_tests_call_the_endpoints(model, tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ItemsApi)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    story = UserStory(key="API-1", summary="Manage items", acceptance_criteria=(
        "- Given POST /api/items, then the item is stored.\n- Given DELETE /api/items/:id, then the item is removed."))
    (tmp_path / "test_generated_playwright.py").write_text(to_playwright([model.generate(story, 40)]), encoding="utf-8")
    (tmp_path / "conftest.py").write_text(CONFTEST, encoding="utf-8")
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", str(tmp_path)],
                             capture_output=True, text=True, timeout=300, cwd=tmp_path,
                             env={**os.environ, "APP_URL": f"http://127.0.0.1:{server.server_port}", "COLUMNS": "300"})
    finally:
        server.shutdown()
    if "Executable doesn't exist" in out.stdout:
        pytest.skip("no Playwright browser installed")
    # The requests are sent and their status asserted (a malformed body gets 400, an unknown id 404); each test then
    # stops at the first check only the team can implement ("check the stored data").
    for name in ("tc03_post_api_items_with_malformed_body", "tc04_delete_api_items_id_with_unknown"):
        assert f"{name}[chromium] - NotImplementedError: check_stored_data" in out.stdout, out.stdout[-3000:]
    assert "SyntaxError" not in out.stdout and "NameError" not in out.stdout
