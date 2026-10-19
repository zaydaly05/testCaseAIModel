"""The deployed path: portable model + HTTP API (no scikit-learn at runtime)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from testcase_ai import TestCaseModel, UserStory
from testcase_ai.model.engine import PORTABLE_PATH
from testcase_ai.model.portable import PortableClassifier, export_portable

ROOT = Path(__file__).parent.parent
REAL = json.loads((ROOT / "samples" / "drive_stories.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def trained():
    return TestCaseModel.train(feedback=None, extra_data=[], evaluate=False)


def test_portable_matches_sklearn(trained):
    portable = PortableClassifier(json.loads(json.dumps(export_portable(trained.classifier))))
    texts = [line.lstrip("- ") for s in REAL[:40] for line in s["acceptance_criteria"].splitlines()]
    texts += ["The password must be at least 12 characters long", "x", ""]
    for a, b in zip(trained.classifier.predict(texts), portable.predict(texts)):
        assert a.category == b.category
        assert a.confidence == pytest.approx(b.confidence, abs=1e-5)


def test_bundled_artifact_loads_and_generates():
    model = TestCaseModel.load_portable()
    suite = model.generate(UserStory(key="A-1", summary="Reset password",
                                     acceptance_criteria="- The password must be at least 12 characters long"))
    assert suite.test_cases and suite.insights[0].category == "boundary"


def test_runtime_import_needs_no_heavy_libraries():
    code = ("import sys; sys.path.insert(0, '.'); import api.index; "
            "print(','.join(m for m in ('sklearn', 'pandas', 'numpy', 'scipy', 'joblib') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


@pytest.fixture(scope="module")
def client():
    sys.path.insert(0, str(ROOT))
    from api.index import app
    return TestClient(app)


def test_api_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["model"]["training_rows"] > 1000


def test_api_generate_and_export(client):
    story = {"key": "SHOP-1", "summary": "Reset password",
             "acceptance_criteria": "- The password must be at least 12 characters long\n- A reset link is emailed"}
    suite = client.post("/api/generate", json={"story": story, "max_cases": 6}).json()
    assert 2 <= len(suite["test_cases"]) <= 6 and len(suite["insights"]) == 2

    csv = client.post("/api/export", json={"suites": [suite], "format": "csv"})
    assert csv.status_code == 200 and csv.text.startswith("Story,ID,Title")
    feature = client.post("/api/export", json={"suites": [suite], "format": "gherkin"}).text
    assert "Feature: Reset password" in feature


def test_api_rejects_bad_input(client):
    assert client.post("/api/generate", json={"story": {"summary": ""}}).status_code == 422
    assert client.post("/api/generate", json={"story": {"summary": "x"}, "max_cases": 500}).status_code == 422


def test_api_jira_needs_server_config(client, monkeypatch):
    monkeypatch.delenv("JIRA_BASE_URL", raising=False)
    monkeypatch.delenv("JIRA_API_TOKEN", raising=False)
    assert client.post("/api/jira/stories", json={"keys": ["A-1"]}).status_code == 503


def test_api_samples(client):
    stories = client.get("/api/samples?count=3").json()
    assert len(stories) == 5 and all("summary" in s for s in stories)


def test_artifact_is_committed():
    assert PORTABLE_PATH.exists() and PORTABLE_PATH.stat().st_size < 10_000_000
