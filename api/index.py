"""HTTP API for the test case generator (Vercel Python function, also runnable locally).

Local:  uvicorn api.index:app --reload   then open http://127.0.0.1:8000
Vercel: every /api/* request is routed here (see vercel.json); the page in public/ calls it.

The model runs from the portable artifact, so this needs no scikit-learn.
"""

from __future__ import annotations

import json
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import PlainTextResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from testcase_ai.config import JiraSettings  # noqa: E402
from testcase_ai.exporters import to_csv, to_gherkin, to_json  # noqa: E402
from testcase_ai.jira_client import JiraClient, JiraError  # noqa: E402
from testcase_ai.model.engine import TestCaseModel  # noqa: E402
from testcase_ai.models import TestSuite, UserStory  # noqa: E402

SAMPLES = ROOT / "samples"
MAX_TEXT = 20_000  # characters per story field, keeps a single request cheap

app = FastAPI(title="Test Case Generator", version="1.0")


@lru_cache(maxsize=1)
def get_model() -> TestCaseModel:
    return TestCaseModel.load_portable()


class StoryIn(BaseModel):
    key: str = Field("STORY-1", max_length=64)
    summary: str = Field(..., min_length=1, max_length=1000)
    description: str = Field("", max_length=MAX_TEXT)
    acceptance_criteria: str = Field("", max_length=MAX_TEXT)
    priority: str | None = Field(None, max_length=32)


class GenerateIn(BaseModel):
    story: StoryIn
    max_cases: int = Field(12, ge=1, le=40)


class ExportIn(BaseModel):
    suites: list[TestSuite] = Field(..., min_length=1, max_length=50)
    format: Literal["csv", "json", "gherkin"] = "csv"


class JiraIn(BaseModel):
    keys: list[str] = Field(default_factory=list, max_length=50)
    jql: str | None = Field(None, max_length=2000)
    limit: int = Field(20, ge=1, le=100)


@app.get("/api/health")
def health() -> dict:
    meta = get_model().metadata
    return {
        "status": "ok",
        "model": {
            "trained_at": meta.get("trained_at"),
            "training_rows": meta.get("n_samples"),
            "gold_rows": meta.get("n_gold"),
            "evaluation": meta.get("evaluation"),
        },
        "jira_configured": JiraSettings.from_env().is_configured,
    }


@app.get("/api/samples")
def samples(count: int = 5) -> list[dict]:
    demo = json.loads((SAMPLES / "sample_stories.json").read_text(encoding="utf-8"))
    real = json.loads((SAMPLES / "drive_stories.json").read_text(encoding="utf-8"))
    return demo + random.sample(real, max(0, min(count, 20)))


@app.post("/api/generate")
def generate(body: GenerateIn) -> TestSuite:
    story = UserStory(**body.story.model_dump())
    return get_model().generate(story, body.max_cases)


@app.post("/api/export", response_class=PlainTextResponse)
def export(body: ExportIn) -> PlainTextResponse:
    writers = {"csv": (to_csv, "text/csv", "csv"), "json": (to_json, "application/json", "json"),
               "gherkin": (to_gherkin, "text/plain", "feature")}
    write, media, ext = writers[body.format]
    return PlainTextResponse(write(body.suites), media_type=media,
                             headers={"Content-Disposition": f'attachment; filename="test_cases.{ext}"'})


@app.post("/api/jira/stories")
def jira_stories(body: JiraIn) -> list[UserStory]:
    """Fetch stories with the Jira credentials stored in the deployment's environment variables."""
    settings = JiraSettings.from_env()
    if not settings.is_configured:
        raise HTTPException(503, "Jira is not configured on the server (set JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN).")
    if not body.keys and not body.jql:
        raise HTTPException(422, "Give issue keys or a JQL query.")
    try:
        client = JiraClient(settings)
        if body.keys:
            return [client.get_story(k) for k in body.keys]
        return client.search_stories(body.jql, body.limit)
    except JiraError as exc:
        raise HTTPException(502, str(exc)) from exc


# Local runs serve the web page too; on Vercel the page is served from public/ directly.
if (ROOT / "public").is_dir():
    app.mount("/", StaticFiles(directory=ROOT / "public", html=True), name="web")
