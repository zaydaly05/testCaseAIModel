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
from urllib.parse import urlencode, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request, Response  # noqa: E402
from fastapi.responses import PlainTextResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from testcase_ai.atlassian_auth import AtlassianAuth, JiraSession, OAuthSettings  # noqa: E402
from testcase_ai.config import JiraSettings  # noqa: E402
from testcase_ai.exporters import to_csv, to_gherkin, to_json, to_playwright, to_pytest  # noqa: E402
from testcase_ai.jira_client import JiraAuthError, JiraClient, JiraError  # noqa: E402
from testcase_ai.model.engine import TestCaseModel  # noqa: E402
from testcase_ai.models import TestSuite, UserStory  # noqa: E402

SAMPLES = ROOT / "samples"
MAX_TEXT = 20_000  # characters per story field, keeps a single request cheap
SESSION_COOKIE, REFRESH_COOKIE, STATE_COOKIE = "jira_session", "jira_refresh", "jira_oauth_state"
SESSION_DAYS = 30
DEFAULT_PORTS = {"http": "80", "https": "443"}

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
    format: Literal["pytest", "playwright", "csv", "json", "gherkin"] = "pytest"


class JiraIn(BaseModel):
    """Issue keys, or a JQL query, or neither to pull every story (see JiraSettings.all_stories_jql)."""

    keys: list[str] = Field(default_factory=list, max_length=50)
    jql: str | None = Field(None, max_length=2000)
    limit: int = Field(200, ge=1, le=500)


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
        "jira_sign_in": OAuthSettings.from_env().is_configured,
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
    writers = {"pytest": (to_pytest, "text/x-python", "test_generated.py"),
               "playwright": (to_playwright, "text/x-python", "test_generated_playwright.py"),
               "csv": (to_csv, "text/csv", "test_cases.csv"),
               "json": (to_json, "application/json", "test_cases.json"),
               "gherkin": (to_gherkin, "text/plain", "test_cases.feature")}
    write, media, filename = writers[body.format]
    return PlainTextResponse(write(body.suites), media_type=media,
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# ---------------------------------------------------------------- Sign in with Atlassian
# Each visitor signs in with their own Atlassian account. Their tokens live in encrypted,
# httpOnly cookies, so the stateless function needs no session store and the page never sees them.

def _auth() -> AtlassianAuth:
    settings = OAuthSettings.from_env()
    if not settings.is_configured:
        raise HTTPException(503, "Sign in with Atlassian is not set up on this server "
                                 "(set ATLASSIAN_CLIENT_ID and ATLASSIAN_CLIENT_SECRET).")
    return AtlassianAuth(settings)


def _origin(scheme: str, netloc: str) -> str:
    """Comparable origin: first value of a forwarded list, lower-case host, default port dropped."""
    scheme = scheme.split(",")[0].strip().lower()
    host = netloc.split(",")[0].strip().lower()
    port = DEFAULT_PORTS.get(scheme)
    if port and host.endswith(f":{port}"):
        host = host.removesuffix(f":{port}")
    return f"{scheme}://{host}"


def _request_origin(request: Request) -> str:
    """The origin the browser used, as seen through Vercel's proxy headers."""
    return _origin(request.headers.get("x-forwarded-proto") or request.url.scheme,
                   request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc)


def _is_https(request: Request) -> bool:
    return _request_origin(request).startswith("https://")


def _redirect_uri(request: Request) -> str:
    return OAuthSettings.from_env().redirect_uri or f"{_request_origin(request)}/api/auth/callback"


def _cookie(response: Response, request: Request, name: str, value: str, max_age: int, path: str = "/api") -> None:
    response.set_cookie(name, value, max_age=max_age, path=path, httponly=True, samesite="lax",
                        secure=_is_https(request))


def _store_session(response: Response, request: Request, auth: AtlassianAuth, session: JiraSession) -> None:
    main, refresh = auth.seal(session)
    _cookie(response, request, SESSION_COOKIE, main, SESSION_DAYS * 86400)
    if refresh:
        _cookie(response, request, REFRESH_COOKIE, refresh, SESSION_DAYS * 86400)


def _clear_session(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/api")
    response.delete_cookie(REFRESH_COOKIE, path="/api")


def _session(request: Request, response: Response, auth: AtlassianAuth) -> JiraSession:
    """The signed-in user's session, refreshing the access token (and its cookie) when it expired."""
    session = auth.unseal(request.cookies.get(SESSION_COOKIE), request.cookies.get(REFRESH_COOKIE))
    if session is None:
        raise HTTPException(401, "Sign in with your Atlassian account to use Jira.")
    if session.expired:
        try:
            session = auth.refresh(session)
        except JiraAuthError as exc:
            raise HTTPException(401, str(exc)) from exc
        except JiraError as exc:
            raise HTTPException(502, str(exc)) from exc
        _store_session(response, request, auth, session)
    return session


def _back_to_page(error: str = "") -> RedirectResponse:
    return RedirectResponse("/?" + urlencode({"jira_error": error}) if error else "/#jira", status_code=303)


@app.get("/api/auth/login")
def auth_login(request: Request, moved: bool = False) -> RedirectResponse:
    auth = _auth()
    redirect_uri = _redirect_uri(request)
    callback = urlparse(redirect_uri)
    canonical = _origin(callback.scheme, callback.netloc)
    # Opened on another address (e.g. a Vercel per-deployment URL): restart the sign-in on the address
    # Atlassian sends users back to, so the state cookie and the session cookies belong to one site.
    # `moved` allows a single hop, so a proxy that still reports another host cannot cause a redirect loop.
    if canonical != _request_origin(request) and not moved:
        return RedirectResponse(f"{canonical}/api/auth/login?moved=1", status_code=303)
    state = auth.make_state()
    response = RedirectResponse(auth.authorize_url(redirect_uri, state), status_code=303)
    _cookie(response, request, STATE_COOKIE, state, 600, path="/api/auth")
    return response


@app.get("/api/auth/callback")
def auth_callback(request: Request, code: str = "", state: str = "", error: str = "",
                  error_description: str = "") -> RedirectResponse:
    auth = _auth()
    if error:
        return _back_to_page(error_description or error)
    # The state must match the cookie that /login set in this browser, and be recent.
    if not code or state != request.cookies.get(STATE_COOKIE) or not auth.check_state(state):
        return _back_to_page("The sign-in link expired or did not start on this page. Try again.")
    try:
        session = auth.exchange_code(code, _redirect_uri(request))
    except JiraError as exc:
        return _back_to_page(str(exc))
    response = _back_to_page()
    response.delete_cookie(STATE_COOKIE, path="/api/auth")
    _store_session(response, request, auth, session)
    return response


@app.get("/api/auth/me")
def auth_me(request: Request, response: Response) -> dict:
    if not OAuthSettings.from_env().is_configured:
        return {"configured": False, "signed_in": False}
    try:
        session = _session(request, response, _auth())
    except HTTPException:
        return {"configured": True, "signed_in": False}
    return {"configured": True, "signed_in": True, **session.public()}


class SiteIn(BaseModel):
    site_id: str = Field(..., max_length=100)


@app.post("/api/auth/site")
def auth_site(body: SiteIn, request: Request, response: Response) -> dict:
    """Switch to another Jira site the signed-in account can access."""
    auth = _auth()
    session = _session(request, response, auth)
    if body.site_id not in {s.id for s in session.sites}:
        raise HTTPException(400, "Your account cannot access that Jira site.")
    session.site_id = body.site_id
    _store_session(response, request, auth, session)
    return {"configured": True, "signed_in": True, **session.public()}


@app.post("/api/auth/logout")
def auth_logout(response: Response) -> dict:
    _clear_session(response)
    return {"signed_in": False}


@app.post("/api/jira/stories")
def jira_stories(body: JiraIn, request: Request, response: Response) -> list[UserStory]:
    """Fetch stories from the signed-in user's own Jira site, with their own permissions."""
    settings = JiraSettings.from_env()
    session = _session(request, response, _auth())
    try:
        client = JiraClient(session.jira_settings(settings))
        if body.keys:
            return [client.get_story(k) for k in body.keys]
        return client.search_stories(body.jql or settings.all_stories_jql(), body.limit)
    except ValueError as exc:
        raise HTTPException(500, str(exc)) from exc
    except JiraAuthError as exc:  # token revoked, or the app's access removed on Atlassian's side
        raise HTTPException(401, "Jira no longer accepts your sign-in. Sign in again.") from exc
    except JiraError as exc:
        raise HTTPException(502, str(exc)) from exc


# Local runs serve the web page too; on Vercel the page is served from public/ directly.
if (ROOT / "public").is_dir():
    app.mount("/", StaticFiles(directory=ROOT / "public", html=True), name="web")
