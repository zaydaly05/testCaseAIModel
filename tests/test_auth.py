"""Sign in with Atlassian: OAuth flow, encrypted session cookies and token refresh."""

import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
import responses
from fastapi.testclient import TestClient

from testcase_ai.atlassian_auth import (RESOURCES_URL, TOKEN_URL, AtlassianAuth, JiraSession, JiraSite,
                                        OAuthSettings)

ROOT = Path(__file__).parent.parent
SITES = [{"id": "cloud-1", "url": "https://one.atlassian.net", "name": "One", "scopes": ["read:jira-work"]},
         {"id": "cloud-2", "url": "https://two.atlassian.net", "name": "Two", "scopes": ["read:jira-work"]},
         {"id": "conf-only", "url": "https://wiki.atlassian.net", "name": "Wiki", "scopes": ["read:confluence-content.all"]}]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ATLASSIAN_CLIENT_ID", "cid")
    monkeypatch.setenv("ATLASSIAN_CLIENT_SECRET", "secret")
    monkeypatch.delenv("ATLASSIAN_REDIRECT_URI", raising=False)
    sys.path.insert(0, str(ROOT))
    from api.index import app
    return TestClient(app)


def auth() -> AtlassianAuth:
    return AtlassianAuth(OAuthSettings(client_id="cid", client_secret="secret"))


def mock_sign_in(rsps: responses.RequestsMock, access: str = "at-1", refresh: str = "rt-1") -> None:
    rsps.post(TOKEN_URL, json={"access_token": access, "refresh_token": refresh, "expires_in": 3600})
    rsps.get(RESOURCES_URL, json=SITES)
    rsps.get("https://api.atlassian.com/ex/jira/cloud-1/rest/api/3/myself", json={"displayName": "Ada Lovelace"})


def sign_in(client: TestClient) -> None:
    login = client.get("/api/auth/login", follow_redirects=False)
    state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]
    with responses.RequestsMock() as rsps:
        mock_sign_in(rsps)
        back = client.get(f"/api/auth/callback?code=abc&state={state}", follow_redirects=False)
    assert back.status_code == 303 and back.headers["location"] == "/#jira"


def test_login_redirects_to_atlassian(client):
    resp = client.get("/api/auth/login", follow_redirects=False)
    url = urlparse(resp.headers["location"])
    query = parse_qs(url.query)
    assert url.netloc == "auth.atlassian.com" and query["client_id"] == ["cid"]
    assert query["redirect_uri"] == ["http://testserver/api/auth/callback"]
    assert "read:jira-work" in query["scope"][0] and "offline_access" in query["scope"][0]
    assert resp.cookies["jira_oauth_state"].strip('"') == query["state"][0]  # quoted because it ends in "="


def test_login_moves_to_the_registered_address(client, monkeypatch):
    monkeypatch.setenv("ATLASSIAN_REDIRECT_URI", "https://app.example.com/api/auth/callback")
    other = client.get("/api/auth/login", follow_redirects=False, headers={"host": "app-git-abc123.vercel.app"})
    assert other.headers["location"] == "https://app.example.com/api/auth/login?moved=1" and not other.cookies
    same = client.get("/api/auth/login", follow_redirects=False,
                      headers={"host": "app.example.com", "x-forwarded-proto": "https"})
    query = parse_qs(urlparse(same.headers["location"]).query)
    assert query["redirect_uri"] == ["https://app.example.com/api/auth/callback"]


@pytest.mark.parametrize("configured, headers", [
    ("https://App.Example.com/api/auth/callback", {"host": "app.example.com"}),  # host case differs
    ("https://app.example.com:443/api/auth/callback", {"host": "app.example.com"}),  # explicit default port
    ("https://app.example.com/api/auth/callback", {"host": "internal", "x-forwarded-host": "app.example.com, lb"}),
])
def test_login_on_the_registered_address_goes_to_atlassian(client, monkeypatch, configured, headers):
    monkeypatch.setenv("ATLASSIAN_REDIRECT_URI", configured)
    resp = client.get("/api/auth/login", follow_redirects=False, headers={"x-forwarded-proto": "https", **headers})
    assert urlparse(resp.headers["location"]).netloc == "auth.atlassian.com"


def test_login_moves_at_most_once(client, monkeypatch):
    """A proxy that keeps reporting another host must not cause a redirect loop."""
    monkeypatch.setenv("ATLASSIAN_REDIRECT_URI", "https://app.example.com/api/auth/callback")
    first = client.get("/api/auth/login", follow_redirects=False, headers={"host": "odd-proxy"})
    assert first.headers["location"] == "https://app.example.com/api/auth/login?moved=1"
    second = client.get("/api/auth/login?moved=1", follow_redirects=False, headers={"host": "odd-proxy"})
    assert urlparse(second.headers["location"]).netloc == "auth.atlassian.com"


def test_login_over_http_moves_to_the_https_address(client, monkeypatch):
    monkeypatch.setenv("ATLASSIAN_REDIRECT_URI", "https://app.example.com/api/auth/callback")
    resp = client.get("/api/auth/login", follow_redirects=False, headers={"host": "app.example.com"})
    assert resp.headers["location"] == "https://app.example.com/api/auth/login?moved=1"


def test_revoked_refresh_token_signs_the_user_out(client):
    session = JiraSession(access_token="old", refresh_token="revoked", expires_at=time.time() - 10, site_id="c",
                          sites=[JiraSite("c", "https://one.atlassian.net", "One")])
    main, refresh = auth().seal(session)
    client.cookies.set("jira_session", main, path="/api")
    client.cookies.set("jira_refresh", refresh, path="/api")
    with responses.RequestsMock() as rsps:
        rsps.post(TOKEN_URL, status=403, json={"error": "invalid_grant"})
        assert client.post("/api/jira/stories", json={"keys": ["ONE-1"]}).status_code == 401


def test_full_sign_in_and_fetch(client):
    sign_in(client)
    me = client.get("/api/auth/me").json()
    assert me["signed_in"] and me["user"] == "Ada Lovelace" and me["site"]["id"] == "cloud-1"
    assert [s["id"] for s in me["sites"]] == ["cloud-1", "cloud-2"]  # sites without Jira access are skipped
    assert "at-1" not in str(me) and "rt-1" not in str(me)  # tokens never reach the page

    switched = client.post("/api/auth/site", json={"site_id": "cloud-2"}).json()
    assert switched["site"]["url"] == "https://two.atlassian.net"
    assert client.post("/api/auth/site", json={"site_id": "someone-elses"}).status_code == 400

    with responses.RequestsMock() as rsps:
        rsps.get("https://api.atlassian.com/ex/jira/cloud-2/rest/api/3/issue/TWO-1",
                 json={"key": "TWO-1", "fields": {"summary": "Checkout"}})
        stories = client.post("/api/jira/stories", json={"keys": ["TWO-1"]}).json()
        assert rsps.calls[0].request.headers["Authorization"] == "Bearer at-1"
    assert stories[0]["url"] == "https://two.atlassian.net/browse/TWO-1"

    client.post("/api/auth/logout", json={})
    assert client.get("/api/auth/me").json()["signed_in"] is False


def test_callback_rejects_state_from_another_browser(client):
    client.get("/api/auth/login", follow_redirects=False)
    forged = auth().make_state()  # valid, but not the one this browser was given
    back = client.get(f"/api/auth/callback?code=abc&state={forged}", follow_redirects=False)
    assert "jira_error=" in back.headers["location"]
    assert client.get("/api/auth/me").json()["signed_in"] is False


def test_callback_reports_denied_consent(client):
    back = client.get("/api/auth/callback?error=access_denied&error_description=User+said+no", follow_redirects=False)
    assert back.headers["location"] == "/?jira_error=User+said+no"


def test_expired_token_is_refreshed(client):
    session = JiraSession(access_token="old", refresh_token="rt-old", expires_at=time.time() - 10, site_id="cloud-1",
                          sites=[JiraSite("cloud-1", "https://one.atlassian.net", "One")])
    main, refresh = auth().seal(session)
    client.cookies.set("jira_session", main, path="/api")
    client.cookies.set("jira_refresh", refresh, path="/api")
    with responses.RequestsMock() as rsps:
        rsps.post(TOKEN_URL, json={"access_token": "new", "refresh_token": "rt-new", "expires_in": 3600})
        rsps.get("https://api.atlassian.com/ex/jira/cloud-1/rest/api/3/issue/ONE-1",
                 json={"key": "ONE-1", "fields": {"summary": "x"}})
        resp = client.post("/api/jira/stories", json={"keys": ["ONE-1"]})
        assert b'"refresh_token": "rt-old"' in rsps.calls[0].request.body
        assert rsps.calls[1].request.headers["Authorization"] == "Bearer new"
    renewed = auth().unseal(resp.cookies["jira_session"], resp.cookies["jira_refresh"])
    assert renewed.access_token == "new" and renewed.refresh_token == "rt-new"


def test_tampered_cookie_is_ignored():
    main, _ = auth().seal(JiraSession("at", time.time() + 60, [JiraSite("c", "https://x", "X")], "c"))
    assert auth().unseal(main[:-4] + "AAAA") is None
    other = AtlassianAuth(OAuthSettings(client_id="cid", client_secret="different"))
    assert other.unseal(main) is None


def test_state_expires():
    a = auth()
    assert a.check_state(a.make_state()) and not a.check_state("junk") and not a.check_state(None)
