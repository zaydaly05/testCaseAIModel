"""Sign in with Atlassian: OAuth 2.0 (3LO) so every user pulls stories from their own Jira Cloud
site, with their own permissions, instead of one shared account.

One-time setup: create an OAuth 2.0 integration at https://developer.atlassian.com/console/myapps/,
add the Jira API scopes read:jira-work and read:jira-user, register the callback URL(s)
(web app: <your URL>/api/auth/callback, Streamlit: http://localhost:8501), then set
ATLASSIAN_CLIENT_ID and ATLASSIAN_CLIENT_SECRET.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import requests
from cryptography.fernet import Fernet, InvalidToken

from .config import JiraSettings
from .jira_client import JiraAuthError, JiraError

AUTH_URL = "https://auth.atlassian.com/authorize"
TOKEN_URL = "https://auth.atlassian.com/oauth/token"
RESOURCES_URL = "https://api.atlassian.com/oauth/token/accessible-resources"
JIRA_GATEWAY = "https://api.atlassian.com/ex/jira"
# offline_access returns a refresh token, so a sign-in lasts longer than the 1 hour access token.
SCOPES = "read:jira-work read:jira-user offline_access"
STATE_MAX_AGE = 600  # seconds the user has to finish signing in on Atlassian's page


@dataclass
class OAuthSettings:
    client_id: str = ""
    client_secret: str = ""
    session_secret: str = ""  # encrypts sessions; derived from the client secret when empty
    redirect_uri: str = ""  # web app callback; derived from the request URL when empty
    streamlit_redirect_uri: str = ""  # Streamlit callback; the app's own URL when empty

    @classmethod
    def from_env(cls) -> OAuthSettings:
        return cls(
            client_id=os.getenv("ATLASSIAN_CLIENT_ID", "").strip(),
            client_secret=os.getenv("ATLASSIAN_CLIENT_SECRET", "").strip(),
            session_secret=os.getenv("SESSION_SECRET", "").strip(),
            redirect_uri=os.getenv("ATLASSIAN_REDIRECT_URI", "").strip(),
            streamlit_redirect_uri=os.getenv("ATLASSIAN_STREAMLIT_REDIRECT_URI", "").strip(),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)


@dataclass
class JiraSite:
    id: str  # Atlassian "cloud id"
    url: str
    name: str


@dataclass
class JiraSession:
    """One signed-in user: their tokens and the Jira sites their account can reach."""

    access_token: str
    expires_at: float
    sites: list[JiraSite]
    site_id: str
    refresh_token: str = ""
    user_name: str = ""

    @property
    def site(self) -> JiraSite:
        return next((s for s in self.sites if s.id == self.site_id), self.sites[0])

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at - 60

    def jira_settings(self, base: JiraSettings | None = None) -> JiraSettings:
        """Settings for JiraClient that call this user's site through Atlassian's API gateway."""
        return dataclasses.replace(base or JiraSettings(), base_url=self.site.url, email="",
                                   api_token=self.access_token, api_version="3", cloud_id=self.site.id)

    def public(self) -> dict[str, Any]:
        """What the browser may see (no tokens)."""
        return {"user": self.user_name, "site": dataclasses.asdict(self.site),
                "sites": [dataclasses.asdict(s) for s in self.sites]}


@dataclass
class AtlassianAuth:
    settings: OAuthSettings
    http: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        if not self.settings.is_configured:
            raise JiraError("Sign in with Atlassian is not set up: set ATLASSIAN_CLIENT_ID and ATLASSIAN_CLIENT_SECRET.")
        secret = self.settings.session_secret or f"testcase-ai-session:{self.settings.client_secret}"
        self._fernet = Fernet(base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest()))

    # ------------------------------------------------------------ OAuth flow
    def authorize_url(self, redirect_uri: str, state: str) -> str:
        return AUTH_URL + "?" + urlencode({
            "audience": "api.atlassian.com", "client_id": self.settings.client_id, "scope": SCOPES,
            "redirect_uri": redirect_uri, "state": state, "response_type": "code", "prompt": "consent",
        })

    def make_state(self) -> str:
        """An encrypted, timestamped nonce: check_state accepts it for STATE_MAX_AGE seconds."""
        return self._fernet.encrypt(secrets.token_bytes(16)).decode()

    def check_state(self, state: str | None) -> bool:
        try:
            self._fernet.decrypt((state or "").encode(), ttl=STATE_MAX_AGE)
            return True
        except InvalidToken:
            return False

    def exchange_code(self, code: str, redirect_uri: str) -> JiraSession:
        tokens = self._token({"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri})
        sites = self._sites(tokens["access_token"])
        if not sites:
            raise JiraError("Your Atlassian account has no Jira site that this app was given access to.")
        session = JiraSession(access_token=tokens["access_token"], expires_at=time.time() + tokens.get("expires_in", 3600),
                              refresh_token=tokens.get("refresh_token", ""), sites=sites, site_id=sites[0].id)
        session.user_name = self._user_name(session)
        return session

    def refresh(self, session: JiraSession) -> JiraSession:
        if not session.refresh_token:
            raise JiraAuthError("Your Jira sign-in expired. Sign in again.")
        tokens = self._token({"grant_type": "refresh_token", "refresh_token": session.refresh_token})
        # Atlassian rotates refresh tokens: the old one stops working once a new one is issued.
        return dataclasses.replace(session, access_token=tokens["access_token"],
                                   expires_at=time.time() + tokens.get("expires_in", 3600),
                                   refresh_token=tokens.get("refresh_token", session.refresh_token))

    def fresh(self, session: JiraSession) -> JiraSession:
        return self.refresh(session) if session.expired else session

    def _token(self, payload: dict[str, str]) -> dict[str, Any]:
        payload = {"client_id": self.settings.client_id, "client_secret": self.settings.client_secret, **payload}
        try:
            resp = self.http.post(TOKEN_URL, json=payload, timeout=30)
        except requests.RequestException as exc:
            raise JiraError(f"Could not reach Atlassian: {exc}") from exc
        if resp.status_code in (400, 401, 403):  # invalid, expired or revoked code / refresh token
            raise JiraAuthError(f"Atlassian sign-in failed ({resp.status_code}): {resp.text[:300]}")
        if not resp.ok:
            raise JiraError(f"Atlassian sign-in failed ({resp.status_code}): {resp.text[:300]}")
        return resp.json()

    def _sites(self, access_token: str) -> list[JiraSite]:
        resp = self.http.get(RESOURCES_URL, headers=_bearer(access_token), timeout=30)
        if not resp.ok:
            raise JiraError(f"Could not list your Jira sites ({resp.status_code}).")
        return [JiraSite(r["id"], r["url"].rstrip("/"), r.get("name", r["url"]))
                for r in resp.json() if any(s.startswith("read:jira") for s in r.get("scopes", []))]

    def _user_name(self, session: JiraSession) -> str:
        try:
            resp = self.http.get(f"{JIRA_GATEWAY}/{session.site_id}/rest/api/3/myself",
                                 headers=_bearer(session.access_token), timeout=30)
            return resp.json().get("displayName", "") if resp.ok else ""
        except (requests.RequestException, ValueError):
            return ""

    # ------------------------------------------------------------ sessions as encrypted cookies
    # Two cookies, because the access and refresh tokens together can exceed a cookie's 4 KB limit.
    def seal(self, session: JiraSession) -> tuple[str, str]:
        main = {"at": session.access_token, "exp": session.expires_at, "site": session.site_id,
                "user": session.user_name, "sites": [[s.id, s.url, s.name] for s in session.sites]}
        main_cookie = self._fernet.encrypt(json.dumps(main).encode()).decode()
        refresh_cookie = self._fernet.encrypt(session.refresh_token.encode()).decode() if session.refresh_token else ""
        return main_cookie, refresh_cookie

    def unseal(self, main_cookie: str | None, refresh_cookie: str | None = None) -> JiraSession | None:
        if not main_cookie:
            return None
        try:
            main = json.loads(self._fernet.decrypt(main_cookie.encode()))
            refresh = self._fernet.decrypt(refresh_cookie.encode()).decode() if refresh_cookie else ""
            return JiraSession(access_token=main["at"], expires_at=main["exp"], site_id=main["site"],
                               user_name=main.get("user", ""), refresh_token=refresh,
                               sites=[JiraSite(*s) for s in main["sites"]])
        except (InvalidToken, ValueError, KeyError, TypeError):
            return None


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}
