"""Minimal Jira REST client for pulling user stories.

Supports Jira Cloud (API v3, email + API token) and Jira Server / Data Center
(API v2, personal access token sent as a Bearer token when no email is set).
"""

from __future__ import annotations

from typing import Any, Iterator

import requests

from .config import JiraSettings
from .models import UserStory

BASE_FIELDS = ["summary", "description", "issuetype", "priority", "labels", "components"]


class JiraError(RuntimeError):
    pass


def adf_to_text(node: Any) -> str:
    """Flatten an Atlassian Document Format node (API v3) into plain text."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(adf_to_text(n) for n in node)

    node_type = node.get("type")
    children = node.get("content", [])

    if node_type == "text":
        return node.get("text", "")
    if node_type == "hardBreak":
        return "\n"
    if node_type in ("mention", "emoji"):
        return node.get("attrs", {}).get("text", "")
    if node_type == "listItem":
        return "- " + adf_to_text(children).strip() + "\n"
    if node_type in ("bulletList", "orderedList"):
        return adf_to_text(children)
    if node_type in ("paragraph", "heading", "blockquote", "codeBlock"):
        return adf_to_text(children).rstrip() + "\n"
    if node_type == "tableRow":
        return " | ".join(adf_to_text(c).strip() for c in children) + "\n"
    return adf_to_text(children)


def _field_text(value: Any) -> str:
    if isinstance(value, dict) and value.get("type") == "doc":
        return adf_to_text(value).strip()
    if isinstance(value, dict):
        return str(value.get("value") or value.get("name") or "")
    if isinstance(value, list):
        return "\n".join(_field_text(v) for v in value)
    return (value or "").strip() if isinstance(value, str) else str(value or "")


class JiraClient:
    def __init__(self, settings: JiraSettings, session: requests.Session | None = None):
        if not settings.is_configured:
            raise JiraError("Jira is not configured: set the base URL and API token.")
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers["Accept"] = "application/json"
        if settings.email:
            self.session.auth = (settings.email, settings.api_token)
        else:
            self.session.headers["Authorization"] = f"Bearer {settings.api_token}"

    @property
    def _api(self) -> str:
        return f"{self.settings.base_url}/rest/api/{self.settings.api_version}"

    @property
    def _fields(self) -> list[str]:
        fields = list(BASE_FIELDS)
        if self.settings.acceptance_criteria_field:
            fields.append(self.settings.acceptance_criteria_field)
        return fields

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            resp = self.session.get(f"{self._api}{path}", params=params, timeout=30)
        except requests.RequestException as exc:
            raise JiraError(f"Could not reach Jira: {exc}") from exc
        if resp.status_code == 401:
            raise JiraError("Jira rejected the credentials (401).")
        if resp.status_code == 404:
            raise JiraError(f"Not found in Jira: {path}")
        if not resp.ok:
            raise JiraError(f"Jira returned {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def to_story(self, issue: dict[str, Any]) -> UserStory:
        fields = issue.get("fields", {})
        ac_field = self.settings.acceptance_criteria_field
        return UserStory(
            key=issue["key"],
            summary=fields.get("summary") or "",
            description=_field_text(fields.get("description")),
            acceptance_criteria=_field_text(fields.get(ac_field)) if ac_field else "",
            issue_type=(fields.get("issuetype") or {}).get("name", "Story"),
            priority=(fields.get("priority") or {}).get("name"),
            labels=fields.get("labels") or [],
            components=[c.get("name", "") for c in fields.get("components") or []],
            url=f"{self.settings.base_url}/browse/{issue['key']}",
        )

    def get_story(self, key: str) -> UserStory:
        data = self._get(f"/issue/{key.strip()}", {"fields": ",".join(self._fields)})
        return self.to_story(data)

    def search_stories(self, jql: str, max_results: int = 50) -> list[UserStory]:
        return [self.to_story(i) for i in self._iter_issues(jql, max_results)]

    def _iter_issues(self, jql: str, max_results: int) -> Iterator[dict[str, Any]]:
        fetched = 0
        params: dict[str, Any] = {"jql": jql, "fields": ",".join(self._fields)}
        if self.settings.api_version == "3":
            # Jira Cloud: token-paginated /search/jql endpoint.
            while fetched < max_results:
                params["maxResults"] = min(100, max_results - fetched)
                data = self._get("/search/jql", params)
                for issue in data.get("issues", []):
                    yield issue
                    fetched += 1
                token = data.get("nextPageToken")
                if not token or data.get("isLast", True) or not data.get("issues"):
                    return
                params["nextPageToken"] = token
        else:
            # Jira Server / Data Center: offset-paginated /search endpoint.
            while fetched < max_results:
                params["startAt"] = fetched
                params["maxResults"] = min(100, max_results - fetched)
                data = self._get("/search", params)
                issues = data.get("issues", [])
                for issue in issues:
                    yield issue
                fetched += len(issues)
                if not issues or fetched >= data.get("total", 0):
                    return
