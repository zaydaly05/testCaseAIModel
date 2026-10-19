"""Settings loaded from environment variables (and a local .env file)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class JiraSettings:
    base_url: str = ""
    email: str = ""
    api_token: str = ""
    api_version: str = "3"
    acceptance_criteria_field: str = ""
    project_key: str = ""
    default_jql: str = ""
    cloud_id: str = ""  # set when signed in with Atlassian: requests go through api.atlassian.com

    @classmethod
    def from_env(cls) -> JiraSettings:
        return cls(
            base_url=os.getenv("JIRA_BASE_URL", "").rstrip("/"),
            email=os.getenv("JIRA_EMAIL", ""),
            api_token=os.getenv("JIRA_API_TOKEN", ""),
            api_version=os.getenv("JIRA_API_VERSION", "3"),
            acceptance_criteria_field=os.getenv("JIRA_ACCEPTANCE_CRITERIA_FIELD", ""),
            project_key=os.getenv("JIRA_PROJECT_KEY", "").strip(),
            default_jql=os.getenv("JIRA_DEFAULT_JQL", "").strip(),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.api_token)

    def all_stories_jql(self) -> str:
        """Query used by "pull all stories": JIRA_DEFAULT_JQL, else every Story (optionally in JIRA_PROJECT_KEY)."""
        if self.default_jql:
            return self.default_jql
        scope = ""
        if self.project_key:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.project_key):
                raise ValueError(f"JIRA_PROJECT_KEY is not a valid project key: {self.project_key!r}")
            scope = f"project = {self.project_key} AND "
        return f"{scope}issuetype = Story ORDER BY created DESC"
