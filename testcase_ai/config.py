"""Settings loaded from environment variables (and a local .env file)."""

from __future__ import annotations

import os
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

    @classmethod
    def from_env(cls) -> "JiraSettings":
        return cls(
            base_url=os.getenv("JIRA_BASE_URL", "").rstrip("/"),
            email=os.getenv("JIRA_EMAIL", ""),
            api_token=os.getenv("JIRA_API_TOKEN", ""),
            api_version=os.getenv("JIRA_API_VERSION", "3"),
            acceptance_criteria_field=os.getenv("JIRA_ACCEPTANCE_CRITERIA_FIELD", ""),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.api_token)

