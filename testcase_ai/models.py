"""Data models shared by the Jira client, the generators and the UI."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class UserStory(BaseModel):
    key: str = Field(description="Jira issue key, e.g. PROJ-123")
    summary: str
    description: str = ""
    acceptance_criteria: str = ""
    issue_type: str = "Story"
    priority: str | None = None
    labels: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    url: str | None = None

    def as_prompt_text(self) -> str:
        parts = [f"Key: {self.key}", f"Summary: {self.summary}"]
        if self.priority:
            parts.append(f"Priority: {self.priority}")
        if self.labels:
            parts.append(f"Labels: {', '.join(self.labels)}")
        if self.components:
            parts.append(f"Components: {', '.join(self.components)}")
        parts.append(f"Description:\n{self.description or '(none)'}")
        if self.acceptance_criteria:
            parts.append(f"Acceptance criteria:\n{self.acceptance_criteria}")
        return "\n".join(parts)


class TestType(str, Enum):
    __test__ = False  # not a pytest test class
    POSITIVE = "positive"
    NEGATIVE = "negative"
    EDGE = "edge"
    SECURITY = "security"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    USABILITY = "usability"
    ACCESSIBILITY = "accessibility"
    COMPLIANCE = "compliance"
    COMPATIBILITY = "compatibility"


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TestStep(BaseModel):
    action: str
    expected_result: str


class TestCase(BaseModel):
    id: str = Field(description="Identifier such as PROJ-123-TC01")
    title: str
    type: TestType
    priority: Priority
    preconditions: list[str]
    test_data: list[str]
    steps: list[TestStep]
    expected_result: str
    acceptance_criterion: str = Field(description="The acceptance criterion this test covers")
    technique: str = Field(default="", description="Test design technique used")
    code: str = Field(default="", description="The test case as a pytest function")


class CriterionInsight(BaseModel):
    """What the model understood about one acceptance criterion."""

    text: str
    category: str
    confidence: float
    requirement_type: str
    secondary: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    limits: list[str] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    states: list[str] = Field(default_factory=list)


class TestSuite(BaseModel):
    story: UserStory
    test_cases: list[TestCase]
    insights: list[CriterionInsight] = Field(default_factory=list)
    coverage_notes: str = ""
    generator: str = Field(description="Which generator produced the suite")
