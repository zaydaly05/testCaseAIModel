# Jira User Story → Test Case Generator

Pulls user stories from Jira and generates test cases for them with AI (Claude),
with a simple Streamlit web interface to try it out.

## How it works

```
Jira REST API ──► JiraClient ──► UserStory ──► Generator ──► TestSuite ──► UI / CSV / JSON / Gherkin
                 (requests)      (pydantic)    │
                                               ├─ ClaudeTestCaseGenerator  (Anthropic SDK, structured output)
                                               └─ RuleBasedGenerator       (offline fallback, no API key needed)
```

- **`testcase_ai/jira_client.py`** — fetches stories by issue key or JQL. Supports Jira Cloud
  (API v3, email + API token, Atlassian Document Format descriptions) and Jira Server/Data Center
  (API v2, personal access token). Acceptance criteria are read from a custom field if you set one,
  otherwise from an "Acceptance Criteria" section in the description.
- **`testcase_ai/generator.py`**
  - `ClaudeTestCaseGenerator` sends the story to Claude and gets back test cases validated against
    a pydantic schema (`client.beta.messages.parse`), so the output is always well-formed: id, title,
    type (positive / negative / edge / security / performance / accessibility), priority,
    preconditions, test data, steps with expected results, and the acceptance criterion covered.
  - `RuleBasedGenerator` parses the acceptance criteria (bullets and Given/When/Then) and derives
    positive, negative and boundary-value tests deterministically. Used when no API key is set.
- **`testcase_ai/exporters.py`** — CSV (importable into Xray / Zephyr / TestRail with column mapping),
  JSON and Gherkin `.feature` files.
- **`app.py`** — Streamlit front end.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in Jira and Anthropic credentials (all optional)
```

## Run the interface

```bash
streamlit run app.py
```

Open http://localhost:8501, then:

1. Load stories — **From Jira** (issue keys or a JQL query), **Enter manually**, or **Sample stories**
   to try it without Jira.
2. Pick the stories and click **Generate test cases**.
3. Review the test cases and download them as CSV, JSON or Gherkin.

Credentials can be entered in the sidebar or pre-filled from `.env`. Without an Anthropic API key
the app uses the offline rule-based generator.

## Use from Python

```python
from testcase_ai import JiraClient, ClaudeTestCaseGenerator
from testcase_ai.config import JiraSettings
from testcase_ai.exporters import to_csv

stories = JiraClient(JiraSettings.from_env()).search_stories("project = PROJ AND issuetype = Story")
suites = [ClaudeTestCaseGenerator().generate(s, max_cases=10) for s in stories]
open("test_cases.csv", "w").write(to_csv(suites))
```

## Tests

```bash
pytest
```

Jira calls are mocked with `responses` and the Claude client with a mock, so tests need no credentials.
