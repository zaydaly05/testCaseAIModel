"""Streamlit front end: pull stories from Jira (or enter them by hand) and generate test cases.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from testcase_ai import (
    ClaudeTestCaseGenerator,
    GenerationError,
    JiraClient,
    JiraError,
    RuleBasedGenerator,
    TestSuite,
    UserStory,
)
from testcase_ai.config import AISettings, JiraSettings
from testcase_ai.exporters import to_csv, to_gherkin, to_json

SAMPLES = Path(__file__).parent / "samples" / "sample_stories.json"
TYPE_ICONS = {
    "positive": "✅",
    "negative": "⛔",
    "edge": "📐",
    "security": "🔒",
    "performance": "⚡",
    "accessibility": "♿",
}

st.set_page_config(page_title="Test Case Generator", page_icon="🧪", layout="wide")
st.session_state.setdefault("stories", [])
st.session_state.setdefault("suites", {})


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Jira connection")
    env_jira = JiraSettings.from_env()
    jira_settings = JiraSettings(
        base_url=st.text_input("Base URL", env_jira.base_url, placeholder="https://your-domain.atlassian.net").rstrip("/"),
        email=st.text_input("Email (Cloud)", env_jira.email, help="Leave empty for Server/DC personal access tokens"),
        api_token=st.text_input("API token", env_jira.api_token, type="password"),
        api_version=st.selectbox("API version", ["3", "2"], index=0 if env_jira.api_version == "3" else 1,
                                 help="3 = Jira Cloud, 2 = Jira Server / Data Center"),
        acceptance_criteria_field=st.text_input(
            "Acceptance criteria field", env_jira.acceptance_criteria_field,
            placeholder="customfield_10035", help="Optional custom field holding acceptance criteria",
        ),
    )

    st.header("AI model")
    env_ai = AISettings.from_env()
    api_key = st.text_input(
        "Anthropic API key", os.getenv("ANTHROPIC_API_KEY", ""), type="password",
        help="Leave empty to use the offline rule-based generator",
    )
    use_ai = st.toggle("Use Claude", value=bool(api_key))
    ai_settings = AISettings(
        model=st.text_input("Model", env_ai.model),
        effort=st.select_slider("Effort", ["low", "medium", "high", "xhigh", "max"], value=env_ai.effort),
    )
    max_cases = st.slider("Max test cases per story", 3, 25, 10)


def get_generator():
    if use_ai:
        return ClaudeTestCaseGenerator(ai_settings, api_key=api_key)
    return RuleBasedGenerator()


def add_stories(new: list[UserStory]) -> None:
    existing = {s.key for s in st.session_state.stories}
    st.session_state.stories.extend(s for s in new if s.key not in existing)


# ---------------------------------------------------------------- main
st.title("🧪 Jira User Story → Test Case Generator")
st.caption("Pull user stories from Jira and let AI draft test cases for them.")

tab_jira, tab_manual, tab_sample = st.tabs(["From Jira", "Enter manually", "Sample stories"])

with tab_jira:
    mode = st.radio("Fetch by", ["Issue key", "JQL query"], horizontal=True)
    if mode == "Issue key":
        keys = st.text_input("Issue key(s)", placeholder="PROJ-123, PROJ-124")
    else:
        jql = st.text_area("JQL", 'project = PROJ AND issuetype = Story AND statusCategory != Done ORDER BY created DESC')
        limit = st.number_input("Max issues", 1, 200, 20)
    if st.button("Fetch stories", type="primary", disabled=not jira_settings.is_configured):
        try:
            client = JiraClient(jira_settings)
            with st.spinner("Contacting Jira..."):
                if mode == "Issue key":
                    fetched = [client.get_story(k) for k in keys.replace(",", " ").split() if k]
                else:
                    fetched = client.search_stories(jql, int(limit))
            add_stories(fetched)
            st.success(f"Fetched {len(fetched)} story(ies).")
        except JiraError as exc:
            st.error(str(exc))
    if not jira_settings.is_configured:
        st.info("Fill in the Jira connection in the sidebar to fetch stories.")

with tab_manual:
    with st.form("manual", clear_on_submit=True):
        m_key = st.text_input("Key", "MANUAL-1")
        m_summary = st.text_input("Summary", placeholder="As a <user>, I want <goal> so that <benefit>")
        m_desc = st.text_area("Description")
        m_ac = st.text_area("Acceptance criteria", placeholder="Given ...\nWhen ...\nThen ...")
        if st.form_submit_button("Add story") and m_summary:
            add_stories([UserStory(key=m_key, summary=m_summary, description=m_desc, acceptance_criteria=m_ac)])

with tab_sample:
    st.write("Load example stories to try the generator without Jira.")
    if st.button("Load sample stories"):
        add_stories([UserStory(**s) for s in json.loads(SAMPLES.read_text())])

stories: list[UserStory] = st.session_state.stories
st.divider()
st.subheader(f"User stories ({len(stories)})")

if not stories:
    st.info("No stories yet. Fetch from Jira, enter one manually, or load the samples.")
    st.stop()

selected = st.multiselect(
    "Stories to generate test cases for",
    [s.key for s in stories],
    default=[s.key for s in stories],
)
c1, c2 = st.columns([1, 5])
generate = c1.button("Generate test cases", type="primary", disabled=not selected)
if c2.button("Clear all"):
    st.session_state.stories, st.session_state.suites = [], {}
    st.rerun()

if generate:
    generator = get_generator()
    progress = st.progress(0.0, text=f"Generating with {generator.name}...")
    for i, story in enumerate(s for s in stories if s.key in selected):
        try:
            st.session_state.suites[story.key] = generator.generate(story, max_cases)
        except GenerationError as exc:
            st.error(f"{story.key}: {exc}")
        progress.progress((i + 1) / len(selected), text=f"Generated {story.key}")
    progress.empty()

for story in stories:
    suite: TestSuite | None = st.session_state.suites.get(story.key)
    label = f"**{story.key}** — {story.summary}" + (f"  ·  {len(suite.test_cases)} test cases" if suite else "")
    with st.expander(label, expanded=suite is not None):
        if story.url:
            st.markdown(f"[Open in Jira]({story.url})")
        st.markdown(f"**Description**\n\n{story.description or '_none_'}")
        if story.acceptance_criteria:
            st.markdown("**Acceptance criteria**")
            st.text(story.acceptance_criteria)
        if not suite:
            continue
        st.caption(f"Generated by {suite.generator}. {suite.coverage_notes}")
        st.dataframe(
            pd.DataFrame(
                [{"ID": t.id, "Type": f"{TYPE_ICONS.get(t.type.value, '')} {t.type.value}",
                  "Priority": t.priority.value, "Title": t.title} for t in suite.test_cases]
            ),
            hide_index=True, use_container_width=True,
        )
        for tc in suite.test_cases:
            with st.container(border=True):
                st.markdown(f"**{tc.id} · {tc.title}**  \n"
                            f"`{tc.type.value}` `{tc.priority.value}` — covers: _{tc.acceptance_criterion}_")
                if tc.preconditions:
                    st.markdown("**Preconditions:** " + "; ".join(tc.preconditions))
                if tc.test_data:
                    st.markdown("**Test data:** " + "; ".join(tc.test_data))
                st.table(pd.DataFrame(
                    [{"#": i, "Action": s.action, "Expected result": s.expected_result}
                     for i, s in enumerate(tc.steps, 1)]
                ).set_index("#"))
                st.markdown(f"**Expected result:** {tc.expected_result}")

suites = list(st.session_state.suites.values())
if suites:
    st.divider()
    st.subheader("Export")
    e1, e2, e3 = st.columns(3)
    e1.download_button("Download CSV", to_csv(suites), "test_cases.csv", "text/csv")
    e2.download_button("Download JSON", to_json(suites), "test_cases.json", "application/json")
    e3.download_button("Download Gherkin", to_gherkin(suites), "test_cases.feature", "text/plain")
