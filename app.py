"""Streamlit front end: pull stories from Jira (or enter them by hand) and generate test cases
with the locally trained TestCaseAI model.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from testcase_ai import JiraClient, JiraError, TestCaseModel, TestSuite, UserStory
from testcase_ai.config import JiraSettings
from testcase_ai.exporters import to_csv, to_gherkin, to_json
from testcase_ai.model.engine import EXTRA_DATA_DIR, FEEDBACK_PATH, MODEL_PATH
from testcase_ai.model.taxonomy import CATEGORIES

SAMPLES = Path(__file__).parent / "samples"
TYPE_ICONS = {
    "positive": "✅", "negative": "⛔", "edge": "📐", "security": "🔒", "performance": "⚡",
    "reliability": "🛟", "usability": "🖱️", "accessibility": "♿", "compliance": "⚖️", "compatibility": "🧩",
}

st.set_page_config(page_title="Test Case Generator", page_icon="🧪", layout="wide")
st.session_state.setdefault("stories", [])
st.session_state.setdefault("suites", {})


@st.cache_resource(show_spinner="Loading the model (training it on first run)...")
def get_model() -> TestCaseModel:
    return TestCaseModel.load_or_train()


def retrain(evaluate: bool = True) -> None:
    with st.spinner("Training the model on all datasets and your corrections..."):
        TestCaseModel.train(evaluate=evaluate).save(MODEL_PATH)
    get_model.clear()
    st.session_state.suites = {}


model = get_model()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🧠 Model")
    meta = model.metadata
    st.caption(f"Trained {meta.get('trained_at', '?')} on **{meta.get('n_samples', '?')}** labeled requirements")
    if ev := meta.get("evaluation"):
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{ev['accuracy']:.0%}")
        c2.metric("Macro F1", f"{ev['macro_f1']:.0%}")
    with st.expander("Training data & evaluation"):
        st.write("Sources:", ", ".join(meta.get("sources", [])))
        if counts := meta.get("class_counts"):
            st.dataframe(pd.Series(counts, name="examples"), use_container_width=True)
        if report := meta.get("evaluation_report"):
            st.code(report, language=None)
        else:
            st.info("Retrain with evaluation to see cross-validation scores.")

    uploaded = st.file_uploader(
        "Add labeled training data", type=["csv", "xlsx", "txt"],
        help="CSV/XLSX with a text column and a category/label column (our categories or PROMISE labels "
             "F, SE, PE, US, ...), or a PROMISE .txt file",
    )
    if uploaded and st.button("Save dataset"):
        EXTRA_DATA_DIR.mkdir(parents=True, exist_ok=True)
        (EXTRA_DATA_DIR / uploaded.name).write_bytes(uploaded.getvalue())
        st.success(f"Saved {uploaded.name}. Retrain to use it.")
    if FEEDBACK_PATH.exists():
        n_feedback = max(0, sum(1 for _ in FEEDBACK_PATH.open(encoding="utf-8")) - 1)
        st.caption(f"{n_feedback} saved correction(s) from reviews")
    if st.button("Retrain model", type="primary", use_container_width=True):
        retrain()
        st.rerun()

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
    max_cases = st.slider("Max test cases per story", 3, 40, 12)


def add_stories(new: list[UserStory]) -> None:
    existing = {s.key for s in st.session_state.stories}
    st.session_state.stories.extend(s for s in new if s.key not in existing)


# ---------------------------------------------------------------- main
st.title("🧪 Jira User Story → Test Case Generator")
st.caption("Your own locally trained model: it classifies each acceptance criterion, extracts limits, fields, "
           "roles and states, and applies test design techniques. No external AI service is used.")

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
        m_ac = st.text_area("Acceptance criteria", placeholder="Given ...\nWhen ...\nThen ...\n- The password must be at least 12 characters")
        if st.form_submit_button("Add story") and m_summary:
            add_stories([UserStory(key=m_key, summary=m_summary, description=m_desc, acceptance_criteria=m_ac)])

with tab_sample:
    st.write("Try the model without Jira.")
    c1, c2 = st.columns(2)
    if c1.button("Load demo stories"):
        add_stories([UserStory(**s) for s in json.loads((SAMPLES / "sample_stories.json").read_text())])
    if c2.button("Load real stories (project dataset)"):
        add_stories([UserStory(**s) for s in json.loads((SAMPLES / "real_stories.json").read_text())][:10])

stories: list[UserStory] = st.session_state.stories
st.divider()
st.subheader(f"User stories ({len(stories)})")

if not stories:
    st.info("No stories yet. Fetch from Jira, enter one manually, or load the samples.")
    st.stop()

selected = st.multiselect("Stories to generate test cases for", [s.key for s in stories], default=[s.key for s in stories])
c1, c2 = st.columns([1, 5])
if c1.button("Generate test cases", type="primary", disabled=not selected):
    for story in stories:
        if story.key in selected:
            st.session_state.suites[story.key] = model.generate(story, max_cases)
if c2.button("Clear all"):
    st.session_state.stories, st.session_state.suites = [], {}
    st.rerun()

category_names = list(CATEGORIES)

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

        st.markdown("#### 🧠 Model analysis")
        st.caption("Correct a category if the model got it wrong, then save — corrections are used the next time you retrain.")
        analysis = pd.DataFrame([{
            "Criterion": i.text, "Category": i.category, "Confidence": i.confidence,
            "Techniques": ", ".join(i.techniques),
            "Facts": "; ".join(filter(None, [", ".join(i.limits), ", ".join(i.fields), ", ".join(i.roles), " → ".join(i.states)])),
        } for i in suite.insights])
        edited = st.data_editor(
            analysis, key=f"analysis-{story.key}", hide_index=True, use_container_width=True,
            disabled=["Criterion", "Confidence", "Techniques", "Facts"],
            column_config={
                "Category": st.column_config.SelectboxColumn(options=category_names, required=True),
                "Confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
            },
        )
        changed = edited[edited["Category"] != analysis["Category"]]
        if not changed.empty and st.button(f"Save {len(changed)} correction(s)", key=f"fb-{story.key}"):
            TestCaseModel.add_feedback(list(zip(changed["Criterion"], changed["Category"])))
            st.success("Saved. Click **Retrain model** in the sidebar to learn from them.")

        st.markdown("#### Test cases")
        st.caption(suite.coverage_notes)
        st.dataframe(
            pd.DataFrame([{"ID": t.id, "Type": f"{TYPE_ICONS.get(t.type.value, '')} {t.type.value}",
                           "Priority": t.priority.value, "Technique": t.technique, "Title": t.title}
                          for t in suite.test_cases]),
            hide_index=True, use_container_width=True,
        )
        for tc in suite.test_cases:
            with st.container(border=True):
                st.markdown(f"**{tc.id} · {tc.title}**  \n"
                            f"`{tc.type.value}` `{tc.priority.value}` `{tc.technique}` — covers: _{tc.acceptance_criterion}_")
                if tc.preconditions:
                    st.markdown("**Preconditions:** " + "; ".join(tc.preconditions))
                if tc.test_data:
                    st.markdown("**Test data:** " + "; ".join(tc.test_data))
                st.table(pd.DataFrame([{"#": i, "Action": s.action, "Expected result": s.expected_result}
                                       for i, s in enumerate(tc.steps, 1)]).set_index("#"))
                st.markdown(f"**Expected result:** {tc.expected_result}")

suites = list(st.session_state.suites.values())
if suites:
    st.divider()
    st.subheader("Export")
    e1, e2, e3 = st.columns(3)
    e1.download_button("Download CSV", to_csv(suites), "test_cases.csv", "text/csv")
    e2.download_button("Download JSON", to_json(suites), "test_cases.json", "application/json")
    e3.download_button("Download Gherkin", to_gherkin(suites), "test_cases.feature", "text/plain")
