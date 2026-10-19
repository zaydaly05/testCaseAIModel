# Jira User Story → Test Case Generator (own model)

Pulls user stories from Jira and generates test cases with **a model you train and run yourself**.
No external AI API is called — everything (training and inference) runs locally on CPU.
A simple Streamlit interface lets you test it, review its decisions and teach it.

## How the model works

```
Jira story ─► split into acceptance criteria
             │
             ├─► Classifier (trained)      what kind of requirement is this?
             │     type head    functional / security / performance / usability /
             │                  reliability / compliance / operational
             │     kind head    functional vs non-functional (adds the 6k-row FR/NFR dataset)
             │     focus head   validation / boundary / error handling /
             │                  workflow / notification / general
             │
             ├─► Extractor (NLP rules)    limits (≥ 12 characters, 1–99, within 5 minutes),
             │                            fields (email, password…), roles, states, channels
             │
             ├─► Behaviour reader         the logic of the criterion: conditions → action → outcomes,
             │                            HTTP requests and status codes, listed options, defaults,
             │                            clamping, and the opposite of each condition / outcome
             │
             └─► Test design engine       test cases that check that logic from every side
                                          + risk-based priority
```

For each criterion the engine derives, from what the criterion itself says:

| Side tested | Example |
|---|---|
| The scenario, every outcome checked | `POST /api/logs` → status 2xx, the entry is stored |
| Each condition reversed (decision table) | *valid key* → *invalid key* refused; *database down* → *up*: normal path, no fallback |
| Each listed option | "filter by date, source, or relevance" → one test per option |
| Limits | 11 / 12 characters, 0 and 6 clamped to 1 and 5, missing rating defaults to 5, first 12 shown + count |
| Timings | updates every second; just before 24 hours the link still works |
| Data changes | read the change back, undo it (save → remove), no duplicate when repeated |
| Endpoints | no credentials → 401, unknown id → 404, malformed body → 400 |
| Who may do it | "Only admins can delete users" → admins can, others cannot |
| Category checks, when the text calls for them | access control, injection, p95 response time, load, accessibility, responsive layout, delivery and content of messages |

Nothing is invented: a format rule is only tested when the story names a field with a format,
and a criterion with no concrete check gets one scenario test, not a template.

| Part | File | Library |
|---|---|---|
| Datasets & loaders | `testcase_ai/model/datasets.py` | pandas, openpyxl |
| Classifier (TF-IDF word + char n-grams → logistic regression, 3 heads) | `testcase_ai/model/classifier.py` | scikit-learn |
| Test cases as code (pytest, Playwright) | `testcase_ai/codegen.py`, `testcase_ai/codegen_playwright.py` | — |
| Information extraction | `testcase_ai/model/extractor.py` | re |
| Behaviour reader (conditions, actions, outcomes, opposites) | `testcase_ai/model/behaviour.py` | re |
| Test design techniques | `testcase_ai/model/test_design.py` | — |
| End-to-end model (train / save / load / generate / feedback) | `testcase_ai/model/engine.py` | joblib |
| Training CLI | `testcase_ai/model/train.py` | argparse |
| Jira client | `testcase_ai/jira_client.py` | requests |
| Local UI (training, corrections, retraining) | `streamlit_app.py` | Streamlit |
| Web API (deployed on Vercel) | `api/index.py` | FastAPI |
| Web page (deployed on Vercel) | `public/index.html` | HTML + JavaScript |
| Portable model (no scikit-learn needed) | `testcase_ai/model/portable.py`, `testcase_ai/model/artifacts/model.json.gz` | Python standard library |

### Training data (all from the project's Google Drive)

| Dataset | Rows used | Labels | Role |
|---|---|---|---|
| PROMISE NFR + software_requirements_extended | 1,429 | 12 PROMISE classes | gold |
| DOSSPRE 1.0 | 1,271 | security / quality classes | gold |
| Handcrafted story-style examples | ~310 | all 12 categories | gold |
| FR_NFR_Dataset | 5,945 | functional / non-functional | weak – functional-vs-non-functional head only |
| Classified_AI_Requirements (LLM-labeled) | 1,091 | mixed | not used by default (`--with-ai-labeled`) |
| f1.txt / f2.txt user stories | 582 stories, 2,200 criteria | none | demo input + coverage check |

Raw files are in `datasets/raw` (see `datasets/README.md`); `python -m testcase_ai.model.datasets`
rebuilds the processed CSVs. Your UI corrections and uploaded datasets are added on retraining
(stored in `models/`, not committed).

### Scores

5-fold cross-validation on 2,898 gold-labeled requirements: accuracy **≈0.76**, macro F1 **≈0.66**.
Strongest: functional 0.81, operational 0.79, usability 0.76, security 0.72, performance 0.72.
Weakest: workflow and compliance (0.46) – they have the fewest examples, so labeling more of
those is the quickest improvement. Every acceptance criterion of all 582 real stories is covered
by at least one test case. How each dataset was chosen is documented in `docs/experiments.md`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # requirements.txt alone = what the deployed API needs
cp .env.example .env   # optional: Sign in with Atlassian (see below)
```

## Train the model

```bash
python -m testcase_ai.model.train                    # train + cross-validate + save to models/
python -m testcase_ai.model.train --data my.csv      # include your own labeled data
python -m testcase_ai.model.train --with-ai-labeled  # also use the LLM-labeled requirements
```

Your own data can be CSV/XLSX with a text column (`text`, `requirement`, …) and a label column
(`category`, `label`, `type`, `class`, `classification`). Labels may be the model's categories
(`functional, validation, boundary, security, error_handling, reliability, workflow, performance,
notification, usability, compliance, operational`) PROMISE codes (`F, SE, PE, US, LF, A, FT, L, O, MN, PO, SC`), DOSSPRE classes (`Authentication`,
`Availability`, …) or `FR` / `NFR`. Every sheet of an Excel file is read.
PROMISE `.txt` files work too. The app trains a model automatically on first start if none exists.

## Run the interface

```bash
streamlit run streamlit_app.py
```

1. Load stories — **From Jira** (issue keys or JQL), **Enter manually**, or **Sample stories**.
2. Click **Generate test cases**.
3. Check the **Model analysis** table: category, confidence, techniques and extracted facts for each
   criterion. Fix wrong categories and **Save corrections**.
4. **Retrain model** in the sidebar — it learns from your corrections and uploaded datasets.
5. Download test cases as pytest, Playwright, CSV, JSON or Gherkin.

## Jira sign-in

There is no shared Jira account: every user clicks **Sign in with Atlassian**, logs in with their own
account and approves read access. Stories are then pulled from *their* Jira Cloud site with *their*
permissions. If their account can reach several sites, they pick one.

One-time setup by whoever runs the app:

1. Create an **OAuth 2.0 integration** at https://developer.atlassian.com/console/myapps/.
2. **Permissions** → Jira API → add the scopes `read:jira-work` and `read:jira-user`.
3. **Authorization** → OAuth 2.0 (3LO) → add the callback URLs you use:
   `http://127.0.0.1:8000/api/auth/callback` (web app, local), `http://localhost:8501` (Streamlit),
   `https://<your-domain>/api/auth/callback` (deployed).
4. **Distribution**: while the app is in development only your own account can sign in. To let anyone
   sign in, set it to *Sharing* and fill in the privacy details.
5. Put the app's **Client ID** and **Secret** into `.env` (or the Vercel environment variables) as
   `ATLASSIAN_CLIENT_ID` and `ATLASSIAN_CLIENT_SECRET`.

In the web app the tokens are kept in encrypted, httpOnly cookies (the page never sees them) and
are refreshed automatically; a sign-in lasts up to 30 days. In Streamlit the sign-in lasts until the
browser tab is reloaded. OAuth works for Jira Cloud only; Jira Server / Data Center can still be used
from Python (below).

## Test cases as code

Every generated test case is also a **pytest function** (`codegen.py`): preconditions and steps become
calls on an `app` object, test data becomes real values (`"a" * 11` for an 11-character password),
and expected results become `assert` lines. The **PYTEST (.py)** export is one runnable file with an
`App` class listing every action and check the tests need: implement those methods once for your
system (Playwright for a web UI, `requests` for an API) and run `pytest test_generated.py`.
Until then each test fails with `NotImplementedError` naming the missing method.

### Playwright web tests

Each test case is also a **Playwright web test** (`codegen_playwright.py`, the **Playwright** tab on a
test case and the **PLAYWRIGHT (.py)** export). Steps the model recognises become browser actions:
sign in as a role, open the page, fill a field with the test value and submit, reload, sign out,
Tab-only navigation, a 375 px mobile viewport, and timing the page 30 times for a p95 check. Expected
results become `expect(...)` checks: an error shown or not, access denied, redirected to sign-in,
quoted text visible, no horizontal scrolling. Steps only your team knows (for example "Only admins
can delete users") become small typed helpers at the top of the file to fill in once.

```bash
pip install pytest pytest-playwright
playwright install chromium
APP_URL=https://your-app.example pytest test_generated_playwright.py --headed
```

At the top of the file, set `PAGES` (the page each story is tested on) and `USERS` (one account per
role; `<ROLE>_EMAIL` / `<ROLE>_PASSWORD` environment variables override them). If VS Code says
`Import "pytest" could not be resolved`, install the two packages above into the interpreter VS Code uses.

Stories that name endpoints (`GET /api/logs`, `DELETE /api/items/:id`) become real HTTP calls through
`page.request` with status checks (`assert response.status == 401`). The requests share the browser's
sign-in cookies; for token-based APIs fill `API_TOKENS`, and set `PATH_VALUES` (for `:id`, `:code`, …)
and `PAYLOADS` (request bodies by `"METHOD /path"`).

## Deploy to Vercel

The deployed app is a FastAPI function (`api/index.py`) plus a static page (`public/index.html`).
Vercel's Python functions have a size limit that numpy + scipy + scikit-learn (~300 MB) would hit,
so the deployment runs the **portable model**: the same trained classifier exported to
`testcase_ai/model/artifacts/model.json.gz` (3 MB) and executed with the Python standard library.
`tests/test_deploy.py` checks that it predicts exactly like the scikit-learn model.

1. After retraining, export the portable model and commit it:
   `python -m testcase_ai.model.train --portable`
2. Import the GitHub repository in Vercel (framework preset "Other"; `vercel.json` sets everything else).
3. Optional, for the Jira tab: add `ATLASSIAN_CLIENT_ID` and `ATLASSIAN_CLIENT_SECRET`
   (see [Jira sign-in](#jira-sign-in)) and optionally `JIRA_ACCEPTANCE_CRITERIA_FIELD` as environment
   variables in the Vercel project, and register `https://<your-domain>/api/auth/callback` as a callback URL.
   **Pull all stories from Jira** fetches every issue of type Story (up to 200, newest first).
   Set `JIRA_PROJECT_KEY` to limit it to one project, or `JIRA_DEFAULT_JQL` to use your own query.
   **Generate test cases for all** then runs the model on every pulled story and exports them together.

Run the deployed version locally: `uvicorn api.index:app --reload`, then open http://127.0.0.1:8000.

## Use from Python

With your own credentials (`JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN` in the environment;
for Jira Server / Data Center leave the email empty and set `JIRA_API_VERSION=2`):

```python
from testcase_ai import JiraClient, TestCaseModel
from testcase_ai.config import JiraSettings
from testcase_ai.exporters import to_csv

model = TestCaseModel.load_or_train()
stories = JiraClient(JiraSettings.from_env()).search_stories("project = PROJ AND issuetype = Story")
suites = [model.generate(s, max_cases=12) for s in stories]
open("test_cases.csv", "w").write(to_csv(suites))
```

## Cross-model review with DeepEval

`evals/` uses [DeepEval](https://github.com/confident-ai/deepeval) as an LLM judge. It checks
that the generated test cases match the Jira criteria before they are run, and can compare the model
with a general-purpose LLM ("Model X") on the same stories. Each acceptance criterion is scored on:

| Metric | Question |
|---|---|
| Grounded in the criteria (G-Eval) | Does every feature, value, message and rule come from the story? |
| Covers every side (G-Eval) | Main behaviour, reversed conditions, every option, both sides of each limit? |
| Logically correct (G-Eval) | Do the expected results follow from the steps and the criterion? |
| Hallucination | Does anything contradict the story? (1 = nothing does) |
| Answer Relevancy | Is every test about this criterion? |

The judge is only used for evaluation; the app itself still calls no external AI. Put a key in `.env`:
`GROQ_API_KEY` (free, no card: https://console.groq.com/keys) or `GEMINI_API_KEY` (free:
https://aistudio.google.com/apikey). `EVAL_JUDGE` picks the model,
e.g. `groq:openai/gpt-oss-120b` (the Groq default).

```bash
python -m evals.compare --limit 5                                   # score the own model
python -m evals.compare --baseline groq:qwen/qwen3.8-27b            # side by side with an LLM
python -m evals.compare --stories my_stories.json --fail-under 0.7  # quality gate: exit code 1 below 0.7
deepeval test run evals/test_deepeval.py                            # the same checks as pytest assertions
```

Reports are written to `evals/reports/` (a summary table, plus every failed criterion with the judge's reason).
Judging one criterion takes about 8 LLM calls, so free tiers are slow or run out (Groq: about 8K tokens
a minute and 1,000 requests a day per model; Gemini: 20 requests per model per day). Rate limits are
waited out automatically, as long as the provider asks; judge calls that still fail are listed in the
report, never counted as scores, and the command then exits with code 3.
When the baseline and the judge are the same LLM, the judge tends to favour its own output, so a
different judge gives a fairer comparison.

## Tests

```bash
pytest tests
```
