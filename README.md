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
             │                            fields (email, password…), roles, states, channels,
             │                            Given / When / Then parts
             │
             └─► Test design engine       technique per category → concrete test cases
                                          (equivalence partitioning, boundary value analysis,
                                          state transitions, access control, fault injection,
                                          load testing, accessibility …) + risk-based priority
```

| Part | File | Library |
|---|---|---|
| Datasets & loaders | `testcase_ai/model/datasets.py` | pandas, openpyxl |
| Classifier (TF-IDF word + char n-grams → logistic regression, 3 heads) | `testcase_ai/model/classifier.py` | scikit-learn |
| Information extraction | `testcase_ai/model/extractor.py` | re |
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
cp .env.example .env   # optional: Jira credentials
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
5. Download test cases as CSV, JSON or Gherkin.

## Deploy to Vercel

The deployed app is a FastAPI function (`api/index.py`) plus a static page (`public/index.html`).
Vercel's Python functions have a size limit that numpy + scipy + scikit-learn (~300 MB) would hit,
so the deployment runs the **portable model**: the same trained classifier exported to
`testcase_ai/model/artifacts/model.json.gz` (3 MB) and executed with the Python standard library.
`tests/test_deploy.py` checks that it predicts exactly like the scikit-learn model.

1. After retraining, export the portable model and commit it:
   `python -m testcase_ai.model.train --portable`
2. Import the GitHub repository in Vercel (framework preset "Other"; `vercel.json` sets everything else).
3. Optional, for the Jira tab: add `JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`
   (and `JIRA_ACCEPTANCE_CRITERIA_FIELD`) as environment variables in the Vercel project.
   The token stays on the server; the page never sees it.

Run the deployed version locally: `uvicorn api.index:app --reload`, then open http://127.0.0.1:8000.

## Use from Python

```python
from testcase_ai import JiraClient, TestCaseModel
from testcase_ai.config import JiraSettings
from testcase_ai.exporters import to_csv

model = TestCaseModel.load_or_train()
stories = JiraClient(JiraSettings.from_env()).search_stories("project = PROJ AND issuetype = Story")
suites = [model.generate(s, max_cases=12) for s in stories]
open("test_cases.csv", "w").write(to_csv(suites))
```

## Tests

```bash
pytest
```
