# Jira User Story → Test Case Generator (own model)

Pulls user stories from Jira and generates test cases with **a model you train and run yourself**.
No external AI API is called — everything (training and inference) runs locally on CPU.
A simple Streamlit interface lets you test it, review its decisions and teach it.

## How the model works

```
Jira story ─► split into acceptance criteria
             │
             ├─► Classifier (trained)      what kind of requirement is this?
             │     stage 1: requirement type   functional / security / performance / usability /
             │                                 reliability / compliance / operational
             │     stage 2: test focus         validation / boundary / error handling /
             │                                 workflow / notification / general
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
| Classifier (TF-IDF word + char n-grams → logistic regression, 2 stages) | `testcase_ai/model/classifier.py` | scikit-learn |
| Information extraction | `testcase_ai/model/extractor.py` | re |
| Test design techniques | `testcase_ai/model/test_design.py` | — |
| End-to-end model (train / save / load / generate / feedback) | `testcase_ai/model/engine.py` | joblib |
| Training CLI | `testcase_ai/model/train.py` | argparse |
| Jira client | `testcase_ai/jira_client.py` | requests |
| UI | `app.py` | Streamlit |

### Training data

- **PROMISE NFR dataset** + **software_requirements_extended** from the project's Google Drive
  (`Graduation Project SWE 12/Datasets/Original Datasets`) — 1,320 labeled requirements after
  de-duplication, mapped to the model's categories. Raw files in `datasets/raw`, see `datasets/README.md`.
- **Handcrafted requirements** (`testcase_ai/model/data/handcrafted_requirements.csv`) — ~290
  story-style examples, mainly for the test-focus categories PROMISE does not have.
- **Your corrections** from the UI and any **datasets you upload** (stored in `models/`, not committed).

### Current scores (5-fold cross-validation, 1,607 requirements)

Accuracy **0.81**, macro F1 **0.72**. Strongest: functional 0.87, notification 0.81, performance 0.78,
operational 0.78, security 0.75. Weakest (fewest examples): workflow 0.53, compliance 0.55 —
adding labeled examples for these is the quickest way to improve the model.

On the 45 real stories in `samples/real_stories.json` (held out, never trained on) every
acceptance criterion is covered by at least one test case.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: Jira credentials
```

## Train the model

```bash
python -m testcase_ai.model.train                    # train + cross-validate + save to models/
python -m testcase_ai.model.train --data my.csv      # include your own labeled data
```

Your own data can be CSV/XLSX with a text column (`text`, `requirement`, …) and a label column
(`category`, `label`, `type`, `class`). Labels may be the model's categories
(`functional, validation, boundary, security, error_handling, reliability, workflow, performance,
notification, usability, compliance, operational`) or PROMISE codes (`F, SE, PE, US, LF, A, FT, L, O, MN, PO, SC`).
PROMISE `.txt` files work too. The app trains a model automatically on first start if none exists.

## Run the interface

```bash
streamlit run app.py
```

1. Load stories — **From Jira** (issue keys or JQL), **Enter manually**, or **Sample stories**.
2. Click **Generate test cases**.
3. Check the **Model analysis** table: category, confidence, techniques and extracted facts for each
   criterion. Fix wrong categories and **Save corrections**.
4. **Retrain model** in the sidebar — it learns from your corrections and uploaded datasets.
5. Download test cases as CSV, JSON or Gherkin.

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
