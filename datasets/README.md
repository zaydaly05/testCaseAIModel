# Datasets

Raw datasets from the project's Google Drive (`Graduation Project SWE 12/Datasets`), used to build
the processed CSVs in `testcase_ai/model/data` (rebuild with `python -m testcase_ai.model.datasets`).

| File | Source | Used as | Licence |
|---|---|---|---|
| `raw/PROMISE_exp.txt` | PROMISE NFR dataset (expanded), Jane Cleland-Huang, 2007 | gold | CC BY-SA 3.0 |
| `raw/software_requirements_extended.csv` | Kaggle "software requirements" (PROMISE + extra FR/NFR) | gold | see Kaggle page |
| `raw/DOSSPRE_1.0Original.csv` | DOSSPRE 1.0 (Mendeley Data) | gold | see Mendeley page |
| `raw/FR_NFR_Dataset_Original.xlsx` | FR/NFR dataset (Mendeley Data) | weak: functional vs non-functional | see Mendeley page |
| `raw/Classified_AI_Requirements.xlsx` | Project data: LLM-generated + LLM-classified requirements | not used by default | project |

Label mapping to the model's categories lives in `testcase_ai/model/taxonomy.py`
(`PROMISE_LABELS` and `EXTRA_LABELS`).

`samples/drive_stories.json` holds the 582 user stories with acceptance criteria parsed from the
project's `Use Cases/f1.txt` and `f2.txt`. They have no category labels and are not used for
training – they are demo input and the coverage check.

Not used: `AI_Requirements*.xlsx` (same texts as the classified file, without labels), the Zenodo
SRS requirement lists (no category labels), `Testing_carlease.csv` (IDs and effort numbers only),
the CarlosKidman test-case set (embeddings only, no text), the small `g*.txt` story files (no
acceptance criteria), and the self-healing / failure-triage datasets (test-run repair, not
test-case design).
