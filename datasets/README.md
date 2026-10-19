# Datasets

Raw datasets used to build `testcase_ai/model/data/promise_requirements.csv`
(rebuild with `python -m testcase_ai.model.datasets`).

| File | Source | Licence |
|---|---|---|
| `raw/PROMISE_exp.txt` | PROMISE NFR dataset (expanded), Jane Cleland-Huang, 2007 | CC BY-SA 3.0 |
| `raw/software_requirements_extended.csv` | Kaggle "software requirements" (PROMISE + extra FR/NFR) | see Kaggle page |

Both come from the project's `Graduation Project SWE 12/Datasets/Original Datasets` folder.
Their labels are mapped to the model's categories in `testcase_ai/model/taxonomy.py`
(`F/FR → functional`, `SE → security`, `PE/SC → performance`, `US/LF → usability`,
`A/FT → reliability`, `L → compliance`, `O/MN/PO → operational`; the generic `NFR` label is dropped).

`samples/real_stories.json` holds 45 user stories with acceptance criteria taken from the
project's model_x validation/test splits; they are used for evaluation and as demo input,
never for training.
