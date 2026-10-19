# Model experiments

All numbers are 5-fold stratified cross-validation of the full model (all classifier heads),
scored on the 12 categories. Weakly labeled rows (FR/NFR only, LLM-assigned labels) are only ever
added to the training folds, never scored.

## Round 1 – which datasets help? (test set: PROMISE + handcrafted, shared vocabulary)

| Training data | Accuracy | Macro F1 |
|---|---|---|
| PROMISE + handcrafted | 0.789 | 0.690 |
| + DOSSPRE | 0.781 | 0.670 |
| + DOSSPRE + FR/NFR (weight 0.3) | 0.776 | 0.651 |
| + DOSSPRE + FR/NFR, binary head 0.3 | 0.780 | 0.662 |
| + DOSSPRE + FR/NFR + AI-labeled | 0.780 | 0.653 |
| + DOSSPRE + story vocabulary | 0.777 | 0.659 |

Scored only on PROMISE-style text, nothing beats PROMISE + handcrafted. An earlier two-stage
model with a separate vocabulary per head scored 0.719 here, so per-head vocabularies were restored.

## Round 2 – all gold data as test set (PROMISE + DOSSPRE + handcrafted, 2,816 rows)

| Training data | All: acc / F1 | DOSSPRE rows: acc | PROMISE + hand: acc / F1 |
|---|---|---|---|
| gold, but DOSSPRE left out of training | 0.670 / 0.625 | 0.516 | 0.791 / 0.699 |
| gold | 0.754 / 0.693 | 0.714 | 0.785 / 0.700 |
| gold + story vocabulary | 0.750 / 0.679 | 0.710 | 0.781 / 0.687 |
| **gold + FR/NFR in a functional-vs-non-functional head (0.3)** | **0.756 / 0.697** | 0.706 | **0.796 / 0.705** |
| gold + FR/NFR + AI-labeled | 0.755 / 0.688 | 0.709 | 0.791 / 0.696 |
| gold + FR/NFR + story vocabulary | 0.755 / 0.681 | 0.712 | 0.788 / 0.688 |

Decisions:
- **DOSSPRE: keep.** Without it the model only reaches 52% on DOSSPRE-style requirements (71% with it).
- **FR/NFR: keep**, but only in a separate functional-vs-non-functional head blended at 0.3 with
  sample weight 0.3. Used directly as training rows it drowns the small classes.
- **AI-labeled requirements: not used by default** (labels are inconsistent – the same
  "log in securely" requirement is labeled Functional, Security and FR in different sheets).
  Opt in with `python -m testcase_ai.model.train --with-ai-labeled`.
- **Story vocabulary** (fitting TF-IDF on the 2,200 unlabeled acceptance criteria): not used, it
  lowered every score. The 582 stories are used as demo input and for the coverage check instead.

## Round 3 – story-style access-control examples

The model labeled "Only administrators can delete user accounts" as functional: the gold data
phrases security as "The system shall …" and has almost no story-style role rules. 20 handcrafted
examples ("Only HR staff can view employee salaries", …) fixed this without turning plain
functional stories into security ("Admins can export the user list" stays functional).

| Split seed | Macro F1 without | Macro F1 with | Accuracy without / with |
|---|---|---|---|
| 42 | 0.693 | 0.663 | 0.762 / 0.757 |
| 7 | 0.663 | 0.645 | 0.755 / 0.755 |
| 123 | 0.674 | 0.664 | 0.759 / 0.755 |

They cost ~0.01–0.02 macro F1 on the "shall"-style test set (split-to-split noise is ±0.015) at
equal accuracy; kept because role rules are common in Jira stories and the test set cannot
reward them. The smallest classes (workflow, compliance, boundary, validation: 24–41 examples)
dominate macro F1 and its noise – more labeled examples there is the best next improvement.

## Final model

8,675 training rows (2,898 gold + 5,777 FR/NFR), accuracy **≈0.76**, macro F1 **≈0.66** (mean of
three splits: 0.657). Per class (seed 42): functional 0.81, operational 0.79, usability 0.76,
performance 0.72, security 0.72, reliability 0.69, notification 0.69, error handling 0.67,
validation 0.61, boundary 0.58, compliance 0.46, workflow 0.46.

On the 582 real stories from `f1.txt` / `f2.txt` every acceptance criterion is covered by at
least one generated test case.
