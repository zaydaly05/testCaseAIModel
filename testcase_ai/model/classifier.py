"""Three-head requirement classifier with a shared text representation.

Heads (each a logistic regression on the same TF-IDF features):

1. type     functional, or which non-functional category: security, performance,
            usability, reliability, compliance, operational.
2. kind     (optional) functional vs non-functional, trained on every row
            including datasets that only say FR / NFR; blended into head 1.
3. focus    for functional requirements, which test design focus fits:
            validation, boundary, error handling, workflow, notification,
            or "general" for plain functional behaviour.

The TF-IDF vocabulary (word + character n-grams) is fitted on the labeled
texts *and* on unlabeled acceptance criteria from real user stories, so the
model also knows the vocabulary of story-style requirements (semi-supervised
feature learning).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion

from .datasets import HANDCRAFTED, is_gold
from .taxonomy import NONFUNCTIONAL

NFR_CLASSES = ("security", "performance", "usability", "reliability", "compliance", "operational")
FOCUS_CLASSES = ("validation", "boundary", "error_handling", "workflow", "notification")
GENERAL = "general"
FOCUS_THRESHOLD = 0.40  # minimum focus probability to replace "functional" with a focus category
SECONDARY_THRESHOLD = 0.20  # other categories above this are reported as secondary
GENERAL_SAMPLE = 150  # functional rows from other datasets added to the focus head as "general"


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+(?:[.,]\d+)?", " 9 ", text)  # all numbers look alike to the model
    text = re.sub(r"[\"'“”‘’]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_features() -> FeatureUnion:
    return FeatureUnion(
        [
            ("words", TfidfVectorizer(preprocessor=preprocess, ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("chars", TfidfVectorizer(preprocessor=preprocess, analyzer="char_wb", ngram_range=(3, 5),
                                      sublinear_tf=True, min_df=2, max_features=80000)),
        ]
    )


def _head(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000)


@dataclass
class Prediction:
    category: str
    confidence: float
    secondary: list[tuple[str, float]]  # other plausible categories, most likely first
    requirement_type: str  # "functional" or the non-functional category


@dataclass
class EvaluationReport:
    accuracy: float
    macro_f1: float
    per_class: dict
    n_samples: int
    n_train_extra: int
    folds: int

    def as_text(self) -> str:
        lines = [
            f"Evaluated on {self.n_samples} gold-labeled requirements ({self.folds}-fold cross-validation);",
            f"{self.n_train_extra} weakly labeled rows were added to every training fold.",
            f"Accuracy: {self.accuracy:.3f}  |  Macro F1: {self.macro_f1:.3f}",
            "",
            f"{'category':<16}{'precision':>10}{'recall':>8}{'f1':>8}{'support':>9}",
        ]
        for name, m in self.per_class.items():
            if isinstance(m, dict) and name not in ("macro avg", "weighted avg"):
                lines.append(f"{name:<16}{m['precision']:>10.2f}{m['recall']:>8.2f}{m['f1-score']:>8.2f}{int(m['support']):>9}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {"accuracy": self.accuracy, "macro_f1": self.macro_f1, "n_samples": self.n_samples, "folds": self.folds}


class RequirementClassifier:
    """
    Parameters
    ----------
    kind_weight:   how much the functional-vs-non-functional head (trained on all
                   rows, including FR/NFR-only datasets) counts when deciding whether
                   a requirement is functional; the rest comes from the type head.
    weak_weight:   sample weight of rows from weakly labeled datasets.
    """

    def __init__(self, C: float = 8.0, seed: int = 42, kind_weight: float = 0.0, weak_weight: float = 0.3,
                 general_sample: int = GENERAL_SAMPLE):
        self.C = C
        self.seed = seed
        self.kind_weight = kind_weight
        self.weak_weight = weak_weight
        self.general_sample = general_sample
        self.features = build_features()
        self.type_head = _head(C)  # functional + the non-functional categories
        self.kind_head = _head(C)  # functional vs non-functional
        self.focus_head = _head(C)

    def _params(self) -> dict:
        return dict(C=self.C, seed=self.seed, kind_weight=self.kind_weight, weak_weight=self.weak_weight,
                    general_sample=self.general_sample)

    # ------------------------------------------------------------------ training
    def _weights(self, rows: pd.DataFrame) -> np.ndarray:
        return np.where(is_gold(rows) | (rows["category"] == NONFUNCTIONAL), 1.0, self.weak_weight)

    def _focus_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        focus = data[data["category"].isin(FOCUS_CLASSES)]
        functional = data[data["category"] == "functional"]
        handmade = functional[functional["source"] == HANDCRAFTED.name]
        others = functional[(functional["source"] != HANDCRAFTED.name) & is_gold(functional)]
        others = others.sample(min(self.general_sample, len(others)), random_state=self.seed)
        return pd.concat([focus, handmade.assign(category=GENERAL), others.assign(category=GENERAL)])

    def fit(self, data: pd.DataFrame, unlabeled: list[str] | None = None) -> "RequirementClassifier":
        """`data` needs columns text, category and source (see datasets.load_training_data)."""
        self.features.fit(pd.concat([data["text"], pd.Series(unlabeled or [], dtype=object)]))

        typed = data[data["category"] != NONFUNCTIONAL]
        type_labels = typed["category"].where(typed["category"].isin(NFR_CLASSES), "functional")
        self.type_head.fit(self.features.transform(typed["text"]), type_labels, sample_weight=self._weights(typed))

        if self.kind_weight > 0:
            is_nfr = data["category"].isin(NFR_CLASSES + (NONFUNCTIONAL,))
            self.kind_head.fit(self.features.transform(data["text"]), np.where(is_nfr, NONFUNCTIONAL, "functional"),
                               sample_weight=np.where(data["source"] == "fr_nfr_requirements.csv", self.weak_weight, 1.0))

        focus = self._focus_rows(data)
        self.focus_head.fit(self.features.transform(focus["text"]), focus["category"])
        return self

    # ---------------------------------------------------------------- inference
    def predict(self, texts: list[str]) -> list[Prediction]:
        if not texts:
            return []
        X = self.features.transform(texts)
        type_p = self.type_head.predict_proba(X)
        type_cls = list(self.type_head.classes_)
        t_func = type_p[:, type_cls.index("functional")]
        if self.kind_weight > 0:
            kind_p = self.kind_head.predict_proba(X)[:, list(self.kind_head.classes_).index("functional")]
            p_funcs = (1 - self.kind_weight) * t_func + self.kind_weight * kind_p
        else:
            p_funcs = t_func
        nfr_cls = [c for c in type_cls if c != "functional"]
        nfr_idx = [type_cls.index(c) for c in nfr_cls]
        nfr = type_p[:, nfr_idx]
        nfr = nfr / np.clip(nfr.sum(axis=1, keepdims=True), 1e-9, None)
        focus = self.focus_head.predict_proba(X)
        focus_cls = list(self.focus_head.classes_)

        out = []
        for p_func, np_, fp in zip(p_funcs, nfr, focus):
            scores = {c: (1 - p_func) * p for c, p in zip(nfr_cls, np_)}
            for c, p in zip(focus_cls, fp):
                scores["functional" if c == GENERAL else c] = p_func * p

            if p_func >= 0.5:
                req_type = "functional"
                best = int(np.argmax(fp))
                if focus_cls[best] != GENERAL and fp[best] >= FOCUS_THRESHOLD:
                    category, confidence = focus_cls[best], float(fp[best] * p_func)
                else:
                    category, confidence = "functional", float(p_func)
            else:
                best = int(np.argmax(np_))
                req_type = category = nfr_cls[best]
                confidence = float(np_[best] * (1 - p_func))

            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            secondary = [(c, float(p)) for c, p in ranked if c != category and p >= SECONDARY_THRESHOLD]
            # Focus areas count as secondary on their own probability too (they refine "functional").
            for c, p in zip(focus_cls, fp):
                if c not in (GENERAL, category) and p >= SECONDARY_THRESHOLD and c not in dict(secondary):
                    secondary.append((c, float(p)))
            out.append(Prediction(category, confidence, secondary, req_type))
        return out

    # --------------------------------------------------------------- evaluation
    def evaluate(self, data: pd.DataFrame, unlabeled: list[str] | None = None, folds: int = 5) -> EvaluationReport:
        """Cross-validate on the gold rows; weak rows are always part of the training folds."""
        gold_mask = is_gold(data)
        gold, weak = data[gold_mask], data[~gold_mask]
        folds = max(2, min(folds, int(gold["category"].value_counts().min())))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        pred = pd.Series(index=gold.index, dtype=object)
        for train_idx, test_idx in cv.split(gold["text"], gold["category"]):
            train = pd.concat([weak, gold.iloc[train_idx]])
            model = RequirementClassifier(**self._params()).fit(train, unlabeled)
            test = gold.iloc[test_idx]
            pred.loc[test.index] = [p.category for p in model.predict(test["text"].tolist())]
        return EvaluationReport(
            accuracy=accuracy_score(gold["category"], pred),
            macro_f1=f1_score(gold["category"], pred, average="macro"),
            per_class=classification_report(gold["category"], pred, output_dict=True, zero_division=0),
            n_samples=len(gold),
            n_train_extra=len(weak),
            folds=folds,
        )
