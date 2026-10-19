"""Two-stage requirement classifier.

Stage 1 (requirement type) is trained on every labeled requirement and tells
functional requirements apart from security, performance, usability,
reliability, compliance and operational ones.

Stage 2 (test focus) runs on functional requirements and decides which test
design focus fits best: validation, boundary, error handling, workflow,
notification - or "general" for a plain functional behaviour.

Both stages are TF-IDF (word + character n-grams) -> logistic regression.
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
from sklearn.pipeline import FeatureUnion, Pipeline

TYPE_CLASSES = ("functional", "security", "performance", "usability", "reliability", "compliance", "operational")
FOCUS_CLASSES = ("validation", "boundary", "error_handling", "workflow", "notification")
GENERAL = "general"
FOCUS_THRESHOLD = 0.40  # minimum stage-2 probability to replace "functional" with a focus category
SECONDARY_THRESHOLD = 0.20  # other categories above this are reported as secondary
GENERAL_SAMPLE = 150  # PROMISE functional rows added to stage 2 as "general" examples


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+(?:[.,]\d+)?", " 9 ", text)  # all numbers look alike to the model
    text = re.sub(r"[\"'“”‘’]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_pipeline(C: float = 8.0) -> Pipeline:
    features = FeatureUnion(
        [
            ("words", TfidfVectorizer(preprocessor=preprocess, ngram_range=(1, 2), sublinear_tf=True)),
            ("chars", TfidfVectorizer(preprocessor=preprocess, analyzer="char_wb", ngram_range=(3, 5),
                                      sublinear_tf=True, min_df=2, max_features=60000)),
        ]
    )
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    return Pipeline([("features", features), ("clf", clf)])


@dataclass
class Prediction:
    category: str
    confidence: float
    secondary: list[tuple[str, float]]  # other plausible categories, most likely first
    requirement_type: str


@dataclass
class EvaluationReport:
    accuracy: float
    macro_f1: float
    per_class: dict
    n_samples: int
    folds: int

    def as_text(self) -> str:
        lines = [
            f"Samples: {self.n_samples}  |  {self.folds}-fold cross-validation",
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
    def __init__(self, C: float = 8.0, seed: int = 42):
        self.C = C
        self.seed = seed
        self.type_model = build_pipeline(C)
        self.focus_model = build_pipeline(C)

    # ------------------------------------------------------------------ training
    def _stage_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        type_data = data.assign(category=data["category"].where(data["category"].isin(TYPE_CLASSES), "functional"))

        focus = data[data["category"].isin(FOCUS_CLASSES)]
        handmade_general = data[(data["category"] == "functional") & (data["source"] != "promise_requirements.csv")]
        promise_general = data[(data["category"] == "functional") & (data["source"] == "promise_requirements.csv")]
        promise_general = promise_general.sample(min(GENERAL_SAMPLE, len(promise_general)), random_state=self.seed)
        general = pd.concat([handmade_general, promise_general]).assign(category=GENERAL)
        return type_data, pd.concat([focus, general], ignore_index=True)

    def fit(self, data: pd.DataFrame) -> "RequirementClassifier":
        """`data` needs columns text, category and source (see datasets.load_training_data)."""
        type_data, focus_data = self._stage_data(data)
        self.type_model.fit(type_data["text"], type_data["category"])
        self.focus_model.fit(focus_data["text"], focus_data["category"])
        return self

    # ---------------------------------------------------------------- inference
    def predict(self, texts: list[str]) -> list[Prediction]:
        if not texts:
            return []
        type_probs = self.type_model.predict_proba(texts)
        focus_probs = self.focus_model.predict_proba(texts)
        type_classes = list(self.type_model.classes_)
        focus_classes = list(self.focus_model.classes_)

        out = []
        for tp, fp in zip(type_probs, focus_probs):
            scores = dict(zip(type_classes, tp))
            functional = scores.pop("functional", 0.0)
            # Split the functional probability mass across the stage-2 focus classes.
            for cls, p in zip(focus_classes, fp):
                scores["functional" if cls == GENERAL else cls] = functional * p

            req_type = type_classes[int(np.argmax(tp))]
            best_focus_idx = int(np.argmax(fp))
            best_focus = focus_classes[best_focus_idx]
            if req_type == "functional":
                if best_focus != GENERAL and fp[best_focus_idx] >= FOCUS_THRESHOLD:
                    category, confidence = best_focus, float(fp[best_focus_idx])
                else:
                    category, confidence = "functional", float(functional)
            else:
                category, confidence = req_type, float(max(tp))

            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            secondary = [(c, float(p)) for c, p in ranked if c != category and p >= SECONDARY_THRESHOLD]
            # Focus classes are reported as secondary on their own stage-2 probability too.
            for cls, p in zip(focus_classes, fp):
                if cls not in (GENERAL, category) and p >= SECONDARY_THRESHOLD and cls not in dict(secondary):
                    secondary.append((cls, float(p)))
            out.append(Prediction(category, confidence, secondary, req_type))
        return out

    # --------------------------------------------------------------- evaluation
    def evaluate(self, data: pd.DataFrame, folds: int = 5) -> EvaluationReport:
        """Cross-validate the whole two-stage model on `data`."""
        counts = data["category"].value_counts()
        folds = max(2, min(folds, int(counts.min())))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        pred = pd.Series(index=data.index, dtype=object)
        for train_idx, test_idx in cv.split(data["text"], data["category"]):
            model = RequirementClassifier(self.C, self.seed).fit(data.iloc[train_idx])
            test = data.iloc[test_idx]
            pred.loc[test.index] = [p.category for p in model.predict(test["text"].tolist())]
        return EvaluationReport(
            accuracy=accuracy_score(data["category"], pred),
            macro_f1=f1_score(data["category"], pred, average="macro"),
            per_class=classification_report(data["category"], pred, output_dict=True, zero_division=0),
            n_samples=len(data),
            folds=folds,
        )
