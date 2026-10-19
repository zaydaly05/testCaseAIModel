"""Portable model: the trained classifier as plain data, run with plain Python.

`export_portable` turns a fitted `RequirementClassifier` into a JSON-able dict
(vocabularies, IDF weights, logistic-regression coefficients). `PortableClassifier`
reproduces scikit-learn's TF-IDF + logistic regression maths without numpy,
scipy or scikit-learn, so the model can run where those are too big to install
(for example a Vercel serverless function).
"""

from __future__ import annotations

import base64
import gzip
import json
import math
import re
from array import array
from collections import Counter
from pathlib import Path

from .heads import Prediction, combine, preprocess

FORMAT_VERSION = 1
_TOKEN = re.compile(r"(?u)\b\w\w+\b")  # scikit-learn's default token_pattern
_WHITE_SPACES = re.compile(r"\s\s+")


def _pack(values) -> str:
    """Float list -> base64 float32 bytes (compact, and fast to load with the stdlib)."""
    return base64.b64encode(array("f", (float(v) for v in values)).tobytes()).decode("ascii")


def _unpack(text: str) -> array:
    return array("f", base64.b64decode(text))


# ----------------------------------------------------------------------------- export

def _export_vectorizer(vec) -> dict:
    terms = [""] * len(vec.vocabulary_)
    for term, idx in vec.vocabulary_.items():
        terms[idx] = term
    return {
        "analyzer": vec.analyzer,
        "ngram_range": list(vec.ngram_range),
        "sublinear_tf": bool(vec.sublinear_tf),
        "terms": terms,
        "idf": _pack(vec.idf_),
    }


def _export_head(features, head) -> dict:
    vectorizers = [_export_vectorizer(vec) for _, vec in features.transformer_list]
    return {
        "classes": [str(c) for c in head.classes_],
        "vectorizers": vectorizers,
        "coef": [_pack(row) for row in head.coef_],
        "intercept": [float(v) for v in head.intercept_],
    }


def export_portable(classifier, metadata: dict | None = None) -> dict:
    data = {
        "format": FORMAT_VERSION,
        "kind_weight": classifier.kind_weight,
        "type": _export_head(classifier.type_features, classifier.type_head),
        "focus": _export_head(classifier.focus_features, classifier.focus_head),
        "metadata": {k: v for k, v in (metadata or {}).items() if k != "evaluation_report"},
    }
    if classifier.kind_weight > 0:
        data["kind"] = _export_head(classifier.kind_features, classifier.kind_head)
    return data


def save_portable(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))
    return path


# ----------------------------------------------------------------------------- inference

class _Vectorizer:
    def __init__(self, spec: dict, offset: int):
        self.analyzer = spec["analyzer"]
        self.min_n, self.max_n = spec["ngram_range"]
        self.sublinear = spec["sublinear_tf"]
        self.index = {term: i for i, term in enumerate(spec["terms"])}
        self.idf = _unpack(spec["idf"])
        self.offset = offset  # position of this vectorizer's block in the joined feature vector

    def _ngrams(self, text: str) -> list[str]:
        text = preprocess(text)
        if self.analyzer == "char_wb":
            text = _WHITE_SPACES.sub(" ", text)
            grams = []
            for w in text.split():
                w = " " + w + " "
                for n in range(self.min_n, self.max_n + 1):
                    offset = 0
                    grams.append(w[offset:offset + n])
                    while offset + n < len(w):
                        offset += 1
                        grams.append(w[offset:offset + n])
                    if offset == 0:
                        break
            return grams
        tokens = _TOKEN.findall(text)
        grams = list(tokens) if self.min_n == 1 else []
        for n in range(max(self.min_n, 2), min(self.max_n + 1, len(tokens) + 1)):
            grams.extend(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
        return grams

    def transform(self, text: str) -> dict[int, float]:
        counts = Counter(i for g in self._ngrams(text) if (i := self.index.get(g)) is not None)
        weights = {}
        for i, tf in counts.items():
            tf = 1.0 + math.log(tf) if self.sublinear else float(tf)
            weights[i] = tf * self.idf[i]
        norm = math.sqrt(sum(v * v for v in weights.values()))
        return {self.offset + i: v / norm for i, v in weights.items()} if norm else {}


class _Head:
    def __init__(self, spec: dict):
        self.classes = spec["classes"]
        self.vectorizers = []
        offset = 0
        for v in spec["vectorizers"]:
            self.vectorizers.append(_Vectorizer(v, offset))
            offset += len(v["terms"])
        self.coef = [_unpack(row) for row in spec["coef"]]
        self.intercept = spec["intercept"]

    def predict_proba(self, text: str) -> list[float]:
        x: dict[int, float] = {}
        for vec in self.vectorizers:
            x.update(vec.transform(text))
        scores = [b + sum(row[i] * v for i, v in x.items()) for row, b in zip(self.coef, self.intercept)]
        if len(scores) == 1:  # binary logistic regression: one score for classes[1]
            p = 1.0 / (1.0 + math.exp(-scores[0]))
            return [1.0 - p, p]
        top = max(scores)
        exps = [math.exp(s - top) for s in scores]
        total = sum(exps)
        return [e / total for e in exps]


class PortableClassifier:
    """Same `predict` interface as `RequirementClassifier`, no heavy dependencies."""

    def __init__(self, data: dict):
        if data.get("format") != FORMAT_VERSION:
            raise ValueError(f"Unsupported portable model format: {data.get('format')}")
        self.kind_weight = data["kind_weight"]
        self.type_head = _Head(data["type"])
        self.focus_head = _Head(data["focus"])
        self.kind_head = _Head(data["kind"]) if "kind" in data else None
        self.metadata = data.get("metadata", {})

    @classmethod
    def load(cls, path: str | Path) -> PortableClassifier:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def predict(self, texts: list[str]) -> list[Prediction]:
        if not texts:
            return []
        type_p = [self.type_head.predict_proba(t) for t in texts]
        focus_p = [self.focus_head.predict_proba(t) for t in texts]
        kind_func = None
        if self.kind_head is not None and self.kind_weight > 0:
            f = self.kind_head.classes.index("functional")
            kind_func = [self.kind_head.predict_proba(t)[f] for t in texts]
        return combine(self.type_head.classes, type_p, self.focus_head.classes, focus_p,
                       kind_func, self.kind_weight)
