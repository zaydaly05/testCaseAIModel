"""Shared, dependency-free part of the classifier.

Both the scikit-learn classifier (training, local use) and the portable
classifier (serverless deployment) produce per-head class probabilities;
`combine` turns them into one `Prediction` per text using plain Python.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NFR_CLASSES = ("security", "performance", "usability", "reliability", "compliance", "operational")
FOCUS_CLASSES = ("validation", "boundary", "error_handling", "workflow", "notification")
GENERAL = "general"
FOCUS_THRESHOLD = 0.40  # minimum focus probability to replace "functional" with a focus category
SECONDARY_THRESHOLD = 0.20  # other categories above this are reported as secondary


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+(?:[.,]\d+)?", " 9 ", text)  # all numbers look alike to the model
    text = re.sub(r"[\"'“”‘’]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Prediction:
    category: str
    confidence: float
    secondary: list[tuple[str, float]]  # other plausible categories, most likely first
    requirement_type: str  # "functional" or the non-functional category


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def combine(type_classes: list[str], type_probs: list[list[float]],
            focus_classes: list[str], focus_probs: list[list[float]],
            kind_functional: list[float] | None = None, kind_weight: float = 0.0) -> list[Prediction]:
    """Merge the heads' probabilities (one row per text) into predictions."""
    f_idx = type_classes.index("functional")
    nfr_cls = [c for c in type_classes if c != "functional"]
    nfr_idx = [type_classes.index(c) for c in nfr_cls]

    out = []
    for row, (tp, fp) in enumerate(zip(type_probs, focus_probs)):
        p_func = tp[f_idx]
        if kind_functional is not None and kind_weight > 0:
            p_func = (1 - kind_weight) * p_func + kind_weight * kind_functional[row]
        nfr = [tp[i] for i in nfr_idx]
        total = max(sum(nfr), 1e-9)
        nfr = [p / total for p in nfr]

        scores = {c: (1 - p_func) * p for c, p in zip(nfr_cls, nfr)}
        for c, p in zip(focus_classes, fp):
            scores["functional" if c == GENERAL else c] = p_func * p

        if p_func >= 0.5:
            req_type = "functional"
            best = _argmax(fp)
            if focus_classes[best] != GENERAL and fp[best] >= FOCUS_THRESHOLD:
                category, confidence = focus_classes[best], fp[best] * p_func
            else:
                category, confidence = "functional", p_func
        else:
            best = _argmax(nfr)
            req_type = category = nfr_cls[best]
            confidence = nfr[best] * (1 - p_func)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        secondary = [(c, float(p)) for c, p in ranked if c != category and p >= SECONDARY_THRESHOLD]
        # Focus areas count as secondary on their own probability too (they refine "functional").
        for c, p in zip(focus_classes, fp):
            if c not in (GENERAL, category) and p >= SECONDARY_THRESHOLD and c not in dict(secondary):
                secondary.append((c, float(p)))
        out.append(Prediction(category, float(confidence), secondary, req_type))
    return out
