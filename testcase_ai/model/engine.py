"""TestCaseModel: the trained model end to end.

story -> acceptance criteria -> (classifier + extractor) -> test design -> ranked test cases
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from ..models import CriterionInsight, Priority, TestCase, TestStep, TestSuite, UserStory
from .classifier import FOCUS_CLASSES, RequirementClassifier
from .datasets import AI_CLASSIFIED, DEFAULT_WEAK, is_gold, load_training_data
from .extractor import CriterionFacts, analyze_criterion, analyze_story, extract_acceptance_criteria
from .taxonomy import CATEGORY_RISK, HIGH_RISK_WORDS
from .test_design import Draft, design

MODEL_DIR = Path(os.getenv("TESTCASE_MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))
MODEL_PATH = MODEL_DIR / "testcase_model.joblib"
FEEDBACK_PATH = MODEL_DIR / "feedback.csv"
EXTRA_DATA_DIR = MODEL_DIR / "extra_data"
LOW_CONFIDENCE = 0.5
MODEL_VERSION = 3
MAX_TECHNIQUES = 3


@dataclass
class CriterionAnalysis:
    facts: CriterionFacts
    category: str
    confidence: float
    requirement_type: str
    secondary: list[str]
    techniques: list[str]

    def insight(self) -> CriterionInsight:
        f = self.facts
        return CriterionInsight(
            text=f.text, category=self.category, confidence=round(self.confidence, 3),
            requirement_type=self.requirement_type, secondary=self.secondary, techniques=self.techniques,
            limits=[l.describe() for l in f.limits], fields=f.fields, roles=f.roles, states=f.states,
        )


class TestCaseModel:
    __test__ = False  # not a pytest test class

    name = "TestCaseAI model"

    def __init__(self, classifier: RequirementClassifier, metadata: dict | None = None):
        self.classifier = classifier
        self.metadata = metadata or {}

    # ------------------------------------------------------------- lifecycle
    @classmethod
    def train(cls, extra_data: list[str | Path] | None = None, feedback: str | Path | None = FEEDBACK_PATH,
              evaluate: bool = True, include_ai_labeled: bool = False) -> "TestCaseModel":
        if extra_data is None:
            extra_data = sorted(p for p in EXTRA_DATA_DIR.glob("*") if p.suffix.lower() in (".csv", ".xlsx", ".xls", ".txt"))
        weak = DEFAULT_WEAK + ((AI_CLASSIFIED,) if include_ai_labeled else ())
        data = load_training_data(extra_data, feedback, weak)
        classifier = RequirementClassifier().fit(data)
        metadata = {
            "version": MODEL_VERSION,
            "sklearn": sklearn.__version__,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_samples": len(data),
            "n_gold": int(is_gold(data).sum()),
            "class_counts": data["category"].value_counts().to_dict(),
            "sources": sorted(data["source"].unique()),
        }
        if evaluate:
            report = RequirementClassifier().evaluate(data)
            metadata["evaluation"] = report.as_dict()
            metadata["evaluation_report"] = report.as_text()
        return cls(classifier, metadata)

    def save(self, path: str | Path = MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"classifier": self.classifier, "metadata": self.metadata}, path)
        return path

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "TestCaseModel":
        bundle = joblib.load(path)
        return cls(bundle["classifier"], bundle["metadata"])

    @classmethod
    def load_or_train(cls, path: str | Path = MODEL_PATH) -> "TestCaseModel":
        """Load the saved model; (re)train it when missing or built with another sklearn/model version."""
        path = Path(path)
        if path.exists():
            try:
                model = cls.load(path)
                meta = model.metadata
                if meta.get("version") == MODEL_VERSION and meta.get("sklearn") == sklearn.__version__:
                    return model
            except Exception:  # corrupt or incompatible pickle: retrain below
                pass
        model = cls.train(evaluate=False)
        model.save(path)
        return model

    # -------------------------------------------------------------- feedback
    @staticmethod
    def add_feedback(rows: list[tuple[str, str]], path: str | Path = FEEDBACK_PATH) -> int:
        """Append (text, category) corrections; they are used the next time the model is trained."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if new:
                writer.writerow(["text", "category"])
            writer.writerows(rows)
        return len(rows)

    # ------------------------------------------------------------- inference
    def analyze(self, story: UserStory) -> list[CriterionAnalysis]:
        criteria = extract_acceptance_criteria(story)
        predictions = self.classifier.predict(criteria)
        analyses = []
        for text, pred in zip(criteria, predictions):
            facts = analyze_criterion(text)
            secondary = [c for c, _ in pred.secondary]
            techniques = [pred.category]
            # Model-suggested secondary focus areas.
            techniques += [c for c in secondary if c in FOCUS_CLASSES or c in ("security", "performance")][:1]
            # When the model is unsure, also test from the runner-up category's angle.
            if pred.confidence < LOW_CONFIDENCE and secondary:
                techniques.append(secondary[0])
            # Facts found by the extractor add techniques the text clearly calls for.
            if facts.limits and not facts.time_limits:
                techniques.append("boundary")
            if facts.states and len(facts.states) >= 2:
                techniques.append("workflow")
            if facts.channels and pred.category in ("functional", "workflow") and "notification" in secondary:
                techniques.append("notification")
            techniques = list(dict.fromkeys(techniques))[:MAX_TECHNIQUES]
            analyses.append(CriterionAnalysis(facts, pred.category, pred.confidence, pred.requirement_type,
                                              secondary, techniques))
        return analyses

    def _priority(self, draft: Draft, category: str, text: str, story: UserStory) -> Priority:
        score = CATEGORY_RISK.get(category, 2) + draft.priority_offset
        low = text.lower()
        if any(w in low for w in HIGH_RISK_WORDS):
            score += 1
        story_priority = (story.priority or "").lower()
        if story_priority in ("highest", "high", "critical", "blocker"):
            score += 1
        elif story_priority in ("low", "lowest", "trivial"):
            score -= 1
        return Priority.HIGH if score >= 4 else Priority.MEDIUM if score >= 2 else Priority.LOW

    def generate(self, story: UserStory, max_cases: int = 10) -> TestSuite:
        ctx = analyze_story(story)
        analyses = self.analyze(story)

        per_criterion: list[list[tuple[Draft, str, CriterionAnalysis]]] = []
        for a in analyses:
            drafts: list[tuple[Draft, str, CriterionAnalysis]] = []
            for i, technique in enumerate(a.techniques):
                designed = design(technique, a.facts, ctx)
                # Main technique contributes all its drafts, extra techniques only their first one.
                drafts += [(d, technique, a) for d in (designed if i == 0 else designed[:1])]
            seen, unique = set(), []
            for d, t, an in drafts:
                if d.title.lower() not in seen:
                    seen.add(d.title.lower())
                    unique.append((d, t, an))
            per_criterion.append(unique)

        # Round-robin over criteria so every criterion gets its most important tests first.
        ordered: list[tuple[Draft, str, CriterionAnalysis]] = []
        depth = 0
        while len(ordered) < max_cases and any(depth < len(c) for c in per_criterion):
            for drafts in per_criterion:
                if depth < len(drafts) and len(ordered) < max_cases:
                    ordered.append(drafts[depth])
            depth += 1

        cases = []
        for n, (d, technique, a) in enumerate(ordered, 1):
            cases.append(TestCase(
                id=f"{story.key}-TC{n:02d}",
                title=d.title,
                type=d.type,
                priority=self._priority(d, technique, a.facts.text, story),
                preconditions=d.preconditions,
                test_data=d.test_data,
                steps=[TestStep(action=s, expected_result=e) for s, e in d.steps],
                expected_result=d.expected,
                acceptance_criterion=a.facts.text,
                technique=d.technique,
            ))

        total = sum(len(c) for c in per_criterion)
        covered = len({c.acceptance_criterion for c in cases})
        categories = ", ".join(sorted({a.category for a in analyses}))
        notes = (
            f"Analysed {len(analyses)} criteria (categories: {categories}); "
            f"covered {covered}/{len(analyses)} with {len(cases)} of {total} designed test cases."
        )
        if len(cases) < total:
            notes += " Increase the maximum number of test cases to include the rest."
        return TestSuite(story=story, test_cases=cases, insights=[a.insight() for a in analyses],
                         coverage_notes=notes, generator=self.name)
