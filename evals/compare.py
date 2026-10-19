"""Compare the own model with a baseline LLM on real stories, judged by DeepEval.

    python -m evals.compare                                  # sample stories, own model only
    python -m evals.compare --baseline groq:qwen/qwen3.8-27b --limit 10
    python -m evals.compare --stories my_stories.json --fail-under 0.7   # quality gate (exit code 1)

Writes evals/reports/report-<time>.md and .json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean

from testcase_ai import UserStory

from .review import OWN_MODEL, CriterionReview, judge, llm, review

ROOT = Path(__file__).resolve().parent.parent


def summary(reviews: list[CriterionReview]) -> dict[str, dict[str, float]]:
    """generator -> metric -> mean score, plus the share of criteria where every metric passed."""
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    passed: dict[str, list[bool]] = defaultdict(list)
    for r in reviews:
        for s in r.scores:
            scores[r.generator][s.metric].append(s.score)
        passed[r.generator].append(r.passed)
    return {g: {**{m: round(mean(v), 3) for m, v in ms.items()}, "pass rate": round(mean(passed[g]), 3)}
            for g, ms in scores.items()}


def quality(per_metric: dict[str, float]) -> float:
    """One number per generator: the mean of all its metric scores."""
    vals = [v for m, v in per_metric.items() if m != "pass rate"]
    return round(mean(vals), 3) if vals else 0.0


def markdown(reviews: list[CriterionReview], judge_name: str) -> str:
    table = summary(reviews)
    gens = list(table)
    metric_names = sorted({m for t in table.values() for m in t if m != "pass rate"})
    out = [f"# Cross-model review ({datetime.now():%Y-%m-%d %H:%M})", "",
           f"Judge: `{judge_name}`. Scores 0-1, higher is better (Hallucination 1 = nothing contradicts the story).", "",
           "| Metric | " + " | ".join(gens) + " |", "|---|" + "---|" * len(gens)]
    for m in [*metric_names, "pass rate"]:
        out.append(f"| {m} | " + " | ".join(f"{table[g][m]:.2f}" if m in table[g] else "—" for g in gens) + " |")
    out.append("| **quality** | " + " | ".join(f"**{quality(table[g]):.2f}**" for g in gens) + " |")
    errors = [e for r in reviews for e in r.errors]
    if errors:
        out += ["", f"**{len(errors)} judge calls failed and are not scored** (first: {errors[0]}). "
                "The scores above are incomplete: rerun later or with --throttle."]
    out += ["", "## Failed criteria", ""]
    failed = [r for r in reviews if not r.passed]
    for r in failed:
        out.append(f"### {r.story} · {r.generator}\n\n> {r.criterion}\n")
        out += [f"- **{s.metric}** {s.score:.2f}: {s.reason}" for s in r.scores if not s.passed]
        out += [f"- judge error, {e}" for e in r.errors]
        out.append("")
    if not failed:
        out.append("None.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stories", type=Path, default=ROOT / "samples" / "sample_stories.json")
    p.add_argument("--limit", type=int, default=5, help="number of stories (0 = all)")
    p.add_argument("--baseline", help='the LLM to compare with, "provider:model"; omit to review the own model only')
    p.add_argument("--threshold", type=float, default=0.7, help="pass mark of each metric")
    p.add_argument("--fail-under", type=float, help="exit with code 1 if the own model's quality is below this")
    p.add_argument("--throttle", type=float, default=0,
                   help="seconds between criteria, judged one at a time (free tiers with a requests-per-minute limit)")
    p.add_argument("--verbose", action="store_true", help="print DeepEval's own per-case output")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    judge_model = judge()
    if judge_model is None:
        print("No judge LLM. Set GROQ_API_KEY (free: https://console.groq.com/keys) or GEMINI_API_KEY "
              "in .env; EVAL_JUDGE picks the model (e.g. groq:openai/gpt-oss-120b).", file=sys.stderr)
        return 2
    stories = [UserStory(**s) for s in json.loads(a.stories.read_text(encoding="utf-8"))]
    stories = stories[: a.limit] if a.limit else stories
    reviews = review(stories, judge_model, llm(a.baseline) if a.baseline else None,
                     threshold=a.threshold, quiet=not a.verbose, throttle=a.throttle)

    report = markdown(reviews, judge_model.get_model_name())
    folder = ROOT / "evals" / "reports"
    folder.mkdir(exist_ok=True)
    stem = folder / f"report-{datetime.now():%Y%m%d-%H%M%S}"
    stem.with_suffix(".md").write_text(report, encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps([asdict(r) for r in reviews], indent=2), encoding="utf-8")
    print(report.split("\n## Failed criteria")[0])
    print(f"\nFull report: {stem.with_suffix('.md')}")

    if any(r.errors for r in reviews):
        print("Some judge calls failed, so the scores are incomplete.", file=sys.stderr)
        return 3
    own = summary(reviews).get(OWN_MODEL)
    if a.fail_under is not None and own and quality(own) < a.fail_under:
        print(f"Quality gate failed: {quality(own):.2f} < {a.fail_under}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
