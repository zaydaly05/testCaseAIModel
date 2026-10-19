"""Train, evaluate and save the test case model.

    python -m testcase_ai.model.train                       # bundled datasets + saved feedback
    python -m testcase_ai.model.train --data my_reqs.csv     # add your own labeled data (CSV/XLSX/PROMISE .txt)
"""

from __future__ import annotations

import argparse

from .engine import FEEDBACK_PATH, MODEL_PATH, TestCaseModel


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="*", default=None,
                        help="Extra labeled datasets (text + category/label columns); default: models/extra_data/*")
    parser.add_argument("--feedback", default=str(FEEDBACK_PATH), help="Corrections file collected from the UI")
    parser.add_argument("--out", default=str(MODEL_PATH), help="Where to save the trained model")
    parser.add_argument("--no-eval", action="store_true", help="Skip cross-validation")
    parser.add_argument("--with-ai-labeled", action="store_true",
                        help="Also train on the LLM-classified requirements (lowered scores in our tests)")
    args = parser.parse_args(argv)

    model = TestCaseModel.train(args.data, args.feedback, evaluate=not args.no_eval,
                                include_ai_labeled=args.with_ai_labeled)
    path = model.save(args.out)
    meta = model.metadata
    print(f"Trained on {meta['n_samples']} labeled requirements from: {', '.join(meta['sources'])}")
    if "evaluation_report" in meta:
        print(meta["evaluation_report"])
    print(f"Saved model to {path}")


if __name__ == "__main__":
    main()
