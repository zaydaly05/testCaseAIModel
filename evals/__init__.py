"""Cross-model review: DeepEval scores the generated test cases against the Jira criteria."""

import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")  # DeepEval reports usage unless told not to
# Rate-limited judges retry for a while; the default per-task timeout cancels them (and drops their cases).
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "1800")
