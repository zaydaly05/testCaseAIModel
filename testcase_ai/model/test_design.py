"""Test design engine: turns a classified criterion + extracted facts into test cases.

Each category maps to classic test design techniques (equivalence
partitioning, boundary value analysis, state transition testing, ...).
The engine returns drafts; engine.py numbers and prioritises them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import TestType
from .extractor import CriterionFacts, Limit, StoryContext, _fmt, _is_time_unit

FIELD_SAMPLES: dict[str, dict[str, list[str]]] = {
    "email": {"valid": ["jane.doe@example.com"], "invalid": ["jane.doe@", "janeexample.com", "jane doe@example.com"]},
    "password": {"valid": ["Str0ng!Passw0rd"], "invalid": ["password", "12345678", "short"]},
    "username": {"valid": ["jane_doe42"], "invalid": ["jane doe", "jane@doe", "ab"]},
    "phone": {"valid": ["+14155550123"], "invalid": ["415-ABC-0123", "123", "+1 (415) 555 01234567"]},
    "date": {"valid": ["15/03/2026"], "invalid": ["31/02/2026", "2026-13-01", "yesterday"]},
    "url": {"valid": ["https://example.com"], "invalid": ["example", "ftp//example.com", "http:/example"]},
    "postal_code": {"valid": ["94103"], "invalid": ["9410", "ABCDE", "941033"]},
    "card": {"valid": ["4111 1111 1111 1111"], "invalid": ["4111 1111 1111 1112", "1234", "abcd efgh ijkl mnop"]},
    "iban": {"valid": ["DE89 3704 0044 0532 0130 00"], "invalid": ["DE00 0000 0000 0000 0000 00", "DE89", "XX89 3704"]},
    "name": {"valid": ["Jane"], "invalid": ["J4ne", "<b>Jane</b>", "   "]},
    "quantity": {"valid": ["2"], "invalid": ["-1", "1.5", "abc"]},
    "file": {"valid": ["report.pdf (1 MB)"], "invalid": ["virus.exe", "empty.pdf (0 bytes)", "corrupt.pdf"]},
    "text": {"valid": ["A normal sentence."], "invalid": ["<script>alert(1)</script>", "   ", "emoji only 😀😀"]},
}
GENERIC_SAMPLES = {"valid": ["A valid value as described in the story"],
                   "invalid": ["Wrong format", "Unsupported characters (<>%$)", "Value of the wrong type"]}

SECURITY_PAYLOADS = ["<script>alert(1)</script>", "' OR '1'='1", "../../etc/passwd"]
BROWSERS = ["chrome", "firefox", "safari", "edge", "android", "ios", "windows", "linux", "mac"]


@dataclass
class Draft:
    title: str
    type: TestType
    technique: str
    steps: list[tuple[str, str]]
    expected: str
    preconditions: list[str] = field(default_factory=list)
    test_data: list[str] = field(default_factory=list)
    priority_offset: int = 0  # 0 = main test of the criterion, negative = less important


def _short(text: str, n: int = 70) -> str:
    text = text.strip().rstrip(".")
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def _actor_pre(ctx: StoryContext, facts: CriterionFacts) -> list[str]:
    pre = list(facts.given)
    if not any(w in " ".join(pre).lower() for w in ("logged", "signed", "login")):
        pre.insert(0, f"Signed in as a {facts.roles[0] if facts.roles else ctx.actor}")
    return pre


def _action(facts: CriterionFacts) -> str:
    return facts.when[0] if facts.when else f"Perform the behaviour: {_short(facts.text, 90)}"


def _outcome(facts: CriterionFacts) -> str:
    return "; ".join(facts.then) if facts.then else facts.text.rstrip(".")


# --------------------------------------------------------------------------- techniques

def design_functional(f: CriterionFacts, ctx: StoryContext) -> list[Draft]:
    pre = _actor_pre(ctx, f)
    steps = [(w, "Action is accepted") for w in f.when[:-1]] + [(_action(f), _outcome(f))]
    if f.ui_elements:
        steps.insert(0, (f"Locate the '{f.ui_elements[0]}' control", "It is visible and enabled"))
    drafts = [
        Draft(f"Happy path: {_short(f.then[0] if f.then else f.text)}", TestType.POSITIVE,
              "Use case / happy path", steps, _outcome(f), pre, ["Valid data as described in the story"]),
        Draft(f"Blocked when preconditions are not met: {_short(f.text, 55)}", TestType.NEGATIVE,
              "Negative testing",
              [(f"Break a precondition ({f.given[0]} is not true)" if f.given else
                "Use missing, invalid or unauthorised input", "Precondition is not satisfied"),
               (_action(f), "The action is refused with a clear message"),
               ("Check the stored data", "Nothing was created or changed")],
              "The system refuses the action gracefully and data stays unchanged", pre[1:] or pre,
              ["Invalid or missing input"], -1),
    ]
    if re.search(r"\b(add|create|save|update|edit|upload|delete|remove|store|submit|book|assign)\w*", f.text, re.I):
        drafts.append(Draft(f"Result persists after reload: {_short(f.text, 55)}", TestType.EDGE,
                            "Persistence check",
                            [(_action(f), _outcome(f)), ("Reload the page and sign in again", "Session is restored"),
                             ("Open the affected record", "The change is still present")],
                            "The outcome is stored permanently", pre, [], -1))
    return drafts


def design_validation(f: CriterionFacts, ctx: StoryContext) -> list[Draft]:
    pre = _actor_pre(ctx, f)
    fields = f.fields or [""]
    drafts = []
    for fld in fields[:2]:
        samples = FIELD_SAMPLES.get(fld, GENERIC_SAMPLES)
        label = fld.replace("_", " ") if fld else "the field"
        drafts.append(Draft(f"Valid {label} is accepted", TestType.POSITIVE, "Equivalence partitioning (valid class)",
                            [(f"Enter {label}: {samples['valid'][0]}", "No validation error is shown"),
                             ("Submit the form", "The data is accepted and saved")],
                            "Valid input passes validation", pre, samples["valid"]))
        drafts.append(Draft(f"Invalid {label} formats are rejected", TestType.NEGATIVE,
                            "Equivalence partitioning (invalid classes)",
                            [(f"Enter {label}: {v} and submit", "Submission is blocked with a field-level error")
                             for v in samples["invalid"]],
                            f"Every invalid {label} is rejected with a clear message and nothing is saved",
                            pre, samples["invalid"]))
    drafts.append(Draft(f"Required input left empty: {_short(f.text, 50)}", TestType.NEGATIVE,
                        "Equivalence partitioning (empty class)",
                        [("Leave the required field(s) empty", "Field is empty"),
                         ("Submit the form", "A 'required' message is shown next to the field"),
                         ("Enter only spaces and submit", "Whitespace-only input is treated as empty")],
                        "Empty or blank input is not accepted", pre, ["(empty)", "'   '"], -1))
    return drafts


def _boundary_values(limit: Limit) -> tuple[list[float], list[float]]:
    """Return (accepted values, rejected values) for boundary value analysis."""
    step = 1.0 if all(v is None or float(v).is_integer() for v in (limit.low, limit.high)) else 0.01
    lo, hi = limit.low, limit.high
    if limit.kind == "range":
        return [lo, lo + step, hi - step, hi], [lo - step, hi + step]
    if limit.kind == "min":
        return [lo, lo + step], [lo - step]
    if limit.kind == "min_exclusive":
        return [lo + step], [lo]
    if limit.kind == "max":
        return [hi - step, hi], [hi + step]
    if limit.kind == "max_exclusive":
        return [hi - step], [hi]
    return [lo], [lo - step]  # exact


def _render(value: float, unit: str) -> str:
    if value < 0:
        value = 0 if unit else value
    v = _fmt(value)
    if unit.startswith("char"):
        n = int(value)
        sample = ("Abcdefghij" * (n // 10 + 1))[:n]
        return f"{v} characters ('{sample[:12]}{'…' if n > 12 else ''}')"
    return f"{v} {unit}".strip()


def design_boundary(f: CriterionFacts, ctx: StoryContext) -> list[Draft]:
    pre = _actor_pre(ctx, f)
    limits = [l for l in f.limits if not (_is_time_unit(l.unit) and l.kind == "exact")] or f.limits
    if not limits:
        return [Draft(f"Boundary values: {_short(f.text, 60)}", TestType.EDGE, "Boundary value analysis",
                      [("Use the minimum allowed value", "Accepted"), ("Use the maximum allowed value", "Accepted"),
                       ("Use a value just below the minimum", "Rejected with a clear message"),
                       ("Use a value just above the maximum", "Rejected with a clear message")],
                      "Limits are enforced exactly at the boundaries", pre,
                      ["Exact limits must be confirmed with the product owner"])]
    drafts = []
    for limit in limits[:2]:
        accepted, rejected = _boundary_values(limit)
        unit = limit.unit
        drafts.append(Draft(f"Values on the boundary are accepted ({limit.describe()})", TestType.EDGE,
                            "Boundary value analysis",
                            [(f"Use {_render(v, unit)}", "Accepted") for v in accepted],
                            f"All values within {limit.describe()} are accepted", pre,
                            [_render(v, unit) for v in accepted]))
        drafts.append(Draft(f"Values outside the boundary are rejected ({limit.describe()})", TestType.NEGATIVE,
                            "Boundary value analysis",
                            [(f"Use {_render(v, unit)}", "Rejected with a message that states the limit")
                             for v in rejected],
                            f"Values outside {limit.describe()} are rejected and nothing is saved", pre,
                            [_render(v, unit) for v in rejected]))
    return drafts


def design_security(f: CriterionFacts, ctx: StoryContext) -> list[Draft]:
    low = f.text.lower()
    allowed = f.roles[0] if f.roles else ctx.actor
    drafts: list[Draft] = []
    access = bool(f.roles) or re.search(r"\b(only|role|permission|access|authori[sz]|admin|owner|cannot (?:view|see|access))\w*", low)
    if access:
        drafts += [
            Draft(f"Authorised {allowed} can access: {_short(f.text, 50)}", TestType.POSITIVE,
                  "Access control (allowed role)",
                  [(f"Sign in as a {allowed}", "Signed in"), (_action(f), "Access is granted")],
                  "The permitted user can perform the action", [f"A {allowed} account exists"]),
            Draft("Unauthorised user is denied", TestType.SECURITY, "Access control (forbidden role)",
                  [(f"Sign in as a user who is not a {allowed}", "Signed in"),
                   (_action(f), "Access is denied (403 / hidden control)"),
                   ("Call the same URL or API endpoint directly", "Request is rejected; no data is returned")],
                  "Users without permission cannot perform the action by any route",
                  ["An account without the required role exists"]),
        ]
    if re.search(r"\b(same|reveal|disclos|mask|plain text|encrypt|hash|log|expos|enumerat|unregistered)\w*", low):
        drafts.append(Draft(f"No sensitive information is disclosed: {_short(f.text, 45)}", TestType.SECURITY,
                            "Information disclosure check",
                            [("Perform the action with a valid / existing value", "Note the response, message and timing"),
                             ("Repeat with an invalid / non-existing value", "Response, message and timing are indistinguishable"),
                             ("Inspect page source, network responses and logs", "No secrets, personal data or internal details appear")],
                            _outcome(f), _actor_pre(ctx, f)[1:], ["An existing and a non-existing account"]))
    if access or re.search(r"\b(login|log in|sign in|authenticat|unauthenticated|guest)\w*", low):
        drafts.append(Draft("Signed-out access is blocked", TestType.SECURITY, "Authentication check",
                            [("Sign out and clear cookies", "Session ended"),
                             ("Open the protected page or endpoint directly", "Redirected to sign-in / 401 returned")],
                            "No protected content is reachable without authentication", [], [], -1))
    if re.search(r"\b(session|token|link|expire|once|logout|log out|lock)\w*", low):
        drafts.append(Draft("Expired, reused or locked credentials are rejected", TestType.SECURITY,
                            "Session / token handling",
                            [("Obtain a valid session, token or link", "It works"),
                             ("Use it after expiry, after logout or a second time", "Rejected; the user must authenticate again")],
                            "Expired or single-use credentials cannot be replayed", [], [], -1))
    if re.search(r"\b(input|sanitiz|inject|script|field|form|search|password|email)\w*", low):
        drafts.append(Draft("Malicious input is neutralised", TestType.SECURITY, "Injection testing",
                            [(f"Enter {p}", "Input is rejected or safely escaped") for p in SECURITY_PAYLOADS],
                            "No script runs and no data leaks", _actor_pre(ctx, f), SECURITY_PAYLOADS, -1))
    if not drafts:
        drafts.append(Draft(f"Security requirement holds: {_short(f.text, 50)}", TestType.SECURITY, "Security verification",
                            [(_action(f), _outcome(f)),
                             ("Attempt to bypass it (direct URL, modified request, another account)", "The attempt fails and is logged")],
                            _outcome(f), _actor_pre(ctx, f)))
    return drafts


def design_error_handling(f: CriterionFacts, ctx: StoryContext) -> list[Draft]:
    pre = _actor_pre(ctx, f)
    m = re.search(r"\b(?:if|when)\s+(.+?)(?:,|\bthen\b|\bthe (?:user|system|app|order|item)\b)", f.text, re.I)
    condition = m.group(1).strip() if m else f.text.rstrip(".")
    return [
        Draft(f"Error handling: {_short(condition, 60)}", TestType.NEGATIVE, "Error guessing / fault injection",
              [(f"Create the situation: {condition}", "Situation is in place"),
               ("Perform the action", "The action is handled safely and a clear, user-friendly message is shown"),
               ("Inspect the message", "It explains what happened and what to do; no stack traces")],
              _outcome(f), pre, [f"Trigger: {condition}"]),
        Draft("No partial changes after the failure", TestType.NEGATIVE, "Data integrity check",
              [(f"Create the situation: {condition}", "Error is shown"),
               ("Check records, balances and stock", "Nothing was created, charged or changed")],
              "The system stays consistent after the error", pre, [], -1),
        Draft("User can recover and retry", TestType.POSITIVE, "Recovery testing",
              [(f"Create the situation: {condition}", "Error is shown; entered data is kept"),
               ("Remove the cause and retry", "The action now succeeds")],
              "The user can complete the task after the problem is resolved", pre, [], -1),
    ]


def design_reliability(f: CriterionFacts, ctx: StoryContext) -> list[Draft]:
    low = f.text.lower()
    drafts = []
    if re.search(r"\b(backup|back up|restore|recover|loss)\w*", low):
        drafts.append(Draft("Backup can be restored without data loss", TestType.RELIABILITY, "Backup / restore test",
                            [("Create known test records", "Records saved"), ("Run or wait for the backup", "Backup completes"),
                             ("Restore the backup to a clean environment", "Restore completes within the agreed time"),
                             ("Compare the records", "All records are present and unchanged")],
                            _outcome(f), ["Access to a staging environment"]))
    if re.search(r"\b(fail|failover|crash|outage|down|standby|fault|offline|unavailable)\w*", low):
        drafts.append(Draft("Service survives a component failure", TestType.RELIABILITY, "Failover / fault injection",
                            [("Start a user transaction", "Transaction in progress"),
                             ("Stop the primary server or dependency", "Failure is injected"),
                             ("Continue using the system", "Requests are served or fail gracefully; no data loss")],
                            _outcome(f), ["Monitoring and a way to stop components in staging"]))
    time_limits = f.time_limits
    if re.search(r"\b(availab|uptime|24|99)\w*", low) or not drafts:
        drafts.append(Draft("Availability meets the target", TestType.RELIABILITY, "Availability monitoring",
                            [("Run a synthetic check every minute for the measurement period", "Checks recorded"),
                             ("Calculate uptime from the results", f"Uptime meets the target ({_outcome(f)})")],
                            _outcome(f), ["Uptime monitoring tool configured"],
                            [l.describe() for l in f.limits] or ["Target from the requirement"]))
    return drafts


def design_workflow(f: CriterionFacts, ctx: StoryContext) -> list[Draft]:
    pre = _actor_pre(ctx, f)
    trigger = _action(f)
    persist = Draft("State is kept after reload and re-login", TestType.EDGE, "Persistence check",
                    [(trigger, _outcome(f)), ("Sign out, sign in again and reopen it", "The same state / data is shown")],
                    "Workflow state is persisted", pre, [], -1)
    if len(f.states) < 2:
        return [
            Draft(f"Process step works: {_short(f.text, 55)}", TestType.POSITIVE, "Process / workflow testing",
                  [("Complete the preceding steps of the process", "Process is at the right step"),
                   (trigger, _outcome(f))],
                  _outcome(f), pre),
            persist,
            Draft("Step cannot be done out of order", TestType.NEGATIVE, "Process / workflow testing (invalid order)",
                  [("Skip the required earlier step(s)", "Process is at an earlier step"),
                   ("Attempt this step", "The step is unavailable or refused with a message")],
                  "The process order is enforced", pre, [], -1),
        ]
    src, dst = f.states[0], f.states[1]
    return [
        Draft(f"Valid transition {src} → {dst}", TestType.POSITIVE, "State transition testing (valid)",
              [(f"Prepare a record in {src}", f"Record shows {src}"), (trigger, f"Record moves to {dst}"),
               ("Open the history / audit trail", "The change, user and time are recorded")],
              _outcome(f), pre),
        Draft("Invalid transition is not allowed", TestType.NEGATIVE, "State transition testing (invalid)",
              [(f"Prepare a record in a state other than {src} (e.g. a final state)", "Record is in that state"),
               ("Attempt the same transition", "The action is unavailable or refused with a message"),
               ("Check the record", "State is unchanged")],
              "Only allowed transitions can happen", pre, [], -1),
        persist,
    ]


def design_performance(f: CriterionFacts, ctx: StoryContext) -> list[Draft]:
    threshold = next((l for l in f.limits if _is_time_unit(l.unit)), None)
    load = next((l for l in f.limits if not _is_time_unit(l.unit) and l.unit in ("users", "requests", "orders", "transactions", "")), None)
    target = threshold.describe() if threshold else "the agreed response-time target"
    drafts = [
        Draft(f"Response time within {target} under normal load", TestType.PERFORMANCE, "Performance test",
              [("Warm up the system with a few requests", "System is ready"),
               (f"Measure {_short(_action(f), 60)} 30 times", "Timings recorded"),
               ("Calculate the 95th percentile", f"p95 is {target}")],
              _outcome(f), ["Production-like test environment", "Performance tool (e.g. Locust, JMeter)"],
              [target]),
    ]
    users = int(load.high or load.low) if load and (load.high or load.low) else 100
    drafts.append(Draft(f"Behaviour under peak load ({users} concurrent)", TestType.PERFORMANCE, "Load test",
                        [(f"Ramp up to {users} concurrent virtual users", "Load reached"),
                         ("Keep the load for 15 minutes", f"Response times stay {target}; error rate < 1%"),
                         ("Ramp down", "System returns to normal; no errors remain")],
                        "Performance holds at peak load", ["Production-like test environment"], [f"{users} users"], -1))
    return drafts


def design_notification(f: CriterionFacts, ctx: StoryContext) -> list[Draft]:
    pre = _actor_pre(ctx, f)
    channels = f.channels or ["email"]
    channel = " / ".join(channels)
    within = next((f"{_fmt(l.high if l.high is not None else l.low)} {l.unit}" for l in f.time_limits), None)
    trigger = f.when[0] if f.when else f"Trigger the event: {_short(f.text, 70)}"
    return [
        Draft(f"{channel.title()} is sent when triggered", TestType.POSITIVE, "Event-driven testing",
              [(trigger, "Event completes"),
               (f"Check the recipient's {channel}", f"Exactly one message arrives{' within ' + within if within else ''}")],
              _outcome(f), pre + ["A test inbox / device is available"]),
        Draft(f"{channel.title()} content is correct", TestType.POSITIVE, "Content verification",
              [(trigger, "Message received"),
               ("Check recipient, subject, personalised data and links", "All details are correct; links work")],
              "The message is accurate and its links work", pre, [], -1),
        Draft(f"No {channel} when the event fails or the user opted out", TestType.NEGATIVE, "Negative testing",
              [("Disable this notification in the user's settings, or make the triggering action fail", "Precondition set"),
               ("Perform the action", "Action completes or fails as set up"),
               (f"Check the {channel}", "No message is sent")],
              "Messages are only sent when they should be", pre, [], -1),
    ]


def design_usability(f: CriterionFacts, ctx: StoryContext) -> list[Draft]:
    pre = _actor_pre(ctx, f)
    drafts = [
        Draft(f"UI meets the requirement: {_short(f.text, 50)}", TestType.USABILITY, "Usability check",
              [("Open the relevant screen", "Screen loads"),
               ("Verify the described UI behaviour", _outcome(f)),
               ("Complete the task without help", "A first-time user can finish it")],
              _outcome(f), pre),
        Draft("Accessible with keyboard and screen reader", TestType.ACCESSIBILITY, "Accessibility (WCAG 2.1 AA)",
              [("Navigate the screen with Tab / Shift+Tab / Enter only", "Every control is reachable; focus is visible"),
               ("Use a screen reader (NVDA / VoiceOver)", "Labels, states and messages are announced"),
               ("Check colour contrast", "Text contrast is at least 4.5:1")],
              "The feature is usable without a mouse and with assistive technology", pre, [], -1),
    ]
    if re.search(r"\b(mobile|tablet|desktop|screen|responsive|layout|device)\w*", f.text, re.I):
        drafts.append(Draft("Layout works on all screen sizes", TestType.USABILITY, "Responsive design check",
                            [(f"View the screen at {w}", "Content fits without horizontal scrolling or overlap")
                             for w in ("375px (phone)", "768px (tablet)", "1440px (desktop)")],
                            "Layout adapts to each screen size", pre, [], -1))
    return drafts


def design_compliance(f: CriterionFacts, ctx: StoryContext) -> list[Draft]:
    pre = _actor_pre(ctx, f)
    return [
        Draft(f"Compliance rule is enforced: {_short(f.text, 50)}", TestType.COMPLIANCE, "Compliance verification",
              [("Perform the regulated activity", "Activity completes"),
               ("Collect evidence (records, logs, consent, retention settings)", _outcome(f))],
              _outcome(f), pre, ["Relevant regulation / policy reference"]),
        Draft("Non-compliant action is blocked or recorded", TestType.NEGATIVE, "Negative compliance check",
              [("Attempt the activity without meeting the rule (e.g. no consent)", "The system blocks it"),
               ("Check the audit log", "The attempt is recorded")],
              "The system does not allow the rule to be broken silently", pre, [], -1),
    ]


def design_operational(f: CriterionFacts, ctx: StoryContext) -> list[Draft]:
    low = f.text.lower()
    envs = [b.title() if b not in ("ios", "android") else b.replace("ios", "iOS").title() for b in BROWSERS
            if re.search(rf"\b{b}\b", low)]
    drafts = []
    if envs:
        drafts.append(Draft("Works on every supported environment", TestType.COMPATIBILITY, "Compatibility testing",
                            [(f"Run the smoke test suite on {e}", "All smoke tests pass") for e in envs],
                            _outcome(f), [], envs))
    if re.search(r"\b(install|deploy|docker|configur|migrat|upgrade|integrat)\w*", low) or not drafts:
        drafts.append(Draft(f"Operational requirement is met: {_short(f.text, 50)}", TestType.COMPATIBILITY,
                            "Operational acceptance test",
                            [("Set up a clean environment as described", "Environment ready"),
                             ("Carry out the operation (install / deploy / configure / integrate)", "It completes without manual workarounds"),
                             ("Run the smoke test suite", "All smoke tests pass")],
                            _outcome(f), ["Clean test environment"]))
    return drafts


TECHNIQUES = {
    "functional": design_functional,
    "validation": design_validation,
    "boundary": design_boundary,
    "security": design_security,
    "error_handling": design_error_handling,
    "reliability": design_reliability,
    "workflow": design_workflow,
    "performance": design_performance,
    "notification": design_notification,
    "usability": design_usability,
    "compliance": design_compliance,
    "operational": design_operational,
}


def design(category: str, facts: CriterionFacts, ctx: StoryContext) -> list[Draft]:
    return TECHNIQUES.get(category, design_functional)(facts, ctx)
