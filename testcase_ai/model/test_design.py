"""Test design engine: turns what a criterion says into test cases that check its logic from every side.

For each criterion the engine reads the behaviour (conditions -> action -> outcomes, see behaviour.py)
and derives:

* the scenario itself, with every expected outcome checked (and the HTTP status for endpoints);
* the other side: each condition made false ("a valid key" -> "an invalid key") with the opposite outcome;
* each option the text lists ("filter by date, source or relevance" -> one test per option);
* limits (boundary value analysis, clamping, defaults, "only the first N"), timings ("every second",
  "after 24 hours") and state transitions;
* for actions that change data: reading the change back, undoing it and repeating it;
* for endpoints: missing credentials, unknown ids and malformed bodies;
* the checks the requirement category calls for (security, performance, accessibility, ...),
  only when the text gives them something concrete to test.

The engine returns drafts; engine.py numbers and prioritises them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import TestType
from .behaviour import (
    REVERSE_VERBS, Behaviour, Scenario, analyze_behaviour, article, as_precondition, as_step, capitalize, fix_articles,
    flip_antonym, flip_condition, is_negative, negate, statuses,
)
from .extractor import CriterionFacts, Limit, StoryContext, fmt_number, is_time_unit

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
SECURITY_PAYLOADS = ["<script>alert(1)</script>", "' OR '1'='1", "../../etc/passwd"]
# Environment keyword -> how it is written in test cases.
ENVIRONMENTS = {"chrome": "Chrome", "firefox": "Firefox", "safari": "Safari", "edge": "Edge", "android": "Android",
                "ios": "iOS", "windows": "Windows", "linux": "Linux", "mac": "Mac"}
# What a successful action looks like, by verb ("Users can search ..." -> "Matching results are listed").
VERB_RESULTS = {
    "search": "Matching results are listed", "find": "Matching results are listed",
    "filter": "Only the matching items are shown", "sort": "Items are shown in the chosen order",
    "save": "It is saved and listed among the saved items", "bookmark": "It is saved and listed among the saved items",
    "view": "The content is shown", "see": "The content is shown", "access": "The content is shown",
    "read": "The content is shown", "watch": "The video plays", "browse": "The items are listed",
    "upload": "The file is uploaded and shown", "download": "The file is downloaded and opens",
    "create": "It is created and shown", "add": "It is added and shown", "edit": "The change is saved and shown",
    "update": "The change is saved and shown", "change": "The change is saved and shown",
    "delete": "It is removed and no longer shown", "remove": "It is removed and no longer shown",
    "send": "It is delivered to the recipient", "communicate": "The message is delivered",
    "message": "The message is delivered", "chat": "The message is delivered", "share": "It is shared with the recipient",
    "book": "The booking is confirmed", "pay": "The payment is confirmed", "order": "The order is confirmed",
    "track": "The current status is shown", "compare": "The items are shown side by side",
    "rate": "The rating is saved and shown", "review": "The review is saved and shown",
    "subscribe": "The subscription is confirmed", "register": "The account is created", "log": "It is recorded",
    "print": "A printable version is produced", "export": "A file with the data is produced",
    "import": "The imported items appear", "receive": "It is received", "customize": "The new settings are applied",
    "set": "The new setting is applied", "select": "The selection is applied", "play": "It plays",
}
GENERIC_ROLES = {"user", "member", "customer", "shopper", "guest", "visitor", "anonymous user",
                 "unauthenticated user", "registered user", "student", "users"}

_SIGN_IN_HINT = re.compile(r"\b(?:signed[ -]in|logged[ -]in|logging (?:in|out)|my account|permission|role|"
                           r"private|dashboard|profile|cart)\b", re.I)
# Flows that happen before signing in: no sign-in precondition.
_SIGNED_OUT_FLOW = re.compile(r"\b(?:login page|log-?in page|sign[- ]?in page|forgot\w*|reset\w* (?:my |the )?password|"
                              r"sign[- ]?up|register\w*|unregistered)\b", re.I)
_STORES = re.compile(r"\b(?:sav|stor|add|creat|updat|record|persist|appear|kept|keep|import|upload|delet|remov|clear|"
                     r"increas|chang|restor)\w*", re.I)
RESTRICTED = re.compile(r"\bonly (?:the )?(?:admins?|administrators?|owners?|managers?|authori[sz]ed|signed[- ]in|"
                        r"logged[- ]in|registered|members?)\b|\b(?:admins?|administrators?|permissions?|roles?|authori[sz]\w*|"
                        r"private|secret|keys?|credentials?|40[13]|not admin|without admin)\b", re.I)
RESPONSIVE = re.compile(r"\b(?:mobile|tablet|desktop|screen size|responsive|layout|horizontal scroll\w*)\b|\d{3,4}\s*px", re.I)
_LOGIN_PROTECTED = re.compile(r"\b(?:sign(?:ed)?[ -]in|log(?:ged)?[ -]in|authenticat\w*|private|my account|dashboard)\b", re.I)
# Security checks only for what the story asks: input handling it calls unsafe, data it calls sensitive.
_UNSAFE_INPUT = re.compile(r"\b(?:sanitiz\w*|sanitis\w*|escap\w*|inject\w*|xss|cross-site|script\w*|html|malicious|"
                           r"sql|untrusted|unsafe|attack\w*|exploit\w*|prompt injection)\b", re.I)
_SENSITIVE = re.compile(r"\b(?:reveal\w*|disclos\w*|mask\w*|plain ?text|encrypt\w*|hash\w*|expos\w*|enumerat\w*|"
                        r"leak\w*|sensitive|secret\w*|personal data|pii)\b", re.I)
# A format the story states ("a valid email", "must contain a digit"); a bare length is a limit, not a format.
_FORMAT_RULE = re.compile(r"\b(?:valid\w*|invalid|format\w*|well-formed|pattern|must (?:contain|include|start|end)|"
                          r"(?:digit|number|upper-?case|lower-?case|special character|symbol|letter)s?\b)", re.I)
_REQUIRED = re.compile(r"\b(?:required|mandatory|must be (?:filled|provided|entered)|cannot be (?:empty|blank)|"
                       r"(?:empty|blank|missing) (?:\w+ )?(?:field|input|value)s?)\b", re.I)
_FAILURE = re.compile(r"\b(?:down|unavailable|unreachable|offline|fail\w*|error|timeout|crash\w*)\b", re.I)
_UI = re.compile(r"\b(?:page|screen|button|form|modal|layout|click\w*|display\w*|visible|menu|slider|image|chart|"
                 r"dialog|tab|link|widget|hero|header|footer|card|interface|ui)s?\b", re.I)
_LISTING = re.compile(r"^(?:\w+\s+)?(?:search\w*|filter\w*|list\w*|browse|find)\b", re.I)
_EMPTY_COVERED = re.compile(r"\bno (?:\w+ )?(?:match\w*|results?|items?|reviews?|data|projects?)\b|\bempty\b", re.I)
_DUPLICATE_COVERED = re.compile(r"\bduplicat\w*|\bagain\b|\btwice\b|\bincreases? (?:its )?quantity", re.I)
# Actions whose result should be unique: doing them twice must not create a second entry.
_REPEATABLE = {"save", "bookmark", "star", "subscribe", "register", "import", "follow", "like", "favorite", "favourite"}
_PATH_PARAM = re.compile(r":(\w+)|<(\w+)>|\{(\w+)\}")
_ACTION_ENDPOINT = re.compile(r"/(?:approve|reject|restore|rebuild|import|new|login|logout|send|sync|reset|verify|"
                              r"cancel|confirm|publish|run|trigger|request-\w+|[\w-]+-approval)/?$", re.I)


@dataclass
class Draft:
    title: str
    type: TestType
    technique: str
    steps: list[tuple[str, str]]
    expected: str
    preconditions: list[str] = field(default_factory=list)
    test_data: list[str] = field(default_factory=list)
    priority_offset: int = 0  # 0 = main tests of the criterion, negative = less important


def _short(text: str, n: int = 70) -> str:
    text = text.strip().rstrip(".")
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def _status_check(codes: list[str]) -> str:
    return f"Response status is {' or '.join(codes)}"


def _is_status_outcome(outcome: str) -> bool:
    return bool(re.fullmatch(r"(?i)(?:the )?response status is .+|[2-5](?:\d\d|xx)", outcome.strip()))


class _Criterion:
    """One criterion with everything the techniques need: facts, behaviour and the story around it."""

    def __init__(self, facts: CriterionFacts, story: StoryContext):
        self.facts = facts
        self.story = story
        self.text = facts.text
        self.behaviour: Behaviour = analyze_behaviour(facts.text, story.goal)

    @property
    def actor_roles(self) -> list[str]:
        """Roles someone acts in ("as admin", "only admins can", "I am admin"), not roles merely mentioned."""
        return [r for r in self.facts.roles if r not in GENERIC_ROLES and re.search(
            rf"\b(?:only|as an?|as|by an?|I am an?|I am|not an?|not|without an?|without)\s+{r}s?\b|\b{r}s?\s+(?:can|may|must|should)\b",
            self.text, re.I)]

    @property
    def role(self) -> str:
        if self.actor_roles:
            return self.actor_roles[0]
        return self.story.actor if self.story.actor.lower() not in GENERIC_ROLES else ""

    def preconditions(self, conditions: list[str]) -> list[str]:
        pre = [as_precondition(c) for c in conditions]
        denied = any(re.match(r"(?i)^(?:without|not)\b", c) for c in conditions)  # "without admin" scenario
        signed_out_flow = not self.actor_roles and (_SIGNED_OUT_FLOW.search(self.text) or _SIGNED_OUT_FLOW.search(self.story.goal))
        if not denied and not signed_out_flow and not any(p.lower().startswith(("signed in", "signed out")) for p in pre):
            if self.role:
                pre.insert(0, f"Signed in as {article(self.role)} {self.role}")
            elif _SIGN_IN_HINT.search(self.text):
                pre.insert(0, "Signed in as a user")
        return list(dict.fromkeys(pre))

    def result_of(self, sc: Scenario) -> list[str]:
        """What to check after the action: the outcomes, or what success of a capability looks like."""
        if sc.capability:
            verb = re.sub(r"(?i)^(?:I|the user|users?)\s+", "", sc.action).split()[0].lower()
            return [VERB_RESULTS.get(verb, f"{capitalize(_short(sc.action, 60))} succeeds")]
        return sc.outcomes or ["The action completes as described"]

    def steps(self, action: str, outcomes: list[str], codes: list[str] | None, request: bool) -> list[tuple[str, str]]:
        checks = [capitalize(o) for o in outcomes if not _is_status_outcome(o)]
        step = as_step(action)
        if request:
            return [(step, _status_check(codes or ["2xx"]))] + [("Inspect the response", c) for c in checks]
        if not checks:
            status = next((o for o in outcomes if _is_status_outcome(o)), "")
            return [(step, capitalize(status) if status else "The action completes as described")]
        return [(step, checks[0])] + [("Inspect the result", c) for c in checks[1:]]

    def scenario_codes(self, sc: Scenario) -> list[str]:
        found = [s for o in sc.outcomes for s in statuses(o)]
        return found or (["4xx"] if sc.negative else ["2xx"])


# --------------------------------------------------------------------------- the scenario itself

def _with_data(c: _Criterion, action: str) -> str:
    """'Leave a review with a rating outside 1-5' -> 'a rating outside 1-5' (the data that varies)."""
    prefix = f"{c.story.goal} with "
    return action[len(prefix):] if c.story.goal and action.startswith(prefix) else ""


def _main_title(c: _Criterion, sc: Scenario) -> str:
    codes = [s for o in sc.outcomes for s in statuses(o)]
    out = next((o for o in sc.outcomes if not _is_status_outcome(o)), f"returns {' or '.join(codes)}" if codes else "")
    if req := sc.request:
        codes = [s for o in sc.outcomes for s in statuses(o)]
        what = _short(out, 45) if out else f"returns {' or '.join(codes) or '2xx'}"
        prefix = f"{capitalize(_short(sc.conditions[-1], 30))}: " if sc.conditions else ""
        return f"{prefix}{req[0]} {req[1]}: {re.sub(r'(?i)^(?:the response|it) ', '', what)}"
    if sc.capability:
        return _short(capitalize(sc.outcomes[0]), 80)
    if data := _with_data(c, sc.action):
        return f"{capitalize(_short(data, 40))}: {_short(re.sub(r'(?i)^it ', '', out), 50)}"
    if sc.conditions:
        return f"{capitalize(_short(sc.conditions[-1], 40))}: {_short(out, 50)}"
    if out.lower().startswith("it ") and sc.action != c.story.goal:
        return f"{as_step(sc.action)}: {out}"
    return _short(capitalize(out or sc.action), 80)


def _is_limit_rule(c: _Criterion, sc: Scenario) -> bool:
    """'A cart can contain at most 50 distinct products': a bare rule the boundary tests check from both sides."""
    numbers = {fmt_number(v) for l in c.facts.limits if not is_time_unit(l.unit) for v in (l.low, l.high) if v is not None}
    return (sc.action == c.story.goal and not sc.conditions and bool(numbers)
            and any(n in re.findall(r"\d+(?:\.\d+)?", o) for o in sc.outcomes for n in numbers))


def scenario_tests(c: _Criterion) -> list[Draft]:
    drafts = []
    for sc in c.behaviour.scenarios:
        if _is_limit_rule(c, sc):
            continue
        outcomes = c.result_of(sc)
        drafts.append(Draft(
            _main_title(c, sc), TestType.NEGATIVE if sc.negative else TestType.POSITIVE,
            "Scenario test (as specified)", c.steps(sc.action, outcomes, c.scenario_codes(sc), bool(sc.request)),
            "; ".join(capitalize(o) for o in outcomes), c.preconditions(sc.conditions)))
    return drafts


# --------------------------------------------------------------------------- the other side

_DEADLINE = re.compile(r",?\s+(?:within|in under|in less than|in at most)\s+[\d.]+\s+\w+", re.I)
# "submit my registered email" -> also try an address that is not one
_EMAIL_INPUT = re.compile(r"\b(?:registered|valid|existing|verified|correct)\s+(?:e-?mail(?: address)?)\b", re.I)
MALFORMED_EMAIL = "jane.doe@"


def _opposite(outcomes: list[str], same: bool) -> list[str]:
    """The outcomes that must NOT happen any more (status-only outcomes are covered by the status check).

    A deadline is dropped first: an unregistered email gets no reset link at all, not "not within 5 minutes".
    """
    if same:
        return outcomes
    return [negate(_DEADLINE.sub("", o)) for o in outcomes if not _is_status_outcome(o)]


def _capable_role(sc: Scenario) -> str:
    """The specific role in 'Instructors can create ...' ('instructor'); '' for anyone ('Users can ...')."""
    m = re.match(r"(?i)^(?:the |all |only )?(.+?)\s+(?:can|cannot)\s", sc.outcomes[0] if sc.outcomes else "")
    if not m:
        return ""
    who = m.group(1).lower()
    if re.search(r"(?:ch|sh|ss|x)es$", who):
        who = who[:-2]  # coaches -> coach
    elif re.search(r"[^s]s$", who):
        who = who[:-1]  # instructors -> instructor
    generic = GENERIC_ROLES | {"user", "customer", "visitor", "people", "everyone", "anyone", "the user", "i", "we"}
    systems = r"\b(?:system|app|application|platform|website|site|service|tool|portal|software|page|api|server)\b"
    return "" if who in generic or re.search(systems, who) or len(who.split()) > 3 else who


def other_side_tests(c: _Criterion) -> list[Draft]:
    """Make each condition (or the qualifier that triggers the behaviour) false and expect the opposite."""
    drafts = []
    same = bool(re.search(r"\b(?:same|identical|indistinguishable|either way)\b", c.text, re.I))
    stated = {cond.lower() for sc in c.behaviour.scenarios for cond in sc.conditions}  # already specified by the story
    limit_numbers = {fmt_number(v) for l in c.facts.limits if not is_time_unit(l.unit)
                     for v in (l.low, l.high) if v is not None}
    # a time in the action ("open it after 24 hours") is checked by the timing tests; a deadline in the
    # outcome ("a link within 5 minutes") still leaves the input to reverse
    time_numbers = {fmt_number(v) for l in c.facts.time_limits for v in (l.low, l.high) if v is not None}
    for sc in [s for s in c.behaviour.scenarios if not s.derived][:3]:
        outcomes = c.result_of(sc)
        if (any(n in re.findall(r"\d+(?:\.\d+)?", o + " " + sc.action) for o in outcomes for n in limit_numbers)
                or time_numbers & set(re.findall(r"\d+(?:\.\d+)?", sc.action + " " + " ".join(sc.conditions)))):
            continue  # a limit rule: the boundary / timing tests check both sides of it
        variants: list[tuple[list[str], str, str]] = []  # (conditions, action, what changed)
        for i, cond in enumerate(sc.conditions):
            flipped = flip_condition(cond)
            if not flipped or flipped.lower() in stated:
                continue
            if flipped.startswith("no ") and re.search(r"\b(?:they|them|their)\b", " ".join(outcomes), re.I):
                continue  # "no status events: they are ignored" has no meaningful opposite
            variants.append((sc.conditions[:i] + [flipped] + sc.conditions[i + 1:], sc.action, flipped))
        if not variants and sc.action != c.story.goal and (flipped := flip_antonym(sc.action)):
            email = _EMAIL_INPUT.search(sc.action)
            label = capitalize(fix_articles(f"a {flip_antonym(email.group(0))}")) if email else None
            variants.append((sc.conditions, flipped, label or _with_data(c, flipped) or as_step(flipped)))
            if email:
                malformed = _EMAIL_INPUT.sub(f"a malformed email address ({MALFORMED_EMAIL})", sc.action, count=1)
                variants.append((sc.conditions, re.sub(r"\bmy a\b", "a", malformed), "A malformed email address"))
        if sc.capability and (who := _capable_role(sc)):
            # "The healthcare professional can review results" -> others cannot
            drafts.append(Draft(f"Users who are not {article(who)} {who} cannot {_short(sc.action, 45)}", TestType.SECURITY,
                                "Access control (forbidden role)",
                                [(f"Sign in as a user who is not {article(who)} {who}", "Signed in"),
                                 (as_step(sc.action), "The function is not offered, or it is refused with a message")],
                                f"Only {who}s can {_short(sc.action, 50)}", ["An account without that role exists"], [], -1))
        if not variants and not sc.conditions and sc.negative and (flipped := flip_antonym(outcomes[0])):
            # "Out-of-stock products cannot be added" -> "In-stock products can be added"
            drafts.append(Draft(capitalize(_short(negate(flipped), 80)), TestType.POSITIVE,
                                "Equivalence partitioning (valid class)",
                                c.steps(sc.action, [negate(flipped)], ["2xx"], bool(sc.request)),
                                capitalize(negate(flipped)), c.preconditions(sc.conditions)))
        for conditions, action, changed in variants[:2]:
            expected = _opposite(outcomes, same)
            original = next((x for x in sc.conditions if x not in conditions), "")
            failure = bool(_FAILURE.search(original) or re.search(r"\b(?:is|are) down\b", original, re.I))
            if failure:
                # "the database is up": the normal path runs instead of the failure handling
                expected = [f"{capitalize(c.story.goal or 'the action')} works normally; the failure handling is not used"]
            if sc.negative or failure:
                codes = ["2xx"]
            else:
                codes = ["4xx"] if is_negative(changed) or re.search(r"\b(?:invalid|wrong|missing|unknown)\b", changed, re.I) else ["2xx"]
            summary = expected[0] if expected else ("the request succeeds" if "2xx" in codes else "the request is refused")
            drafts.append(Draft(
                f"{capitalize(_short(changed, 45))}: {_short(summary, 45)}",
                TestType.POSITIVE if sc.negative or failure else TestType.NEGATIVE, "Decision table (condition reversed)",
                c.steps(action, expected or ([] if sc.request else [summary]), codes, bool(sc.request)),
                capitalize("; ".join(expected) or summary),
                c.preconditions(conditions)))
    return drafts


# --------------------------------------------------------------------------- listed options

def option_tests(c: _Criterion) -> list[Draft]:
    b = c.behaviour
    if not b.options:
        return []
    sc = b.main
    drafts = []
    for option in b.options:
        outcomes = c.result_of(sc)
        if b.options_in_outcome:  # each listed criterion of the result is checked on its own
            action = f"{as_step(sc.action)} with a value that matches by {option} only"
            expected = [sc.outcomes[0].replace(b.options_text, option, 1)] + outcomes[1:]
        else:
            action = sc.action.replace(b.options_text, option, 1)
            action = re.sub(r"\(\s*(?:e\.g\.,?|such as|like)?\s*([^()]+?)\s*\)", r"\1", action)  # "(e.g. nutrition)"
            expected = [f"{outcomes[0]} ({option})"] + outcomes[1:] if sc.capability else outcomes
        drafts.append(Draft(f"{as_step(action)}", TestType.NEGATIVE if sc.negative else TestType.POSITIVE,
                            "Equivalence partitioning (each listed option)",
                            c.steps(action, expected, c.scenario_codes(sc), bool(sc.request)),
                            "; ".join(capitalize(e) for e in expected), c.preconditions(sc.conditions), [option], -1))
    return drafts


# --------------------------------------------------------------------------- data changes

def change_tests(c: _Criterion) -> list[Draft]:
    """For actions that change data: read the change back, undo it, repeat it."""
    b, sc = c.behaviour, c.behaviour.main
    if not b.writes or sc.negative:
        return []
    if not sc.request and (sc.action == c.story.goal or not any(_STORES.search(o) for o in c.result_of(sc))):
        return []  # nothing is stored by this action (or the criterion names no action of its own)
    pre = c.preconditions(sc.conditions)
    action = as_step(sc.action)
    first = c.result_of(sc)[0]
    drafts = []
    if req := sc.request:
        method, path = req
        if method == "DELETE" and _PATH_PARAM.search(path):
            read = [(action, _status_check(["2xx"])), (f"Send GET {path}", _status_check(["404"])),
                    ("List the remaining items", "The deleted item is no longer listed")]
            drafts.append(Draft(f"Deleted item is gone when read back ({path})", TestType.EDGE, "Read-back check", read,
                                "The deleted item can no longer be read", pre, [], -1))
        elif method == "DELETE":  # clearing a whole collection
            read = [(action, _status_check(["2xx"])), (f"Send GET {path}", _status_check(["2xx"])),
                    ("Inspect the response", "The list is empty")]
            drafts.append(Draft(f"Collection is empty when read back ({path})", TestType.EDGE, "Read-back check", read,
                                "Nothing is left after clearing", pre, [], -1))
        elif not _ACTION_ENDPOINT.search(path):  # /changes/:code/approve is a command, not something to read back
            read = [(action, _status_check(["2xx"])), (f"Send GET {path}", _status_check(["2xx"])),
                    ("Inspect the response", "It contains the change just made")]
            drafts.append(Draft(f"Change is returned when read back ({method} {path})", TestType.EDGE, "Read-back check",
                                read, "The stored data reflects the change", pre, [], -1))
    else:
        drafts.append(Draft(f"Result persists after reload: {_short(first, 45)}", TestType.EDGE, "Persistence check",
                            [(action, capitalize(first)), ("Reload the page", "The page reloads"),
                             ("Open the affected item again", "The change is still there")],
                            "The outcome is stored, not only shown", pre, [], -1))
    verb = b.verb
    if verb in REVERSE_VERBS and not sc.request:
        undo = REVERSE_VERBS[verb]
        drafts.append(Draft(f"{capitalize(verb)} then {undo}: {_short(sc.action, 40)}", TestType.EDGE, "Reverse action",
                            [(action, capitalize(first)), (f"{capitalize(undo)} it again", "It is no longer there"),
                             ("Check the related lists and counters", "They are back to their previous values")],
                            f"The {verb} can be undone and leaves no trace", pre, [], -1))
    if verb in _REPEATABLE and not sc.request and not _DUPLICATE_COVERED.search(c.story.criteria or c.text):
        drafts.append(Draft(f"Repeating it does not create a duplicate: {_short(sc.action, 35)}", TestType.EDGE,
                            "Idempotency check",
                            [(action, capitalize(first)), (f"{action} again with the same data", "The repeat is handled"),
                             ("Check the stored items", "There is no duplicate entry")],
                            "Repeating the action does not create duplicates", pre, [], -1))
    return drafts


def empty_state_tests(c: _Criterion) -> list[Draft]:
    sc = c.behaviour.main
    lists = any(re.search(r"\b(?:listed|results|are returned|lists?|entries|items)\b", o, re.I) for o in c.result_of(sc))
    listing = _LISTING.search(sc.action) or lists
    if not listing or sc.negative or c.behaviour.writes or _EMPTY_COVERED.search(c.story.criteria or c.text):
        return []
    if sc.request:
        steps = [("Make sure no matching data exists", "Nothing matches"), (as_step(sc.action), _status_check(["2xx"])),
                 ("Inspect the response", "An empty list is returned, not an error")]
    else:
        steps = [("Make sure no matching data exists", "Nothing matches"),
                 (as_step(sc.action), "A clear empty-state message is shown, not an error")]
    return [Draft("Nothing matches: an empty state is shown, not an error", TestType.EDGE, "Equivalence partitioning (empty)",
                  steps, "An empty result is handled gracefully", c.preconditions(sc.conditions), [], -1)]


# --------------------------------------------------------------------------- endpoints

def endpoint_tests(c: _Criterion) -> list[Draft]:
    sc = c.behaviour.main
    if not (req := sc.request):
        return []
    method, path = req
    writes = method != "GET"
    nothing = [("Check the stored data", "Nothing was changed")] if writes else []
    covered = {s for s2 in c.behaviour.scenarios for o in s2.outcomes for s in statuses(o)}
    pre = [p for p in c.preconditions(sc.conditions) if not p.lower().startswith("signed in")]
    base = re.sub(r"\s+(?:as|with|without)\b.*$", "", as_step(sc.action))
    role = m.group(0) if (m := re.search(r"\s+as\s+(?:an?\s+)?\w+", sc.action)) else ""
    drafts = []
    if ("/admin" in path or RESTRICTED.search(c.text)) and not covered & {"401", "403"}:
        drafts.append(Draft(f"{method} {path} without credentials is refused", TestType.SECURITY, "Access control (no credentials)",
                            [(f"{base} without credentials", _status_check(["401"]))] + nothing,
                            "Unauthenticated requests are refused and change nothing", pre, [], -1))
    if (param := _PATH_PARAM.search(path)) and not re.search(r"\bunknown\b", c.text, re.I):
        name = next(g for g in param.groups() if g)
        drafts.append(Draft(f"{method} {path} with an unknown {name} returns 404", TestType.NEGATIVE, "Error guessing (unknown id)",
                            [(f"{base}{role} with an unknown {name}", _status_check(["404"]))] + nothing,
                            f"An unknown {name} is reported as not found", c.preconditions(sc.conditions), [], -1))
    if method in ("POST", "PUT", "PATCH") and not covered & {"400", "422"}:
        drafts.append(Draft(f"{method} {path} with a malformed body returns 400", TestType.NEGATIVE, "Input validation (request body)",
                            [(f"{base}{role} with a missing or malformed body", _status_check(["400"]))] + nothing,
                            "Invalid requests are refused with 400 and change nothing", c.preconditions(sc.conditions),
                            [], -1))
    return drafts


# --------------------------------------------------------------------------- limits, defaults, timings

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
    v = fmt_number(value)
    if unit.startswith("char"):
        n = int(value)
        sample = ("Abcdefghij" * (n // 10 + 1))[:n]
        return f"{v} characters ('{sample[:12]}{'…' if n > 12 else ''}')"
    return f"{v} {unit}".strip()


def limit_tests(c: _Criterion) -> list[Draft]:
    b, f = c.behaviour, c.facts
    pre = c.preconditions([])  # the values under test replace whatever condition named them
    limits = [l for l in f.limits if not is_time_unit(l.unit)]  # timings are checked by the timing / performance tests
    drafts = []
    for limit in limits[:2]:
        accepted, rejected = _boundary_values(limit)
        unit = limit.unit
        if unit and all(v is None or v >= 0 for v in (limit.low, limit.high)):
            # A count with a non-negative limit ("0 to 10 items") cannot go below 0, so -1 is no test value.
            accepted, rejected = [v for v in accepted if v >= 0], [v for v in rejected if v >= 0]
        if b.truncate:
            n = fmt_number(limit.high if limit.high is not None else limit.low)
            shown = f"Only {n} are shown" + (" with a count of the total" if b.shows_count else "")
            drafts.append(Draft(f"Up to {n}: everything is shown ({limit.describe()})", TestType.EDGE, "Boundary value analysis",
                                [(f"Use {_render(v, unit)}", "All of them are shown") for v in accepted],
                                f"Lists within {limit.describe()} are shown in full", pre, [_render(v, unit) for v in accepted]))
            drafts.append(Draft(f"Above {n}: only the first {n} are shown", TestType.EDGE, "Boundary value analysis",
                                [(f"Use {_render(v, unit)}", shown) for v in rejected],
                                shown, pre, [_render(v, unit) for v in rejected]))
            continue
        drafts.append(Draft(f"Values on the boundary are accepted ({limit.describe()})", TestType.EDGE,
                            "Boundary value analysis", [(f"Use {_render(v, unit)}", "Accepted") for v in accepted],
                            f"All values within {limit.describe()} are accepted", pre, [_render(v, unit) for v in accepted]))
        if not rejected:
            continue
        if b.clamp:
            edge = lambda v: limit.low if limit.low is not None and v < limit.low else limit.high  # noqa: E731
            drafts.append(Draft(f"Values outside the range are clamped ({limit.describe()})", TestType.EDGE,
                                "Boundary value analysis",
                                [(f"Use {_render(v, unit)}", f"Stored as {_render(edge(v), unit)}") for v in rejected],
                                f"Values outside {limit.describe()} are stored as the nearest limit", pre,
                                [_render(v, unit) for v in rejected]))
        else:
            drafts.append(Draft(f"Values outside the boundary are rejected ({limit.describe()})", TestType.NEGATIVE,
                                "Boundary value analysis",
                                [(f"Use {_render(v, unit)}", "Rejected with a message that states the limit") for v in rejected],
                                f"Values outside {limit.describe()} are rejected and nothing is saved", pre,
                                [_render(v, unit) for v in rejected]))
    if b.default:
        name, value = b.default
        drafts.append(Draft(f"Missing {name} defaults to {value}", TestType.EDGE, "Default value check",
                            [(f"Leave the {name} out", "The request is accepted"), ("Inspect the stored item", f"The {name} is {value}")],
                            f"A missing {name} is stored as {value}", pre, ["(empty)"], -1))
    return drafts


def timing_tests(c: _Criterion) -> list[Draft]:
    sc, pre = c.behaviour.main, c.preconditions(c.behaviour.main.conditions)
    drafts = []
    if m := re.search(r"\bevery\s+(?:(\d+)\s+)?(second|minute|hour|day|week)s?\b", c.text, re.I):
        n, unit = m.group(1) or "1", m.group(2)
        drafts.append(Draft(f"Updates every {n} {unit}{'s' if n != '1' else ''}", TestType.POSITIVE, "Timing check",
                            [("Read the current value", "Value noted"), (f"Wait {n} {unit}", "The value has changed"),
                             (f"Wait another {n} {unit}", "It has changed again by one interval")],
                            f"The value refreshes every {n} {unit}", pre))
    if (m := re.search(r"\bafter\s+(\d+)\s+(second|minute|hour|day|week)s?\b", sc.action, re.I)) and sc.outcomes:
        before = sc.action[: m.start()] + f"just before {m.group(1)} {m.group(2)}s" + sc.action[m.end():]
        expected = negate(sc.outcomes[0])
        drafts.append(Draft(f"Just before {m.group(1)} {m.group(2)}s: {_short(expected, 45)}", TestType.EDGE,
                            "Boundary value analysis (time)", [(as_step(before), capitalize(expected))],
                            capitalize(expected), pre))
    return drafts


# --------------------------------------------------------------------------- state transitions

def workflow_tests(c: _Criterion) -> list[Draft]:
    f, sc = c.facts, c.behaviour.main
    if len(f.states) < 2:
        return []
    pre, trigger = c.preconditions(sc.conditions), as_step(sc.action)
    src, dst = f.states[0], f.states[1]
    return [
        Draft(f"Valid transition {src} → {dst}", TestType.POSITIVE, "State transition testing (valid)",
              [(f"Prepare a record in {src}", f"Record shows {src}"), (trigger, f"Record moves to {dst}"),
               ("Open the history / audit trail", "The change, user and time are recorded")],
              f"The record moves from {src} to {dst}", pre),
        Draft(f"Transition to {dst} is refused from other states", TestType.NEGATIVE, "State transition testing (invalid)",
              [(f"Prepare a record in a state other than {src} (e.g. {dst} already)", "Record is in that state"),
               (trigger, "The action is unavailable or refused with a message"), ("Check the record", "State is unchanged")],
              "Only allowed transitions can happen", pre, [], -1),
    ]


# --------------------------------------------------------------------------- category checks

def validation_tests(c: _Criterion) -> list[Draft]:
    """Format checks for fields whose format the criterion states; endpoints get the malformed-body test instead.

    A length rule ("at least 12 characters") is tested by the boundary tests, not by invented format classes,
    and an empty-input test needs the story to say the input is required.
    """
    if c.behaviour.main.request:
        return []
    pre = c.preconditions([])
    drafts = []
    if not _FORMAT_RULE.search(c.text):
        return _required_test(c, pre)
    # Only fields with a format of their own; "name" or free text have no rule unless the story states one.
    for fld in [f for f in c.facts.fields if f not in ("name", "text", "quantity", "file")][:2]:
        samples = FIELD_SAMPLES[fld]
        label = fld.replace("_", " ")
        drafts.append(Draft(f"Valid {label} is accepted", TestType.POSITIVE, "Equivalence partitioning (valid class)",
                            [(f"Enter {label}: {samples['valid'][0]}", "No validation error is shown"),
                             ("Submit the form", "The data is accepted and saved")],
                            "Valid input passes validation", pre, samples["valid"]))
        drafts.append(Draft(f"Invalid {label} formats are rejected", TestType.NEGATIVE, "Equivalence partitioning (invalid classes)",
                            [(f"Enter {label}: {v} and submit", "Submission is blocked with a field-level error")
                             for v in samples["invalid"]],
                            f"Every invalid {label} is rejected with a clear message and nothing is saved", pre,
                            samples["invalid"]))
    return drafts + _required_test(c, pre)


def _required_test(c: _Criterion, pre: list[str]) -> list[Draft]:
    if not _REQUIRED.search(c.text):
        return []
    return [Draft("Required input left empty is refused", TestType.NEGATIVE, "Equivalence partitioning (empty class)",
                  [("Leave the required field(s) empty", "Field is empty"),
                   ("Submit the form", "A 'required' message is shown next to the field"),
                   ("Enter only spaces and submit", "Whitespace-only input is treated as empty")],
                  "Empty or blank input is not accepted", pre, ["(empty)", "'   '"], -1)]


def security_tests(c: _Criterion) -> list[Draft]:
    low, sc = c.text.lower(), c.behaviour.main
    drafts: list[Draft] = []
    roles = c.actor_roles
    if roles and not sc.request and not (sc.capability and _capable_role(sc)):  # capabilities test this themselves
        allowed = roles[0]
        drafts += [
            Draft(f"{capitalize(allowed)} can {_short(sc.action, 50)}", TestType.POSITIVE, "Access control (allowed role)",
                  [(f"Sign in as {article(allowed)} {allowed}", "Signed in"), (as_step(sc.action), "Access is granted")],
                  "The permitted user can perform the action", [f"{capitalize(article(allowed))} {allowed} account exists"]),
            Draft(f"Users who are not {article(allowed)} {allowed} are denied", TestType.SECURITY, "Access control (forbidden role)",
                  [(f"Sign in as a user who is not {article(allowed)} {allowed}", "Signed in"),
                   (as_step(sc.action), "Access is denied (403 / hidden control)"),
                   ("Call the same URL or API endpoint directly", "Request is rejected; no data is returned")],
                  "Users without permission cannot perform the action by any route",
                  ["An account without the required role exists"], [], -1),
        ]
    if _SENSITIVE.search(low):
        drafts.append(Draft("No sensitive information is disclosed", TestType.SECURITY, "Information disclosure check",
                            [("Perform the action with a valid / existing value", "Note the response, message and timing"),
                             ("Repeat with an invalid / non-existing value", "Response, message and timing are indistinguishable"),
                             ("Inspect page source, network responses and logs", "No secrets, personal data or internal details appear")],
                            "Nothing reveals whether the value exists", [], ["An existing and a non-existing account"], -1))
    if (roles or _LOGIN_PROTECTED.search(low)) and not sc.request:
        drafts.append(Draft("Signed-out access is blocked", TestType.SECURITY, "Authentication check",
                            [("Sign out and clear cookies", "Session ended"),
                             ("Open the protected page or endpoint directly", "Redirected to sign-in / 401 returned")],
                            "No protected content is reachable without authentication", [], [], -1))
    if re.search(r"\b(?:session|expire\w*|logout|log out|reset link|one-time|single-use|locked out)\b", low):
        drafts.append(Draft("Expired, reused or revoked credentials are rejected", TestType.SECURITY, "Session / token handling",
                            [("Obtain a valid session, token or link", "It works"),
                             ("Use it after expiry, after logout or a second time", "Rejected; the user must authenticate again")],
                            "Expired or single-use credentials cannot be replayed", [], [], -1))
    if _UNSAFE_INPUT.search(low):
        drafts.append(Draft("Malicious input is neutralised", TestType.SECURITY, "Injection testing",
                            [(f"Enter {p}", "Input is rejected or safely escaped") for p in SECURITY_PAYLOADS],
                            "No script runs and no data leaks", c.preconditions([]), SECURITY_PAYLOADS, -1))
    return drafts  # nothing concrete to check: the scenario and reversed-condition tests cover the rule


def performance_tests(c: _Criterion) -> list[Draft]:
    f, sc = c.facts, c.behaviour.main
    if f.channels:
        return []  # "a reset link by email within 5 minutes": the delivery time is checked by the notification test
    threshold = next((l for l in f.limits if is_time_unit(l.unit) and l.kind != "exact"), None)
    if not threshold and not re.search(r"\b(?:fast|quick\w*|response time|load (?:test|time)|under load|concurrent|throughput|"
                                       r"peak|scal\w*)\b", c.text, re.I):
        return []
    target = threshold.describe() if threshold else "the agreed response-time target"
    action = _short(as_step(sc.action), 60)
    drafts = [Draft(f"{action} responds within {target}", TestType.PERFORMANCE, "Performance test",
                    [("Warm up the system with a few requests", "System is ready"),
                     (f"Measure {action} 30 times", "Timings recorded"),
                     ("Calculate the 95th percentile", f"p95 is {target}")],
                    f"p95 response time is {target}", ["Production-like test environment"], [target])]
    load = next((l for l in f.limits if not is_time_unit(l.unit) and re.search(r"user|request|session|order|transaction", l.unit)), None)
    if load or re.search(r"\b(?:concurrent|peak|under load|load test|scal\w*)\b", c.text, re.I):
        users = int(load.high or load.low) if load and (load.high or load.low) else 100
        drafts.append(Draft(f"Holds up with {users} concurrent users", TestType.PERFORMANCE, "Load test",
                            [(f"Ramp up to {users} concurrent virtual users", "Load reached"),
                             ("Keep the load for 15 minutes", f"Response times stay {target}; error rate < 1%"),
                             ("Ramp down", "System returns to normal; no errors remain")],
                            "Performance holds at peak load", ["Production-like test environment"], [f"{users} users"], -1))
    return drafts


def usability_tests(c: _Criterion) -> list[Draft]:
    if not (_UI.search(c.text) or _UI.search(c.story.goal) or RESPONSIVE.search(c.text)):
        return []
    pre = c.preconditions([])
    drafts = [Draft("Accessible with keyboard and screen reader", TestType.ACCESSIBILITY, "Accessibility (WCAG 2.1 AA)",
                    [("Navigate the screen with Tab / Shift+Tab / Enter only", "Every control is reachable; focus is visible"),
                     ("Use a screen reader (NVDA / VoiceOver)", "Labels, states and messages are announced"),
                     ("Check colour contrast", "Text contrast is at least 4.5:1")],
                    "The feature is usable without a mouse and with assistive technology", pre, [], -1)]
    if RESPONSIVE.search(c.text):
        drafts.append(Draft("Layout works on all screen sizes", TestType.USABILITY, "Responsive design check",
                            [(f"View the screen at {w}", "Content fits without horizontal scrolling or overlap")
                             for w in ("375px (phone)", "768px (tablet)", "1440px (desktop)")],
                            "Content fits without horizontal scrolling on every screen size", pre, [], -1))
    return drafts


def notification_tests(c: _Criterion) -> list[Draft]:
    sc = c.behaviour.main
    channels = c.facts.channels
    if sc.negative or not (channels or re.search(r"\b(?:notif\w*|message|reply|alert)\b", c.text, re.I)):
        return []  # "no reply is sent" is checked by the scenario test itself
    channel = " / ".join(channels) or "message"
    pre = c.preconditions(sc.conditions) + [f"Access to the recipient's {channel}"]
    trigger = as_step(sc.action)
    within = next((f"{fmt_number(l.high if l.high is not None else l.low)} {l.unit}" for l in c.facts.time_limits), None)
    drafts = []
    # What the message carries, as the criterion names it ("a reset link", "a one-time code").
    carried = re.search(r"\b(?:an? |the )?((?:[\w-]+ )?(?:link|code|otp|token|receipt|invoice|ticket|password))\b",
                        " ".join(c.result_of(sc)), re.I)
    if carried:
        thing = carried.group(1)
        drafts.append(Draft(f"The {channel} contains a working {thing}", TestType.POSITIVE, "Content verification",
                            [(trigger, f"The {channel} is received"),
                             (f"Use the {thing} from the {channel}", f"The {thing} works for this account")],
                            f"The {thing} in the {channel} can be used", pre, [], -1))
    if re.search(r"\b(?:once|duplicat\w*|single|only one|one (?:email|message|notification|sms))\b", c.text, re.I):
        drafts.append(Draft(f"{capitalize(channel)} arrives exactly once", TestType.POSITIVE, "Event-driven testing",
                            [(trigger, "Event completes"),
                             (f"Check the recipient's {channel}",
                              f"Exactly one message arrives{' within ' + within if within else ''}")],
                            "One message per event, no duplicates", pre, [], -1))
    return drafts


def reliability_tests(c: _Criterion) -> list[Draft]:
    low = c.text.lower()
    drafts = []
    if re.search(r"\b(?:backup|back up|restore|recover\w*|data loss)\b", low):
        drafts.append(Draft("Backup can be restored without data loss", TestType.RELIABILITY, "Backup / restore test",
                            [("Create known test records", "Records saved"), ("Run or wait for the backup", "Backup completes"),
                             ("Restore the backup to a clean environment", "Restore completes within the agreed time"),
                             ("Compare the records", "All records are present and unchanged")],
                            "Restored data matches the original", ["Access to a staging environment"], [], -1))
    if re.search(r"\b(?:availab\w*|uptime|24/7|99(?:\.\d+)?\s*%)", low):
        drafts.append(Draft("Availability meets the target", TestType.RELIABILITY, "Availability monitoring",
                            [("Run a synthetic check every minute for the measurement period", "Checks recorded"),
                             ("Calculate uptime from the results", f"Uptime meets the target ({_short(c.text, 50)})")],
                            "Measured availability meets the requirement", ["Uptime monitoring tool configured"],
                            [l.describe() for l in c.facts.limits] or ["Target from the requirement"], -1))
    cond = next((x for x in c.behaviour.main.conditions if _FAILURE.search(x)), "")
    if cond and c.behaviour.writes:
        drafts.append(Draft(f"No partial data after the failure ({_short(cond, 35)})", TestType.RELIABILITY, "Data integrity check",
                            [(f"Make it happen: {cond}", "Failure is in place"), (as_step(c.behaviour.main.action), "It is handled"),
                             ("Check the stored data", "Nothing was half-saved; data is consistent")],
                            "The system stays consistent when the failure happens", [], [], -1))
    return drafts


def compliance_tests(c: _Criterion) -> list[Draft]:
    if not re.search(r"\b(?:audit|consent|retention|retain|gdpr|hipaa|pci|regulat\w*|legal|law|policy|privacy|log(?:ged|s)?)\b",
                     c.text, re.I):
        return []
    return [Draft("Evidence of the rule is recorded", TestType.COMPLIANCE, "Compliance verification",
                  [(as_step(c.behaviour.main.action), capitalize(c.result_of(c.behaviour.main)[0])),
                   ("Collect evidence (audit log, consent record, retention setting)", "The evidence exists and is complete")],
                  "The rule can be demonstrated to an auditor", c.preconditions([]), ["Relevant regulation / policy reference"], -1)]


def operational_tests(c: _Criterion) -> list[Draft]:
    low = c.text.lower()
    envs = [name for keyword, name in ENVIRONMENTS.items() if re.search(rf"\b{keyword}\b", low)]
    drafts = []
    if envs:
        drafts.append(Draft("Works on every supported environment", TestType.COMPATIBILITY, "Compatibility testing",
                            [(f"Run the smoke test suite on {e}", "All smoke tests pass") for e in envs],
                            "Every named environment is supported", [], envs))
    if re.search(r"\b(?:install\w*|deploy\w*|docker|configur\w*|migrat\w*|upgrade\w*|integrat\w*)\b", low):
        drafts.append(Draft(f"Operational requirement is met: {_short(c.text, 45)}", TestType.COMPATIBILITY,
                            "Operational acceptance test",
                            [("Set up a clean environment as described", "Environment ready"),
                             ("Carry out the operation (install / deploy / configure / integrate)", "It completes without manual workarounds"),
                             ("Run the smoke test suite", "All smoke tests pass")],
                            "The operation works on a clean environment", ["Clean test environment"], [], -1))
    return drafts


# --------------------------------------------------------------------------- putting it together

def design(techniques: list[str], facts: CriterionFacts, ctx: StoryContext) -> list[Draft]:
    """All test drafts for one criterion, most important first.

    `techniques` are the requirement categories the model picked (main category first); they add
    category checks on top of the tests derived from the criterion's own logic.
    """
    c = _Criterion(facts, ctx)
    core = other_side_tests(c) + limit_tests(c) + timing_tests(c) + workflow_tests(c) + option_tests(c) + endpoint_tests(c)
    lenses: list[Draft] = []
    extra = {
        "validation": lambda: validation_tests(c),
        "security": lambda: security_tests(c),
        "performance": lambda: performance_tests(c),
        "usability": lambda: usability_tests(c),
        "notification": lambda: notification_tests(c),
        "reliability": lambda: reliability_tests(c),
        "error_handling": lambda: reliability_tests(c),
        "compliance": lambda: compliance_tests(c),
        "operational": lambda: operational_tests(c),
    }
    for technique in dict.fromkeys(techniques):
        if technique in extra:
            lenses += extra[technique]()
    if "validation" not in techniques and c.facts.fields:
        lenses += validation_tests(c)
    # "Email must be valid": a bare quality statement is tested by the concrete checks it calls for, when there are any.
    statement = all(sc.action == ctx.goal and not sc.conditions and not sc.capability and not sc.negative
                    for sc in c.behaviour.scenarios)
    main = [] if statement and lenses else scenario_tests(c)
    return main + core + lenses + change_tests(c) + empty_state_tests(c)
