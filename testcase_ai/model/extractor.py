"""Rule-based information extraction from user stories and acceptance criteria.

The classifier decides *what kind* of requirement a sentence is; this module
pulls out the concrete facts (limits, fields, roles, states, timings, Given /
When / Then parts) that the test design engine turns into test data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import UserStory

_NUM = r"(\d+(?:[.,]\d+)?)"
_UNIT = r"\s*(%|percent|[a-zA-Z]+)?"

# (regex, kind) - kind decides how the number(s) are read.
_LIMIT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\bbetween\s+{_NUM}\s+and\s+{_NUM}{_UNIT}", re.I), "range"),
    (re.compile(rf"\bfrom\s+{_NUM}\s+to\s+{_NUM}{_UNIT}", re.I), "range"),
    (re.compile(rf"\b{_NUM}\s*(?:to|-)\s*{_NUM}{_UNIT}", re.I), "range"),
    (re.compile(rf"\b(?:at least|minimum of|min(?:imum)?|no less than|not less than)\s+{_NUM}{_UNIT}", re.I), "min"),
    (re.compile(rf"\b{_NUM}{_UNIT}\s+or\s+(?:more|older|above|greater|higher)\b", re.I), "min"),
    (re.compile(rf"\b(?:more than|over|above|greater than|exceeds?)\s+{_NUM}{_UNIT}", re.I), "min_exclusive"),
    (re.compile(rf"\b(?:at most|up to|maximum of|max(?:imum)?|no more than|not more than|limited to|"
                rf"cannot exceed|can not exceed|must not exceed|not exceed|exceed|within|in under|under)\s+"
                rf"(?:the\s+\w+\s+limit\s+of\s+)?{_NUM}{_UNIT}", re.I), "max"),
    (re.compile(rf"\b(?:less than|fewer than|below|smaller than)\s+{_NUM}{_UNIT}", re.I), "max_exclusive"),
    (re.compile(rf"\b{_NUM}{_UNIT}\s+or\s+(?:less|fewer|younger|below)\b", re.I), "max"),
    (re.compile(rf"\b(?:after|every)\s+{_NUM}{_UNIT}", re.I), "exact"),
    (re.compile(rf"\b(?:first|top|latest|last)\s+{_NUM}{_UNIT}", re.I), "max"),  # "only the first 12 projects"
]
_EVERY_UNIT = re.compile(r"\bevery\s+(second|minute|hour|day|week)\b", re.I)  # "every second" = every 1 second

FIELD_VOCAB = {
    "email": ["email", "e-mail"],
    "password": ["password", "passcode", "pin"],
    "username": ["username", "user name", "login name"],
    "phone": ["phone", "mobile number", "phone number"],
    "date": ["date", "date of birth", "start date", "end date"],
    "url": ["url", "website", "link"],
    "postal_code": ["postal code", "zip", "postcode"],
    "card": ["credit card", "card number", "card"],
    "iban": ["iban"],
    "name": ["first name", "last name", "name", "title"],
    "quantity": ["quantity", "amount", "number of"],
    "file": ["file", "upload", "image", "attachment", "picture"],
    "text": ["comment", "description", "bio", "message", "query", "text"],
}

ROLE_VOCAB = [
    "super admin", "administrator", "admin", "manager", "editor", "viewer", "guest", "customer",
    "shopper", "support agent", "agent", "teacher", "student", "owner", "reviewer", "assignee",
    "registered user", "unauthenticated user", "anonymous user", "user", "member",
]

CHANNEL_VOCAB = {
    "email": ["email", "e-mail", "mail"],
    "sms": ["sms", "text message"],
    "push": ["push notification", "push"],
    "in-app": ["in app", "in-app", "banner", "alert"],
}

_GWT = re.compile(r"\b(Given|When|Then|And|But)\b\s+", re.I)
_STATE_FROM_TO = re.compile(r"\bfrom\s+([A-Z][\w ]{1,25}?)\s+to\s+([A-Z][\w ]{1,25}?)(?=\s+(?:when|after|if|once)\b|[.,]|$)")
_STATE_TO = re.compile(r"\b(?:moves to|becomes|returns to|changes to|set to|starts in the|starts in)\s+([A-Z][\w]*(?:\s[A-Z]?[\w]*)?)")
_QUOTED = re.compile(r"[\"“']([^\"”']{2,40})[\"”']")
_STORY = re.compile(r"as an?\s+(.+?),?\s+i (?:want|would like|need|can)\s+(?:to\s+)?(.+?)(?:,?\s+so that\b.*)?$", re.I | re.S)


@dataclass
class Limit:
    kind: str  # min, max, min_exclusive, max_exclusive, range, exact
    low: float | None
    high: float | None
    unit: str = ""

    def describe(self) -> str:
        u = f" {self.unit}" if self.unit else ""
        return {
            "range": f"{fmt_number(self.low)}–{fmt_number(self.high)}{u}",
            "min": f"≥ {fmt_number(self.low)}{u}",
            "min_exclusive": f"> {fmt_number(self.low)}{u}",
            "max": f"≤ {fmt_number(self.high)}{u}",
            "max_exclusive": f"< {fmt_number(self.high)}{u}",
            "exact": f"= {fmt_number(self.low)}{u}",
        }[self.kind]


@dataclass
class CriterionFacts:
    text: str
    given: list[str] = field(default_factory=list)
    when: list[str] = field(default_factory=list)
    then: list[str] = field(default_factory=list)
    limits: list[Limit] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    states: list[str] = field(default_factory=list)
    ui_elements: list[str] = field(default_factory=list)

    @property
    def time_limits(self) -> list[Limit]:
        return [l for l in self.limits if is_time_unit(l.unit)]


@dataclass
class StoryContext:
    actor: str = "user"  # who the story is for: "As a <actor>, I want ..."
    goal: str = ""  # what they want to do; the action of criteria that name none
    criteria: str = ""  # all acceptance criteria of the story, to see what other criteria already cover


def fmt_number(n: float | None) -> str:
    if n is None:
        return "?"
    return str(int(n)) if float(n).is_integer() else str(n)


def _to_num(s: str) -> float:
    return float(s.replace(",", "."))


def is_time_unit(unit: str) -> bool:
    return unit.lower().rstrip("s") in {"second", "sec", "minute", "min", "hour", "day", "week", "millisecond", "m"}


_STOP_UNITS = {"and", "or", "the", "a", "of", "to", "in", "is", "are", "if", "per", "when", "for", "on", "with"}
_SINGULAR_UNITS = {"%", "mb", "gb", "kb", "tb", "px", "ms", "second", "sec", "minute", "min", "hour", "day", "week",
                   "month", "year", "character", "char", "digit", "item", "user", "request", "order", "file", "page",
                   "dollar", "euro", "pound", "usd", "eur", "attempt", "time", "member", "product", "pixel", "line"}
_UNIT_ADJECTIVES = {"distinct", "different", "unique", "active", "consecutive", "concurrent", "simultaneous",
                    "failed", "open", "total", "new", "additional", "more"}


def extract_limits(text: str) -> list[Limit]:
    limits: list[Limit] = []
    taken: list[tuple[int, int]] = []
    for pattern, kind in _LIMIT_PATTERNS:
        for m in pattern.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            groups = [g for g in m.groups()]
            unit = (groups[-1] or "").lower()
            if unit in _UNIT_ADJECTIVES:
                nxt = re.match(r"\s*([a-zA-Z]+)", text[m.end():])
                unit = f"{unit} {nxt.group(1).lower()}" if nxt else unit
            if unit in _STOP_UNITS or not (unit in _SINGULAR_UNITS or unit in _UNIT_ADJECTIVES
                                           or (len(unit) > 2 and unit.endswith("s")) or unit == "percent"):
                unit = ""
            if unit == "percent":
                unit = "%"
            if kind == "range":
                low, high = _to_num(groups[0]), _to_num(groups[1])
                if low > high:
                    continue
                limits.append(Limit("range", low, high, unit))
            elif kind in ("min", "min_exclusive", "exact"):
                limits.append(Limit(kind, _to_num(groups[0]), None, unit))
            else:
                limits.append(Limit(kind, None, _to_num(groups[0]), unit))
            taken.append((m.start(), m.end()))
    limits += [Limit("exact", 1.0, None, m.group(1).lower()) for m in _EVERY_UNIT.finditer(text)]
    return limits


def _find_vocab(text: str, vocab: dict[str, list[str]]) -> list[str]:
    low = text.lower()
    return [key for key, words in vocab.items() if any(re.search(rf"\b{re.escape(w)}\b", low) for w in words)]


def _split_gwt(text: str) -> tuple[list[str], list[str], list[str]]:
    given: list[str] = []
    when: list[str] = []
    then: list[str] = []
    parts = _GWT.split(text)
    bucket = None
    # split() yields [prefix, kw, body, kw, body, ...]
    for kw, body in zip(parts[1::2], parts[2::2]):
        kw = kw.lower()
        if kw in ("given", "when", "then"):
            bucket = {"given": given, "when": when, "then": then}[kw]
        body = body.strip().rstrip(".")
        if bucket is not None and body:
            bucket.append(body)
    return given, when, then


def analyze_criterion(text: str) -> CriterionFacts:
    given, when, then = _split_gwt(text)
    low = text.lower()
    roles = []
    for role in ROLE_VOCAB:
        if re.search(rf"\b{re.escape(role)}s?\b", low) and not any(role in r for r in roles):
            roles.append(role)
    states: list[str] = []
    for a, b in _STATE_FROM_TO.findall(text):
        states += [a.strip(), b.strip()]
    states += [s.strip() for s in _STATE_TO.findall(text)]
    return CriterionFacts(
        text=text,
        given=given,
        when=when,
        then=then,
        limits=extract_limits(text),
        fields=_find_vocab(text, FIELD_VOCAB),
        roles=roles,
        channels=_find_vocab(text, CHANNEL_VOCAB),
        states=list(dict.fromkeys(states)),
        ui_elements=_QUOTED.findall(text),
    )


def analyze_story(story: UserStory) -> StoryContext:
    summary = story.summary.strip().rstrip(".")
    m = _STORY.search(summary)
    if not m:
        return StoryContext(goal=summary)
    return StoryContext(actor=m.group(1).strip(), goal=m.group(2).strip().rstrip("."))


# ------------------------------------------------------------ splitting stories into criteria

_AC_HEADING = re.compile(r"^\s*(?:#+\s*|\*+)?acceptance\s+criteria\s*:?\**\s*$", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)]|AC\s*\d+\s*[:.-])\s*", re.I)
_GWT_LINE = re.compile(r"^\s*(given|when|then|and|but)\b", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def extract_acceptance_criteria(story: UserStory) -> list[str]:
    """Split a story into individual testable criteria.

    Uses the acceptance criteria field when present, otherwise an
    "Acceptance Criteria" section of the description, otherwise the
    description's sentences. Given/When/Then blocks stay together.
    """
    text = story.acceptance_criteria
    if not text.strip():
        lines = story.description.splitlines()
        start = next((i for i, l in enumerate(lines) if _AC_HEADING.match(l)), None)
        if start is not None:
            text = "\n".join(lines[start + 1 :])
        else:
            text = "\n".join(_SENTENCE.split(story.description))

    criteria: list[str] = []
    gwt: list[str] = []
    for raw in text.splitlines():
        line = _BULLET.sub("", raw).strip()
        if not line:
            continue
        m = _GWT_LINE.match(line)
        if m:
            if m.group(1).lower() == "given" and gwt:
                criteria.append(" ".join(gwt))
                gwt = []
            gwt.append(line)
            continue
        if gwt:
            criteria.append(" ".join(gwt))
            gwt = []
        criteria.append(line)
    if gwt:
        criteria.append(" ".join(gwt))
    return criteria or [story.summary]
