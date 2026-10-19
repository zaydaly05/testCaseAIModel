"""What a criterion says should happen, so tests can check the behaviour itself and its other side.

The classifier decides what *kind* of requirement a sentence is and the extractor pulls out facts
(limits, fields, roles); this module reads the logic of the sentence:

    "Given a valid key, when I GET /api/admin/ping, then it returns ok:true"
    -> Scenario(conditions=["a valid key"], action="GET /api/admin/ping", outcomes=["it returns ok:true"])

    "Users must be able to filter search results by date, source, or relevance"
    -> Scenario(action="filter search results by date, source, or relevance", capability=True),
       options=["date", "source", "relevance"]

It can also state the opposite of a condition or an outcome ("a valid key" -> "an invalid key",
"the entry is stored" -> "the entry is not stored"), which the test design uses for the other side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------- patterns

REQUEST = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\s+(/[^\s,;]*)")
_KEYWORD = re.compile(r"(?:^|(?<=[,;.]\s))(?i:given|when|then)\b\s*|\b(?:Given|When|Then|And|But)\b\s*")
_STARTS_GWT = re.compile(r"^\s*(?:given|when|then)\b|\b(?:Given|When|Then)\b")
_ACTION_LIKE = re.compile(
    r"^(?:I\s+(?!am\b|have\b|has\b|was\b|can\b|see\b|receive\b|get\b)\w+"
    r"|(?-i:GET|POST|PUT|PATCH|DELETE)\b|an?\s+(?-i:GET|POST|PUT|PATCH|DELETE)\b"
    r"|the page (?:is )?(?:re)?load(?:s|ed)?\b|[\w ]{0,30}\b(?:posts|sends|calls|submits|clicks|uploads)\b|sync runs)", re.I)
_LOCATION = re.compile(r"^(?:the |any |an? )?[\w ]{0,25}\b(?:screen|page|tab|view|modal|dialog|hero|terminal|widget|menu)s?$", re.I)
_CAPABILITY = re.compile(
    r"^(?P<subject>.+?)\s+(?:must|should|shall|can|will|may)\s+(?P<neg>not\s+)?(?:be\s+(?:able|allowed)\s+to)\s+(?P<act>.+)$", re.I)
# "Only admins can delete users": a modal without "be able to" counts as a capability when a person is the subject.
_ROLE_CAPABILITY = re.compile(
    r"^(?P<subject>(?:only\s+|all\s+|the\s+)?[\w ]{0,30}?\b(?:users?|admins?|administrators?|customers?|shoppers?|members?|"
    r"managers?|instructors?|teachers?|students?|staff|owners?|editors?|viewers?|guests?|visitors?|patients?|doctors?|"
    r"professionals?|agents?|I|we))\s+(?:can|may|must|should|shall|will)\s+(?P<neg>not\s+)?(?P<act>[a-z].+)$", re.I)
_ALLOW = re.compile(r"^(?:the\s+)?(?:system|app|application|platform|website|site|service|tool|portal)\b.*?\b"
                    r"(?:allow|enable|let|permit)s?\s+(?P<subject>.+?)\s+to\s+(?P<act>.+)$", re.I)
_IF_THEN = re.compile(r"^(?P<kw>if|whenever|once|when|after|on)\s+(?P<cond>.+?),\s*(?:then\s+)?(?P<out>.+)$", re.I)
_GERUND = re.compile(r"^(?P<act>[A-Z][a-z]+ing\b.+?)\s+(?P<out>(?:\w+s|is|are|does|do|will|must|should|can(?:not)?)\b.+)$")
_STATUS = re.compile(r"(?<![\w./-])([2-5]\d\d|[2-5]xx)(?![\w-])", re.I)
_STATUS_CONTEXT = re.compile(r"\b(?:return|status|respond|rejected|reject|error|answer|code|page|with)\w*|\(|/", re.I)
_SUB_SCENARIO = re.compile(r"^(?P<cond>(?:without|with|if|for|given|when)\s+.+?),\s*(?P<out>.+)$", re.I)
_DEFAULT = re.compile(r"\b(?:a |an |the )?(?:missing|empty|no|omitted)\s+(?P<field>[\w ]{1,20}?)\s+defaults?\s+to\s+(?P<value>[^\s,;.]+)", re.I)

_NEGATIVE = re.compile(
    r"\b(?:not|no|nothing|never|cannot|can't|isn't|aren't|doesn't|don't|won't|without|reject\w*|refus\w*|den(?:y|ied)"
    r"|block\w*|errors?|fail\w*|invalid|unauthori[sz]ed|forbidden|ignored|prevent\w*)\b|\b[45](?:\d\d|xx)\b", re.I)
_HARMLESS_NEGATIVE = re.compile(r"\bwithout (?:any )?(?:errors?|duplicates?|delay|data loss|interruption)\b|\bno (?:errors?|data loss)\b", re.I)

WRITE_VERBS = {"add", "create", "save", "update", "edit", "upload", "delete", "remove", "submit", "book", "assign",
               "import", "post", "put", "patch", "change", "set", "store", "approve", "reject", "star", "follow",
               "subscribe", "register", "rate", "review", "write", "modify", "rebuild", "restore", "record", "log"}
REVERSE_VERBS = {"add": "remove", "save": "remove", "create": "delete", "upload": "delete", "follow": "unfollow",
                 "subscribe": "unsubscribe", "enable": "disable", "open": "close", "lock": "unlock", "like": "unlike",
                 "star": "unstar", "book": "cancel", "assign": "unassign", "bookmark": "remove", "join": "leave",
                 "delete": "restore", "favorite": "unfavorite", "favourite": "unfavourite", "pin": "unpin"}

# Opposites used to state the other side of a condition: the first one found in the text is swapped.
_ANTONYMS = {
    "invalid": "valid", "valid": "invalid", "wrong": "correct", "correct": "wrong", "incorrect": "correct",
    "unknown": "existing", "unavailable": "available", "is available": "is unavailable", "unreachable": "reachable",
    "is reachable": "is unreachable", "is offline": "is online", "is online": "is offline", "is down": "is up", "are down": "are up",
    "is up": "is down", "disabled": "enabled", "enabled": "disabled", "is closed": "is open", "is open": "is closed",
    "are closed": "are open", "are open": "are closed", "empty": "non-empty", "missing": "present",
    "provided": "missing", "present": "missing", "unregistered": "registered", "registered": "unregistered",
    "out-of-stock": "in-stock", "in-stock": "out-of-stock", "out of stock": "in stock", "in stock": "out of stock",
    "signed out": "signed in", "signed in": "signed out", "logged out": "logged in", "logged in": "logged out",
    "unauthenticated": "authenticated", "authenticated": "unauthenticated", "unauthorized": "authorized",
    "unauthorised": "authorised", "inactive": "active", "active": "inactive", "private": "public",
    "public": "private", "outside": "inside", "inside": "outside", "without": "with", "expired": "valid",
    "locked": "unlocked", "unlocked": "locked", "already in": "not yet in", "not yet in": "already in",
    "fails": "succeeds", "succeeds": "fails", "exceeds": "stays within", "unrecognized": "recognized",
    "unrecognised": "recognised", "is long": "is short", "is short": "is long", "unverified": "verified",
    "verified": "unverified",
}
# Whole phrases whose opposite is not a one-word swap.
_PHRASE_OPPOSITES = [(re.compile(r"(?i)\bany (\w+?)s? (?:is|are) (?:empty|missing|blank)\b"), r"every \1 is filled"),
                     (re.compile(r"(?i)\ball (?:the )?(?:\w+ )?(\w+?)s are (?:filled|provided|present)\b"), r"a \1 is empty")]
_ANTONYM_RE = re.compile(r"(?<![\w-])(" + "|".join(sorted(map(re.escape, _ANTONYMS), key=len, reverse=True)) + r")(?![\w-])", re.I)
_AUX = re.compile(r"\b(am|is|are|was|were|can|will|should|must|has|have|does|do)\b(?!\s+not\b)", re.I)
_VERBS = ("return", "show", "display", "open", "appear", "list", "send", "receive", "get", "contain", "add", "increase",
          "decrease", "change", "update", "work", "succeed", "fill", "advance", "render", "reach", "serve", "echo", "store",
          "save", "create", "delete", "remove", "load", "redirect", "include", "match", "count", "keep", "accept", "allow",
          "print", "play", "download", "upload", "exist", "answer", "reply", "animate", "move", "become", "use", "reflect")
_VERB_3RD = re.compile(r"\b(" + "|".join(v + ("es" if v.endswith(("sh", "ch", "o", "x")) else "s") for v in _VERBS) + r")\b", re.I)
_VERB_BASE = re.compile(r"\b(" + "|".join(_VERBS) + r")\b", re.I)


# ----------------------------------------------------------------------------- data

@dataclass
class Scenario:
    conditions: list[str] = field(default_factory=list)
    action: str = ""
    outcomes: list[str] = field(default_factory=list)
    capability: bool = False  # "X must be able to <action>": the outcome is that the action is possible
    derived: bool = False  # split off another scenario's outcome ("...; without admin, 401")

    @property
    def request(self) -> tuple[str, str] | None:
        m = REQUEST.search(self.action)
        return (m.group(1), m.group(2).rstrip(".)")) if m else None

    @property
    def negative(self) -> bool:
        return any(is_negative(o) for o in self.outcomes)


@dataclass
class Behaviour:
    text: str
    scenarios: list[Scenario]
    options: list[str] = field(default_factory=list)  # enumerated alternatives: "by date, source, or relevance"
    options_text: str = ""  # the listed options as written in the first scenario's action ...
    options_in_outcome: bool = False  # ... or in its first outcome
    default: tuple[str, str] | None = None  # ("rating", "5") from "a missing rating defaults to 5"
    clamp: bool = False  # out-of-range values are clamped instead of rejected
    truncate: bool = False  # "only the first 12 are shown": values above the limit are cut off, not rejected
    shows_count: bool = False

    @property
    def main(self) -> Scenario:
        return self.scenarios[0]

    @property
    def verb(self) -> str:
        """The main verb of the first action, in base form ("save", "delete", "get")."""
        return action_verb(self.main.action)

    @property
    def writes(self) -> bool:
        req = self.main.request
        return (req is not None and req[0] != "GET") or (req is None and self.verb in WRITE_VERBS)


# ----------------------------------------------------------------------------- parsing

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip(",;").rstrip(".").strip()


def _split_outcomes(body: str) -> list[str]:
    parts = []
    for part in re.split(r";\s*|\.\s+(?=[A-Z])", body):
        pieces = re.split(r",?\s+and\s+(?=(?:the|a|an|it|I|they)\s)", part)
        if len(pieces) > 1 and all(_AUX.search(x) or _VERB_3RD.search(x) for x in pieces[1:]):
            parts += pieces
        else:
            parts.append(part)
    outcomes = [_clean(p) for p in parts if _clean(p)]
    return [re.sub(r"(?i)^(?:the response |it )?returns? ([2-5](?:\d\d|xx))$", r"the response status is \1", o)
            for o in outcomes]


def _parse_gwt(text: str) -> list[Scenario]:
    scenarios: list[Scenario] = []
    current = Scenario()
    last = "given"
    marks = list(_KEYWORD.finditer(text))
    for i, m in enumerate(marks):
        kw = m.group(0).strip().lower()
        body = _clean(text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)])
        if not body:
            continue
        kw = last if kw in ("and", "but") else kw
        if kw in ("given", "when") and current.outcomes:  # a new Given / When after a Then starts the next scenario
            scenarios.append(current)
            current = Scenario()
        if kw == "given":
            current.conditions.append(body)
        elif kw == "when":
            current.action = f"{current.action} and {body}" if current.action else body
        else:
            current.outcomes += _split_outcomes(body)
        last = kw
    scenarios.append(current)
    return [s for s in scenarios if s.conditions or s.action or s.outcomes]


def _complete(s: Scenario, goal: str) -> Scenario:
    """Give every scenario an action: a Given that is really an event, a request named in the outcome, or the goal."""
    if not s.outcomes and s.conditions and (m := re.match(r"(.+?),\s+(.+)$", s.conditions[-1])):
        s.conditions[-1], s.outcomes = _clean(m.group(1)), _split_outcomes(m.group(2))
    if not s.action:
        acting = [c for c in s.conditions if _ACTION_LIKE.match(c)]
        if acting:
            s.action = acting[-1]
            s.conditions.remove(acting[-1])
        elif s.outcomes and (m := REQUEST.search(s.outcomes[0])) and m.start() < 3:
            s.action = m.group(0)
            rest = _clean(s.outcomes[0][m.end():])
            s.outcomes[0:1] = _split_outcomes(f"the response {rest}") if rest else ["the request succeeds"]
        elif s.conditions and re.match(r"(?i)^(?:a|an|some)\s", s.conditions[-1]) and not (
                _AUX.search(s.conditions[-1]) or _VERB_3RD.search(s.conditions[-1])):
            s.action = f"{goal} with {s.conditions.pop()}"  # "a rating outside 1-5" is the data the action uses
        else:
            s.action = goal
    return s


def _parse_sentence(text: str, goal: str) -> list[Scenario]:
    if m := (_CAPABILITY.match(text) or _ROLE_CAPABILITY.match(text)):
        subject, act = _clean(m.group("subject")), _clean(m.group("act"))
        if m.group("neg"):
            return [Scenario(conditions=[f"signed in as {subject}"], action=act, outcomes=[f"{subject} cannot {act}"])]
        return [Scenario(action=act, outcomes=[f"{subject} can {act}"], capability=True)]
    if m := _ALLOW.match(text):
        return [Scenario(action=_clean(m.group("act")), outcomes=[f"{_clean(m.group('subject'))} can {_clean(m.group('act'))}"],
                         capability=True)]
    if m := _IF_THEN.match(text):
        cond, out = _clean(m.group("cond")), _clean(m.group("out"))
        if m.group("kw").lower() in ("when", "after", "on"):
            return [Scenario(action=cond, outcomes=_split_outcomes(out))]
        return [Scenario(conditions=[cond], action=goal, outcomes=_split_outcomes(out))]
    if m := _GERUND.match(text):
        act = m.group("act")
        verb = act.split()[0]
        base = verb[:-3]
        if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "ls" and base.lower() != "add":
            base = base[:-1]  # Submitting -> Submit
        elif base.endswith(("at", "iz", "ov", "ir", "ag", "uc", "lud", "ag", "us")):
            base += "e"  # Updating -> Update, Using -> Use
        return [Scenario(action=base + act[len(verb):], outcomes=[f"it {_clean(m.group('out'))}"])]
    return [Scenario(action=goal, outcomes=[_clean(text)])]


def _sub_scenarios(s: Scenario) -> list[Scenario]:
    """'logs are returned; without admin, 401' -> a second scenario: condition 'without admin', outcome '401'."""
    extra, keep = [], []
    for o in s.outcomes:
        m = _SUB_SCENARIO.match(o)
        if m and keep:
            cond, out = _clean(m.group("cond")), _clean(m.group("out"))
            cond = re.sub(r"^(?:given|when|if)\s+", "", cond, flags=re.I)
            out = f"the response status is {out}" if re.fullmatch(r"[2-5](?:\d\d|xx)", out, re.I) else out
            action = s.action
            if who := re.match(r"(?i)^(?:without|not)\s+(?:an?\s+)?(\w+)", cond):  # "without admin": drop "as admin"
                action = re.sub(rf"(?i)\s+as\s+(?:an?\s+)?{who.group(1)}\b", "", action)
            extra.append(Scenario(conditions=s.conditions + [cond], action=action, outcomes=[out], derived=True))
        else:
            keep.append(o)
    s.outcomes = keep
    return [s, *extra]


# A list item: one to three words, none of them a word that connects the list to the rest of the sentence.
_STOP = r"(?!(?:and|or|of|to|for|in|on|with|from|their|his|her|its|my|your|our|as|at|by|through|via|before|after|based|"
_STOP += r"needed|that|which|the|a|an)\b)"
_ITEM = rf"(?:(?:a|an|the)\s+)?{_STOP}[A-Za-z][\w\-/]*(?:\s+(?:of\s+)?{_STOP}[\w\-/]+){{0,2}}"  # "number of guests"
_LIST_LEAD = re.compile(r"\b(by|such as|e\.g\.,?|including|like|via|through)\s+\(?", re.I)
_LED_LIST = re.compile(rf"{_ITEM}(?:,\s+{_ITEM}){{0,5}}(?:,?\s+(?:and|or)\s+{_ITEM})?")
_BARE_LIST = re.compile(rf"{_ITEM}(?:,\s+{_ITEM}){{1,5}},?\s+(?:and|or)\s+{_ITEM}")  # three or more, no lead word
_VAGUE = re.compile(r"^(?:other|more|various|etc|similar|any|several|many)\b", re.I)
_INNER_LEAD = re.compile(r"\b(?:such as|including|e\.g\.|like)\b", re.I)


def _options(text: str, leads: tuple[str, ...] = ()) -> tuple[list[str], str]:
    """Listed alternatives ("by date, source, or relevance"); `leads` restricts the word that introduces them.

    Returns the options and the list as written, so a test can put one option in its place.
    """
    candidates = []
    for lead in _LIST_LEAD.finditer(text):
        if (not leads or lead.group(1).lower() in leads) and (m := _LED_LIST.match(text, lead.end())):
            candidates.append((m.start(), m.group(0), True))
    if not leads:
        candidates += [(m.start(), m.group(0), False) for m in _BARE_LIST.finditer(text)]
    for _, raw, led in sorted(candidates, key=lambda c: (not c[2], c[0])):  # lists after a lead word first
        if _INNER_LEAD.search(raw):
            continue  # "history, including medications": the real list starts after the inner lead word
        items = [i.strip() for i in re.split(r",\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+", raw) if i.strip()]
        if not led and len(items[0].split()) > 1:  # "I reply STOP, ..." -> the options start at "STOP"
            head = items[0].rsplit(" ", 1)[0] + " "
            items[0] = items[0][len(head):]
            raw = raw[len(head):]
        items = [i for i in items if not _VAGUE.match(i)]  # "other relevant criteria" is no option to test
        if text.startswith(raw) and len(items) >= 2:  # a list of verbs: "create, edit, and publish documentation"
            if any(len(i.split()) > 1 for i in items[:-1]):
                continue
            words = items[-1].split()
            items = [f"{verb} {' '.join(words[1:])}".strip() for verb in items[:-1] + [words[0]]]
        if (2 <= len(items) <= 6 and all(len(i.split()) <= 4 for i in items)
                and not any(_AUX.search(i) or _VERB_3RD.search(i) for i in items)):
            return items, raw
    return [], ""


def analyze_behaviour(text: str, goal: str) -> Behaviour:
    text = _clean(text)
    scenarios = _parse_gwt(text) if _STARTS_GWT.search(text) else _parse_sentence(text, goal)
    scenarios = [x for s in scenarios for x in _sub_scenarios(_complete(s, goal))] or [Scenario(action=goal, outcomes=[text])]
    behaviour = Behaviour(text=text, scenarios=scenarios)
    if scenarios[0].action != goal:  # options of the story title are not this criterion's
        behaviour.options, behaviour.options_text = _options(scenarios[0].action)
    if not behaviour.options and scenarios[0].outcomes and re.search(r"\b(?:filter|search|sort|match|find)\w*",
                                                                     scenarios[0].outcomes[0], re.I):
        # "results are filtered by name, stack or description"
        behaviour.options, behaviour.options_text = _options(scenarios[0].outcomes[0], leads=("by",))
        behaviour.options_in_outcome = bool(behaviour.options)
    if m := _DEFAULT.search(text):
        behaviour.default = (m.group("field").strip(), m.group("value"))
        for sc in scenarios:
            sc.outcomes = [o for o in sc.outcomes if not _DEFAULT.search(o)] or sc.outcomes
    behaviour.clamp = bool(re.search(r"\bclamp\w*|\bcapped\b|\brounded to the (?:nearest )?limit", text, re.I))
    behaviour.truncate = bool(re.search(r"\b(?:only )?the (?:first|top|latest|last) \d+\b|\bshows? (?:at most|up to) \d+", text, re.I))
    behaviour.shows_count = bool(re.search(r"\bcount of (?:the )?total\b|\btotal count\b|\bwith a count\b", text, re.I))
    return behaviour


# ----------------------------------------------------------------------------- wording helpers

def is_negative(text: str) -> bool:
    return bool(_NEGATIVE.search(_HARMLESS_NEGATIVE.sub("", text)))


def statuses(text: str) -> list[str]:
    """HTTP status codes the text expects ("returns 401", "rejected (401/403)", "4xx")."""
    found = []
    for m in _STATUS.finditer(text):
        before = text[: m.start()].rstrip()
        window = text[max(0, m.start() - 25): m.end() + 12]
        if _STATUS_CONTEXT.search(window) or not before or before.endswith(","):
            found.append(m.group(1).lower())
    return list(dict.fromkeys(found))


def fix_articles(text: str) -> str:
    text = re.sub(r"\b([Aa])n (?=[^aeiouAEIOU\W])", r"\1 ", text)
    return re.sub(r"\b([Aa]) (?=[aeiouAEIOU])", r"\1n ", text)


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def capitalize(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _swap(m: re.Match) -> str:
    word = m.group(0)
    new = _ANTONYMS[word.lower()]
    return capitalize(new) if word[:1].isupper() else new


def flip_antonym(text: str) -> str | None:
    """Swap the first word that has an opposite ("valid" -> "invalid"); 'missing or wrong' -> 'present and correct'.

    Returns "" when only part of an "X or Y" can be swapped ("unknown or already-resolved"): there is
    no clean opposite, so no test should be built on one.
    """
    for pattern, repl in _PHRASE_OPPOSITES:
        if pattern.search(text):
            return fix_articles(pattern.sub(repl, text, count=1))
    matches = list(_ANTONYM_RE.finditer(text))
    if not matches:
        return None
    first = matches[0]
    out = text[: first.start()] + _swap(first)
    end = first.end()
    if len(matches) > 1 and re.fullmatch(r"\s+or\s+", text[end: matches[1].start()]):
        out += " and " + _swap(matches[1])
        end = matches[1].end()
    elif re.match(r"\s+or\s+\w", text[end:]):
        return ""
    return fix_articles(out + text[end:])


def _negate_joined(rest: str) -> str:
    """'... saved and appears in the list' -> '... saved and does not appear in the list'."""
    def repl(m: re.Match) -> str:
        word = m.group(1)
        base = word[:-2] if word.lower().endswith(("shes", "ches", "oes", "xes")) else word[:-1]
        return f"and does not {base}"
    return re.sub(r"\band (\w+)\b", lambda m: repl(m) if _VERB_3RD.fullmatch(m.group(1)) else m.group(0), rest)


def negate(clause: str) -> str:
    """State that the clause does not hold: 'the entry is stored' -> 'the entry is not stored' (and back).

    The main clause is the one changed: the earliest negation or verb decides, so
    'they are ignored so the bot never answers' -> 'they are not ignored so the bot never answers'.
    """
    c = _clean(clause)
    verb = min((m.start() for m in (_AUX.search(c), _VERB_3RD.search(c)) if m), default=len(c))
    for pattern, repl in ((r"\bcannot\b", "can"), (r"\bcan't\b", "can"), (r"\bnever\s+", ""),
                          (r"\b(am|is|are|was|were|will|should|must|does|do|has|have) not\b", r"\1"),
                          (r"\b(has|have) no\b", r"\1"), (r"\bnothing\b", "something")):
        if (m := re.search(pattern, c, re.I)) and m.start() <= verb:
            return fix_articles(re.sub(pattern, repl, c, count=1, flags=re.I))
    if m := re.match(r"(?i)^no\s+(\w+)", c):
        rest = c[m.start(1):]
        return capitalize(rest) if m.group(1).endswith("s") else fix_articles(f"a {rest}")
    if (m := re.search(r"(?i)\bno\s+(?=\w)", c)) and m.start() <= verb:
        return fix_articles(c[: m.start()] + "a " + c[m.end():])
    if (m := _VERB_3RD.search(c)) and not ((a := _AUX.search(c)) and a.start() < m.start()):
        word = m.group(1)
        base = word[:-2] if word.lower().endswith(("shes", "ches", "oes", "xes")) else word[:-1]
        return c[: m.start()] + f"does not {base.lower()}" + c[m.end():]
    if m := re.match(r"(?i)^(I|we|you|they)\s+(?!am\b|are\b|can\b|will\b|do\b|have\b)(\w+)", c):
        return c[: m.start(2)] + "do not " + c[m.start(2):]
    if m := re.search(r"(?i)\b(have|has)\s+(?:a|an|the)\s+", c):
        return c[: m.start()] + f"{m.group(1)} no " + c[m.end():]
    if re.match(r"(?i)^(?:a|an|some|any)\s", c) and not (_AUX.search(c) or _VERB_3RD.search(c)):
        return "no " + re.sub(r"(?i)^(?:a|an|some|any)\s+", "", c)
    if m := _AUX.search(c):
        if m.group(1).lower() == "can":
            return c[: m.start()] + "cannot" + c[m.end():]
        return c[: m.end()] + " not" + _negate_joined(c[m.end():])
    if m := _VERB_3RD.search(c):
        word = m.group(1)
        base = word[:-2] if word.lower().endswith(("shes", "ches", "oes", "xes")) else word[:-1]
        return c[: m.start()] + f"does not {base.lower()}" + c[m.end():]
    if m := _VERB_BASE.search(c):
        return c[: m.start()] + "do not " + c[m.start():]
    if not re.search(r"\b(?:I|it|they|we|you)\b", c):  # a noun phrase: "a pending change" -> "no pending change"
        return "no " + re.sub(r"(?i)^(?:a|an|the|some|any)\s+", "", c)
    return f"it is not the case that {c}"


def flip_condition(condition: str) -> str | None:
    """The opposite of a condition, or None when it is a place or an event rather than a state."""
    if _LOCATION.match(condition) or _ACTION_LIKE.match(condition):
        return None
    flipped = flip_antonym(condition)
    if flipped == "":
        return None
    return flipped or negate(condition)


def action_verb(action: str) -> str:
    if req := REQUEST.search(action):
        return {"GET": "get", "POST": "create", "PUT": "update", "PATCH": "update", "DELETE": "delete"}[req.group(1)]
    words = re.sub(r"(?i)^(?:I|the user|users?|they|we|you)\s+", "", action).split()
    return words[0].lower().rstrip(",") if words else ""


def as_step(action: str) -> str:
    """An action as a test step: 'I click Download CV' -> 'Click Download CV', 'GET /x' -> 'Send GET /x'."""
    a = re.sub(r"(?i)^I\s+", "", _clean(action))
    a = re.sub(r"^(?:[Aa]n?\s+)?(?=(?:GET|POST|PUT|PATCH|DELETE)\b)", "Send ", a)  # HTTP methods are upper case
    return capitalize(a)


def as_precondition(condition: str) -> str:
    """'I am admin' -> 'Signed in as an admin'; 'I am not admin' -> 'Signed in as a user who is not an admin'."""
    c = _clean(condition)
    if m := re.match(r"(?i)^(?:I am not|not|without) (?:an? )?(admin|administrator|owner|manager|editor|moderator|\w+ rights)$", c):
        role = re.sub(r"(?i)\s+rights$", "", m.group(1))
        return f"Signed in as a user who is not {article(role)} {role.lower()}"
    if m := re.match(r"(?i)^I am not (?:an? )?(\w+)$", c):
        return f"Signed in as a user who is not {article(m.group(1))} {m.group(1).lower()}"
    if (m := re.match(r"(?i)^(?:I am|signed in as) (?:an? |the )?(\w+)$", c)) and m.group(1).lower() not in (
            "on", "in", "at", "here", "there", "logged", "signed", "online", "offline"):
        return f"Signed in as {article(m.group(1))} {m.group(1).lower()}"
    return capitalize(c)
