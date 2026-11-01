"""The check that decides whether an answer may be returned as it is.

The model writes the answer; it does not get to say whether the answer is supported. That
is decided here, in code, by the same kind of check the demonstrator's summariser (CAx)
applies to its summaries: every substantive figure in the answer must be a figure some
tool actually returned, or that the question itself contained.

The check is one-sided on purpose. An unsupported figure is evidence of a problem; the
absence of one is not proof that the answer is right, because a sentence can be wrong
without containing a number. The record says exactly that much and no more.

The verdict is the output of a named, versioned rule applied to recorded values, so it can
be replayed from the record the agent writes (see record.py).

Rule v2 adds what testing with off-topic questions showed v1 could not see: an
answer that consulted nothing (v1 accepted it if it had no figures), an answer that repeats
an earlier answer in the conversation (the figures pass, because the model fetched them
again), and a question the agent declined because the records do not cover it.

Rule v3 adds the role of a figure: every figure the tools return is labelled
with where it was (``groups.tasks``, ``run.seconds``), and a real figure stated in a role
its labels rule out is sent back (see :func:`misplaced`). Built narrow on purpose: replayed
on the three frozen tests it flags 6 of 126 accepted-but-wrong answers and 0 of 2,231
right ones (in sample: the same runs were used to tune it).

Numbers written as words are read as figures by both checks: "the most common
dtype is list, which appears in three entities" passed as "no figures to check" beside a
dtype held by 30. The rule is unchanged; it now sees what it was not shown before.

A count below the floor is checked when it counts something ("3 tasks", "2 workers"): such a
number is a claim about the records, not prose. Each list a tool returned counts as a fact of
its own length, so a count of returned items is supported. The small numbers still left
unchecked are counted, so a client never says "nothing to check" when there was.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

#: A figure standing on its own: not digits inside an identifier ("example_GR0_0",
#: "MSELoss_Training") and not a version string. Signs, thousands separators, decimals and
#: exponents are part of the figure.
NUMERAL = re.compile(r"(?<![\w.])-?\d(?:[\d,]*\d)?(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\.?\w)")

#: Whole numbers below this are prose ("3 activities", "the 2 runs", "step 1") as often as
#: they are claims. Checking them would reject faithful answers; the same floor as CAx.
#: Decimals are always checked: nobody writes 0.0963 as a figure of speech.
INTEGER_FLOOR = 10

#: An ISO 8601 timestamp. Its parts are facts too: an answer that writes
#: "2026-05-20T10:10:05" as "May 20, at 10:10:05" states the day, 20, which NUMERAL alone
#: never extracts from the timestamp because it is glued to the month by a hyphen.
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?")

#: An answer this similar to an earlier answer in the conversation (word-set Jaccard), given
#: to a different question, is taken to answer the earlier question again.
REPEAT_THRESHOLD = 0.8

#: The number of an ordered-list item ("10. ", "11) ") at the start of a line: a position in
#: the answer's own list, not a claim. Without this, a list of ten or more items failed the
#: check on its own numbering (found by the evaluation).
LIST_MARKER = re.compile(r"^(\s*)\d+[.)](?=\s)", re.MULTILINE)

#: Cardinal numbers written as words. "one" alone is left as it is: it is a pronoun ("the
#: most common one") as often as a count, and as a count it is below the floor anyway.
_UNIT_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
)  # fmt: skip
_TEN_WORDS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_UNITS = {w: i for i, w in enumerate(_UNIT_WORDS)}
_TENS = {w: 10 * i for i, w in enumerate(_TEN_WORDS, 2)}
_SCALES = {"hundred": 100, "thousand": 1000}
_NUMBER_WORD = "|".join([*_UNITS, *_TENS, *_SCALES])
SPELLED = re.compile(
    rf"\b(?:{_NUMBER_WORD})(?:(?:[\s-]+|\s+and\s+)(?:{_NUMBER_WORD}))*\b", re.IGNORECASE
)


def _spelled_value(words: str) -> int:
    total = current = 0
    for word in re.findall(r"[a-z]+", words.lower()):
        if word in _UNITS:
            current += _UNITS[word]
        elif word in _TENS:
            current += _TENS[word]
        elif word == "hundred":
            current = max(current, 1) * 100
        elif word == "thousand":
            total += max(current, 1) * 1000
            current = 0
    return total + current


def spelled_as_digits(text: str) -> str:
    """The text with every number written as words in digits ("thirty-six" as "36"), so
    the checks read a count however the answer wrote it."""

    def replace(match: re.Match[str]) -> str:
        if match.group().lower() == "one":
            return match.group()
        return str(_spelled_value(match.group()))

    return SPELLED.sub(replace, text)


#: The things a small count can count. A number below the floor followed (within two words)
#: by one of these is a claim and is checked: "3 move_mean tasks", "2 failed workers".
COUNTED = re.compile(
    r"\s+(?:[\w`.-]+\s+){0,2}?(?:task|activit|entit|node|worker|step|run|file|failure|"
    r"error|agent|decision|operator|record|document|metric|parameter|model|dataset|epoch|"
    r"relation|value|type|group|module|process|executor|input|output|cell)\w*\b",
    re.IGNORECASE,
)

Scalar = str | int | float | bool | None


@dataclass(frozen=True)
class Rule:
    """A decision rule: a pure function from recorded evidence to a verdict.

    The name carries a version. A change in the logic is a new name, so a record that
    names a rule names exactly one function.
    """

    name: str
    apply: Callable[[Mapping[str, Scalar]], str]


def _grounding_rule(evidence: Mapping[str, Scalar]) -> str:
    """ACCEPT an answer that asserts no figure the tools did not return.

    An answer that states figures but consulted no tool cannot have taken them from the
    records, whatever they are, so it is not accepted either.
    """
    if evidence["ungrounded_figures"]:
        return "REVISE"
    if evidence["figures_checked"] and not evidence["tool_calls"]:
        return "REVISE"
    return "ACCEPT"


def _grounding_rule_v2(evidence: Mapping[str, Scalar]) -> str:
    """OUT_OF_SCOPE if the agent declined: the records do not cover the question.

    Otherwise REVISE an answer that states a figure the tools did not return, that rests
    on no successful lookup at all, or that repeats an earlier answer in the conversation
    (``repeat_similarity`` at or above REPEAT_THRESHOLD). ACCEPT the rest.
    """
    if evidence["declined"]:
        return "OUT_OF_SCOPE"
    if evidence["ungrounded_figures"]:
        return "REVISE"
    if not evidence["data_lookups"]:
        return "REVISE"
    if float(evidence["repeat_similarity"] or 0) >= REPEAT_THRESHOLD:
        return "REVISE"
    return "ACCEPT"


def _grounding_rule_v3(evidence: Mapping[str, Scalar]) -> str:
    """v2, and REVISE an answer that states a returned figure in a role the tools did not
    give it (``misplaced_figures``, see :func:`misplaced`): the busiest worker's "4 tasks"
    when the same lookup returned a worker with 5."""
    verdict = _grounding_rule_v2(evidence)
    if verdict == "ACCEPT" and evidence.get("misplaced_figures"):
        return "REVISE"
    return verdict


GROUNDING_RULE_V1 = Rule("qa.grounding/v1", _grounding_rule)
GROUNDING_RULE_V2 = Rule("qa.grounding/v2", _grounding_rule_v2)
GROUNDING_RULE = Rule("qa.grounding/v3", _grounding_rule_v3)

#: Every rule a record may name, so an old record still replays with the rule it names.
RULES = {r.name: r for r in (GROUNDING_RULE_V1, GROUNDING_RULE_V2, GROUNDING_RULE)}


def reasons(
    evidence: Mapping[str, Scalar],
    ungrounded: list[str],
    misplaced_figures: Iterable[Mapping[str, Any]] = (),
) -> list[str]:
    """Why a verdict is REVISE, in words, for the user and for the model's next try."""
    out = []
    if ungrounded:
        out.append("figures not found in the records: " + ", ".join(ungrounded))
    out += misplaced_reasons(misplaced_figures)
    if not evidence["data_lookups"]:
        out.append("nothing was looked up in the records")
    if float(evidence["repeat_similarity"] or 0) >= REPEAT_THRESHOLD:
        out.append("it repeats an earlier answer in this conversation")
    return out


def misplaced_reasons(misplaced_figures: Iterable[Mapping[str, Any]]) -> list[str]:
    """Each misplaced figure (see :func:`misplaced`), in words."""
    return [
        f"{m['figure']} is given as {m['said_as']}, but the records return it as"
        f" {', '.join(m['returned_as'])}" + (f" ({m['but']})" if m.get("but") else "")
        for m in misplaced_figures
    ]


WORD = re.compile(r"\w+")


def similarity(a: str, b: str) -> float:
    """Jaccard similarity of the two texts' word sets, case-insensitive."""
    wa, wb = set(WORD.findall(a.lower())), set(WORD.findall(b.lower()))
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def _specifics(text: str) -> set[str]:
    """The words that name or measure something: identifiers and figures (any word with a
    digit or an underscore, e.g. ``example_GR0_2``, ``ca1``, ``0.0281``)."""
    return {w for w in WORD.findall(text.lower()) if any(c.isdigit() or c == "_" for c in w)}


def repeat_similarity(answer: str, question: str, history: Iterable[tuple[str, str]]) -> float:
    """How closely ``answer`` repeats an earlier answer given to a different question.

    Skipped, because they are not repeats: an earlier turn whose question is (nearly) the
    same one (asking again and getting the same answer is not a fault), and an earlier
    answer that lacks an identifier or figure this answer states (the same sentence about
    another run, "example_GR0_0 used Adam" after "example_GR0_2 used Adam", is new
    information however similar the wording).
    """
    specifics = _specifics(answer)
    best = 0.0
    for earlier_question, earlier_answer in history:
        if similarity(question, earlier_question) >= REPEAT_THRESHOLD:
            continue
        if not specifics <= _specifics(earlier_answer):
            continue
        best = max(best, similarity(answer, earlier_answer))
    return round(best, 3)


def replay(rule_name: str, rule_inputs: str | Mapping[str, Scalar]) -> str:
    """Re-derive a recorded verdict from the rule's name and its recorded inputs."""
    inputs = json.loads(rule_inputs) if isinstance(rule_inputs, str) else rule_inputs
    return RULES[rule_name].apply(inputs)


def _to_float(literal: str) -> float | None:
    try:
        value = float(literal.replace(",", ""))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _decimals(literal: str) -> int:
    mantissa = literal.lower().split("e")[0]
    return len(mantissa.split(".")[1]) if "." in mantissa else 0


def facts_from(values: Iterable[Any]) -> list[float]:
    """Every number in a set of tool results, including numbers written inside strings
    (a justification that says "21634 bytes" makes 21634 a fact)."""
    found: list[float] = []

    def walk(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, int | float):
            if math.isfinite(value):
                found.append(float(value))
        elif isinstance(value, str):
            for match in NUMERAL.finditer(value):
                number = _to_float(match.group())
                if number is not None:
                    found.append(number)
            for match in TIMESTAMP.finditer(value):
                parts = re.split(r"[-T :]", match.group())
                found.extend(float(part) for part in parts)
                # A clock is written with whole seconds: 16:31:55 for 16:31:55.628597. As a
                # rounded figure "55" is 0.63 off and was flagged; the whole
                # seconds are the same fact.
                if "." in parts[-1]:
                    found.append(float(math.floor(float(parts[-1]))))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list | tuple):
            # A list's length is a fact: "3 tasks" over a returned list of three tasks.
            found.append(float(len(value)))
            for item in value:
                walk(item)

    for value in values:
        walk(value)
    return found


def labelled_facts(values: Iterable[Any]) -> list[tuple[float, str]]:
    """Every number in a set of tool results, with where it was: the path of keys that led
    to it ("run.seconds", "workflow.slowest_task.seconds"). The path is the figure's role:
    a real figure in the wrong role (the whole run's 7.81 s given as a step's) passes the
    figure check, and is the commonest error left (24 of 39 in the frozen v3 test)."""
    found: list[tuple[float, str]] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list | tuple):
            for item in value:
                walk(item, path)
        elif isinstance(value, str):
            # Digits inside a name ("tcp://127.0.0.1:40013/6") are no figure of a list.
            found.extend((fact, path + TEXT) for fact in facts_from([value]))
        else:
            found.extend((fact, path) for fact in facts_from([value]))

    for value in values:
        walk(value, "")
    return found


#: Marks a path whose figure was read out of a string, not given as a number.
TEXT = "#text"


def _role(path: str) -> tuple[str, str]:
    """A path's scope (the whole ``run``, or a ``part`` of it) and statistic."""
    p = path.lower().removesuffix(TEXT)
    last = p.rsplit(".", 1)[-1]
    whole = p.startswith("run.") or p in WHOLE_RUN
    if "wall_clock" in last:
        stat = "wall"
    elif last.startswith("summed") or last == "sum":
        stat = "sum"
    elif "slowest" in p or last in ("max", "max_seconds"):
        stat = "max"
    elif "fastest" in p or last in ("min", "min_seconds"):
        stat = "min"
    else:
        stat = "other"
    return ("run" if whole else "part", stat)


#: Figures of the whole run in describe_document's workflow overview.
WHOLE_RUN = {"workflow.wall_clock_seconds", "workflow.summed_task_seconds"}

#: What the words around a figure say it is, and the roles it cannot then have: a figure
#: is misplaced only if every place the tools returned it is a role the words rule out.
#: Deliberately few and narrow; a figure also found in any other role is never flagged.
ROLE_CUES: tuple[tuple[str, re.Pattern[str], Callable[[set[tuple[str, str]]], bool]], ...] = (
    (
        "one step or task",
        # Narrow on purpose: "from its first task to its last task" is about the run.
        re.compile(
            r"\b(?:slowest|longest|shortest|fastest|quickest|single)\s+"
            r"(?:step|task|activity|operator)\b"
            r"|\b(?:step|task|activity|operator)\s+(?:that|which)\s+(?:took|ran|lasted)\b",
            re.I,
        ),
        lambda roles: all(scope == "run" for scope, _ in roles),
    ),
    (
        "wall-clock time",
        re.compile(r"wall[- ]?clock|elapsed|end[- ]to[- ]end|start to finish", re.I),
        lambda roles: all(stat == "sum" for _, stat in roles),
    ),
    (
        "summed time",
        re.compile(r"\bsummed\b|added up|cumulative|combined", re.I),
        lambda roles: all(stat == "wall" for _, stat in roles),
    ),
    (
        "a peak value",
        re.compile(r"\b(?:peak|highest|maximum)\b", re.I),
        lambda roles: all(stat in ("sum", "min") for _, stat in roles),
    ),
    (
        "the slowest or longest",
        re.compile(r"\b(?:slowest|longest)\b", re.I),
        lambda roles: all(stat == "min" for _, stat in roles),
    ),
    (
        "the fastest or shortest",
        re.compile(r"\b(?:fastest|shortest|quickest)\b", re.I),
        lambda roles: all(stat == "max" for _, stat in roles),
    ),
)

#: Where a clause ends: what is said about one figure stays in its clause.
CLAUSE = re.compile(r"[.!?](?:\s|$)|\n|;|\(|\)|\b(?:and|but|while|whereas)\b", re.I)

#: Where a sentence ends.
SENTENCE = re.compile(r"[.!?](?:\s|$)|\n")

#: Words that pick the largest, or the smallest, of a list ("the busiest worker").
#: "The next most used" is not the most.
MOST = re.compile(
    r"(?<!next )(?<!second )(?<!third )\b(?:busiest|most|highest|largest|biggest|greatest)\b",
    re.I,
)
LEAST = re.compile(r"(?<!next )(?<!second )\b(?:least|fewest|lowest|smallest)\b", re.I)


def _span(pattern: re.Pattern[str], text: str, start: int, end: int) -> tuple[int, int]:
    before = [m.end() for m in pattern.finditer(text, 0, start)]
    after = pattern.search(text, end)
    return (before[-1] if before else 0, after.start() if after else len(text))


def _nearest_cue(clause: str, at: int) -> tuple[str, Any] | None:
    """The role cue nearest before the figure in its clause: in "summed 35.0 s, with a wall
    clock time of 9.0 s" each figure is read by its own words, not its neighbour's. Words
    after a figure name the next one ("summing to 1.96e-06 kg, with a maximum of ...")."""
    best = None
    for said, cue, conflicts in ROLE_CUES:
        for m in cue.finditer(clause, 0, at):
            distance = at - m.end()
            if best is None or distance < best[0]:
                best = (distance, said, conflicts)
    return None if best is None else (best[1], best[2])


def _in_name(text: str, at: int) -> bool:
    """Whether the figure at ``at`` is part of a name: an address ("127.0.0.1:62017/4"), a
    path, or anything in `code`."""
    return (at > 0 and text[at - 1] in "/:@#-_") or text.count("`", 0, at) % 2 == 1


def misplaced(answer: str, labelled: list[tuple[float, str]]) -> list[dict[str, Any]]:
    """Figures the tools returned, stated in a role the tools did not give them.

    Two kinds. A figure whose nearest words say "step" when every place it was returned is
    the whole run's (or "peak" when every place is a sum). And the first figure after
    "busiest", "most" or "highest" in a sentence, when the same lookup returned a larger
    figure in the same place for another item: "the busiest worker handled 4 tasks" beside
    a worker with 5 (the commonest wrong pick in the frozen v3 test).
    """
    out: list[dict[str, Any]] = []
    prose = spelled_as_digits(LIST_MARKER.sub(r"\1", answer))
    by_path: dict[str, list[float]] = {}
    for fact, path in labelled:
        by_path.setdefault(path, []).append(fact)
    picked: set[int] = set()  # sentences whose superlative figure was checked
    for match in NUMERAL.finditer(prose):
        literal = match.group()
        value = _to_float(literal)
        if value is None or literal in {o["figure"] for o in out}:
            continue
        # A small count is prose in general, but not as the busiest worker's 4 tasks.
        small = "." not in literal and "e" not in literal.lower() and abs(value) < INTEGER_FLOOR
        paths = sorted({p for f, p in labelled if _supported(literal, value, [f])})
        if not paths:
            continue  # unsupported figures are the figure check's business
        start, end = _span(CLAUSE, prose, match.start(), match.end())
        cue = None if small else _nearest_cue(prose[start:end], match.start() - start)
        if cue is not None and cue[1]({_role(p) for p in paths}):
            out.append({"figure": literal, "said_as": cue[0], "returned_as": paths[:3]})
            continue
        s_start, _ = _span(SENTENCE, prose, match.start(), match.end())
        # Only a plain value in a list of its kind: digits of a name are no value, and
        # "slowest" and "max" fields are already picked by code.
        lists = [
            p
            for p in paths
            if _role(p)[1] == "other" and not p.endswith(TEXT) and len(by_path[p]) > 1
        ]
        if s_start in picked or _in_name(prose, match.start()):
            continue
        picked.add(s_start)  # the sentence's first value is the one its superlative picks
        if not lists:
            continue
        lead = prose[s_start : match.start()]
        for pattern, better in ((MOST, max), (LEAST, min)):
            if pattern.search(lead) and all(better(by_path[p]) != value for p in lists):
                p = lists[0]
                out.append(
                    {
                        "figure": literal,
                        "said_as": "the " + ("largest" if better is max else "smallest"),
                        "returned_as": [p],
                        "but": f"{better(by_path[p]):g} is also returned as {p}",
                    }
                )
                break
    return out


def _supported(literal: str, value: float, facts: list[float]) -> bool:
    """A figure is supported if some fact rounds to it at the precision it was written in.

    "0.0963" is supported by 0.09629829227924347; "0.097" is not. An exponent form is
    compared at the precision of its mantissa.
    """
    mantissa = literal.lower().split("e")
    scale = 10 ** int(mantissa[1]) if len(mantissa) == 2 else 1
    tolerance = 0.5 * 10 ** -_decimals(literal) * scale
    return any(abs(fact - value) <= tolerance * (1 + 1e-9) for fact in facts)


@dataclass(frozen=True)
class GroundingResult:
    verdict: str
    figures_checked: int
    ungrounded: list[str]
    evidence: dict[str, Scalar]
    misplaced: list[dict[str, Any]] = field(default_factory=list)


def check(
    answer: str,
    facts: list[float],
    tool_calls: int,
    *,
    data_lookups: int | None = None,
    repeat_similarity: float = 0.0,
    declined: bool = False,
    labelled: list[tuple[float, str]] | None = None,
) -> GroundingResult:
    """Apply the grounding rule to an answer, given the facts the tools returned.

    ``data_lookups`` counts the successful calls to query tools (default: ``tool_calls``);
    ``repeat_similarity`` is :func:`repeat_similarity` for this turn; ``declined`` is true
    when the agent called ``decline``; ``labelled`` is :func:`labelled_facts` of the
    same results, for the role check (without it, no figure is found misplaced).
    """
    checked = small_unchecked = 0
    ungrounded: list[str] = []
    prose = spelled_as_digits(LIST_MARKER.sub(r"\1", answer))
    for match in NUMERAL.finditer(prose):
        literal = match.group()
        value = _to_float(literal)
        if value is None:
            continue
        small = "." not in literal and "e" not in literal.lower() and abs(value) < INTEGER_FLOOR
        if small and (_in_name(prose, match.start()) or not COUNTED.match(prose, match.end())):
            small_unchecked += 1
            continue
        checked += 1
        if not _supported(literal, value, facts) and literal not in ungrounded:
            ungrounded.append(literal)
    evidence: dict[str, Scalar] = {
        "figures_checked": checked,
        "ungrounded_figures": len(ungrounded),
        "tool_calls": tool_calls,
        "data_lookups": tool_calls if data_lookups is None else data_lookups,
        "repeat_similarity": repeat_similarity,
        "declined": declined,
        "small_numbers_unchecked": small_unchecked,
    }
    wrong_role = misplaced(answer, labelled) if labelled and not declined else []
    evidence["misplaced_figures"] = len(wrong_role)
    verdict = GROUNDING_RULE.apply(evidence)
    return GroundingResult(verdict, checked, ungrounded, evidence, wrong_role)
