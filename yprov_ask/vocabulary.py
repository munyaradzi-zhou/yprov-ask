"""The words a user says, and the words a provenance file writes for the same thing.

A user asks for a "date range"; Ophidia writes ``subset_filter=2090-01-01_2090-01-15``. A
user asks which "worker"; yProv4WfS writes ``processed_on``. Search by the user's words
alone found nothing, and the agent then said "not recorded".

This is a written-down table, not a learned similarity: every match it makes can be
explained ("range, as subset_filter"), and a test can pin what it does. Embedding search
would find more, and could say why for none of it.
"""

from __future__ import annotations

import re

#: A word the user may say -> what files write for it. Each term is matched as a
#: case-insensitive substring of a node's id, label, type or attributes.
CONCEPTS: dict[str, tuple[str, ...]] = {
    "period": ("subset_filter", "subset"),
    "range": ("subset_filter", "subset", "interval"),
    "window": ("subset_filter", "subset", "interval"),
    "interval": ("subset_filter", "subset"),
    "date": ("subset_filter", "date"),
    "worker": ("processed_on", "host", "executor"),
    "machine": ("processed_on", "host", "hostname"),
    "host": ("processed_on", "hostname"),
    "size": ("nbytes", "bytes", "size"),
    "bytes": ("nbytes",),
    "memory": ("ram", "memory"),
    "ram": ("ram_usage", "memory"),
    "cpu": ("cpu_usage", "cpu"),
    "carbon": ("carbon", "emission", "co2"),
    "emissions": ("carbon", "emission", "co2"),
    "co2": ("carbon", "emission"),
    "energy": ("energy", "power", "kwh"),
    "status": ("status", "state"),
    "fail": ("status", "error", "fail"),
    "failure": ("status", "error", "fail"),
    "error": ("status", "error", "fail"),
    "success": ("status", "success", "completed"),
    "level": ("level",),
    "variable": ("variable", "measure"),
    "dataset": ("dataset",),
    "input": ("src_path", "source", "input"),
    "source": ("src_path", "source", "input"),
    "output": ("output", "result"),
    "identifier": ("pid", "doi", "handle", "identifier"),
    "pid": ("pid", "persistent_id", "handle"),
    "operation": ("operation", "operator"),
    "operator": ("operator", "operation"),
    "command": ("command", "cmd", "args"),
    "arguments": ("arguments", "args"),
    "library": ("module", "library", "package"),
    "module": ("module",),
    "duration": ("duration", "execution_time", "seconds"),
    "learning": ("learning_rate",),
    "optimizer": ("optimizer", "optim"),
    "epoch": ("epoch", "epochs"),
    "batch": ("batch_size", "batch"),
}

#: Words that say nothing about which node is meant.
STOP = frozenset(
    """a about after all an and any are as at be been before by can did do does done each
    file for from had has have how i in into is it its many me much my of on or record
    recorded records run show tell that the their them there these this those to used was
    were what when where which who why will with
    """.split()  # noqa: SIM905
)

_WORD = re.compile(r"[a-z0-9]+")


def stem(word: str) -> str:
    """The word without a plural or past ending, enough to look it up: "failed" -> "fail"."""
    for ending in ("ures", "ure", "ed", "es", "s"):
        if len(word) > len(ending) + 2 and word.endswith(ending):
            base = word[: -len(ending)]
            if base in CONCEPTS:
                return base
    return word


def content_words(text: str) -> list[str]:
    """The words of a question that could name a node: no stop words, no repeats."""
    out: list[str] = []
    for w in _WORD.findall(text.lower()):
        if w not in STOP and len(w) > 1 and w not in out:
            out.append(w)
    return out


def alternatives(word: str) -> list[tuple[str, str | None]]:
    """The terms to look for for one word: the word itself (no explanation needed), then
    what files write for it, each with how it was reached."""
    w = word.lower()
    out: list[tuple[str, str | None]] = [(w, None)]
    for term in CONCEPTS.get(stem(w), ()):
        if term != w:
            out.append((term, f"{word}, as {term}"))
    return out
