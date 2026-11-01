"""Evaluate the agent in single-file mode on the question bank for the real files in ``data/``.

This is how the yProv Explorer uses the agent: the user has one file open, and every question
is about that file. Each item's file is uploaded once over ``POST /files`` (as the widget
does), and each question goes through ``ask`` with that ``file_id``, in a fresh conversation.
The gold answers and checks come from ``eval/questions/data_v3.json`` (built from the raw
JSON by ``eval/gold_data.py``, independently of the agent's tools). Earlier runs name the
bank they used in their manifest; data_v2.json is kept unchanged beside it.

* Out-of-scope items name no file; they are asked with a file open (chosen by a hash of the
  item id), since in the Explorer there is always one.
* Items that compare several files cannot be asked in single-file mode; they are excluded
  and listed in the manifest.

Tuning happens on the dev and used splits only (used: data_v2's test split, run once and
seen since). The test split, new in data_v3, is run once, on purpose:

    python -m eval.run_data                         # dev split, first phrasing, 1 repeat
    python -m eval.run_data --phrasing all          # every phrasing
    python -m eval.run_data --split used --only input-files     # items whose id contains this
    python -m eval.run_data --split test --repeats 3 --frozen   # the confirmatory run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
from mcp import Client
from starlette.testclient import TestClient

from yprov_ask import __version__
from yprov_ask.agent import FILE_PROMPT_SHA, PROMPT_SHA, ProvenanceAgent, default_model
from yprov_ask.grounding import GROUNDING_RULE
from yprov_ask.server import Settings, build_server
from yprov_ask.store import ProvStore
from yprov_ask.widget import http_app

from .run_eval import (
    PRICE_PER_MILLION,
    ROOT,
    Check,
    _digest,
    _load_env,
    absent,
    allowed_model,
    rate,
    summarise,
)

DATA = ROOT / "data"
BANK = Path(__file__).resolve().parent / "questions" / "data_v3.json"
LOCAL = "http://127.0.0.1:8000"


#: The scorer's version, recorded with every run and re-score. v1: substring terms. v2: whole
#: words; an in-scope answer that says the fact is not recorded fails; a figure and its term
#: must be near each other (figure_and_any: one step and its figure); figures in other units
#: count after conversion. v3: "not recorded for every process" is a correct partial answer,
#: not a hedge (6 correct carbon-coverage answers failed under v2 in the frozen test run).
#: v4: adds the check type not_recorded (samples_v1); v3's checks are unchanged.
#: v5: "not recorded" fails an answer only when it comes before the answer's evidence.
SCORER = "data-scorer/v5"

#: Saying the file lacks the fact. An in-scope answer that says so is hedging, even when the
#: right word appears elsewhere in it ("the variable is not named... the step open_dataset-tas").
#: "Not recorded for every / all / each ..." says the fact is partial, so it does not count.
NOT_RECORDED = re.compile(
    r"\bnot (?:explicitly |directly |clearly )?(?:recorded|specified|listed|named|mentioned|"
    r"stated|available|provided|included|given)\b(?!\s+(?:for|by|on|in)\s+(?:every|all|each)\b)"
    r"|\bdoes(?: not|n't) (?:explicitly )?(?:record|"
    r"specify|name|mention|state|list|indicate|include)\b|\bno information\b|"
    r"\bcannot (?:determine|find|tell|identify)\b",
    re.IGNORECASE,
)

#: Saying the file does not hold a fact, for questions whose right answer is exactly that
#: (check type ``not_recorded``): NOT_RECORDED's phrases and more, because here the phrase
#: is the answer rather than a hedge ("no activity records a status", "only a start time",
#: "the values are in a separate CSV file").
UNRECORDED = re.compile(
    NOT_RECORDED.pattern
    + r"|\bno (?:status|statuses|end time|end times|durations?|record of|failure information)\b"
    r"|\bnot (?:tracked|captured|logged|present|part of|in this file|in the file)\b"
    r"|\b(?:cannot|can't|can ?not) be (?:determined|told|known|answered)\b"
    r"|\bonly (?:a |the |its |their )?start times?\b"
    r"|\bseparate (?:csv|file|files)\b|\bdoes(?: not|n't) (?:contain|hold|have|say)\b"
    r"|\bdo(?:es)?(?: not|n't) cover\b",
    re.IGNORECASE,
)

#: A figure and the term it belongs to ("getter ... 0.785 s") must be this close, in
#: characters: one sentence or one short list, not opposite ends of the answer.
NEAR = 160

#: Units an answer may use instead of the gold's base unit (bytes, seconds).
UNITS = {
    "b": 1,
    "byte": 1,
    "bytes": 1,
    "kb": 1e3,
    "kib": 2**10,
    "mb": 1e6,
    "mib": 2**20,
    "gb": 1e9,
    "gib": 2**30,
    "ms": 1e-3,
    "millisecond": 1e-3,
    "milliseconds": 1e-3,
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "h": 3600,
    "hour": 3600,
    "hours": 3600,
}
FIGURE = re.compile(
    r"(?<![\w.])(-?\d(?:[\d,]*\d)?(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![\d.]\d)\s?([A-Za-z]+)?"
)


#: Small whole numbers written as words ("two executors").
WORDS = {
    w: i
    for i, w in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ]
    )
}
NUMBER_WORD = re.compile(r"(?<![A-Za-z])(" + "|".join(WORDS) + r")(?![A-Za-z])", re.IGNORECASE)


def figures(answer: str, value: float, rel: float) -> list[int]:
    """Where the answer states ``value``: as written, or converted from another unit."""
    found = [m.start() for m in NUMBER_WORD.finditer(answer) if WORDS[m.group(1).lower()] == value]
    for m in FIGURE.finditer(answer):
        try:
            x = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        unit = (m.group(2) or "").lower()
        candidates = [x] + ([x * UNITS[unit]] if unit in UNITS else [])
        if any(abs(c - value) <= max(abs(value) * rel, 1e-12) for c in candidates):
            found.append(m.start())
    return found


def has_figure(value: float, rel: float) -> Check:
    return lambda a, v: bool(figures(a, value, rel))


def figure_near(value: float, rel: float, terms: list[str], every: bool) -> Check:
    """The figure, and the term(s), within NEAR characters of one stated figure."""
    patterns = [word(t) for t in terms]

    def check(answer: str, verdict: str) -> bool:
        for at in figures(answer, value, rel):
            window = answer[max(0, at - NEAR) : at + NEAR]
            hits = [bool(p.search(window)) for p in patterns]
            if all(hits) if every else any(hits):
                return True
        return False

    return check


def word(term: str) -> re.Pattern[str]:
    """A term as a whole word: not inside a longer run of letters and digits. Plain
    substrings were wrong: the variable "tas" is inside "dataset", "air" inside "repair".
    Hyphens, underscores and dots separate words, so open_dataset-tas names tas."""
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.IGNORECASE)


def has_all(*terms: str) -> Check:
    patterns = [word(t) for t in terms]
    return lambda a, v: all(p.search(a) for p in patterns)


def has_any(*terms: str) -> Check:
    patterns = [word(t) for t in terms]
    return lambda a, v: any(p.search(a) for p in patterns)


def check_for(spec: dict[str, Any]) -> Check:
    """The bank's check spec as a function of (answer, verdict)."""
    inner = _check_for(spec)
    if spec["type"] in ("verdict", "absent", "not_recorded"):
        return inner

    def check(answer: str, verdict: str) -> bool:
        if not inner(answer, verdict):
            return False
        hedge = NOT_RECORDED.search(answer)
        if hedge is None:
            return True
        # v5: "not recorded" before the answer's evidence hedges it ("not named... the step
        # open_dataset-tas"); after it, it is a remark about another fact ("19 metrics. Their
        # values are not included in this file", 13 correct answers failed under v4).
        at = evidence_at(spec, answer)
        return at is not None and hedge.start() > at

    return check


def evidence_at(spec: dict[str, Any], answer: str) -> int | None:
    """Where the answer first states the gold: one of its figures or terms."""
    found = figures(answer, spec["value"], spec.get("rel", 0)) if "value" in spec else []
    for term in spec.get("terms", []):
        if m := word(term).search(answer):
            found.append(m.start())
    return min(found) if found else None


def _check_for(spec: dict[str, Any]) -> Check:
    kind = spec["type"]
    if kind == "figure":
        return has_figure(spec["value"], spec["rel"])
    if kind == "all_terms":
        return has_all(*spec["terms"])
    if kind == "any_terms":
        return has_any(*spec["terms"])
    if kind == "figure_and_terms":
        # A total and the list it sums (7 failures: merge_cubes, multiply, ...): a long
        # list puts the total far from some names, so here both need only appear.
        fig, terms = has_figure(spec["value"], spec["rel"]), has_all(*spec["terms"])
        return lambda a, v: fig(a, v) and terms(a, v)
    if kind == "figure_and_any":
        return figure_near(spec["value"], spec["rel"], spec["terms"], every=False)
    if kind == "verdict":
        return lambda a, v: v == spec["value"]
    if kind == "absent":
        return absent(spec["fact"])
    if kind == "not_recorded":
        # Right: declining, or saying the file does not hold it. Banning the figure that
        # would wrongly answer it (the run's 7.81 s as a step's) also failed the right
        # answer "step durations are not recorded; the whole run took 7.81 s", so a figure
        # stated alongside is left to the person-scored audit.
        return lambda a, v: v == "OUT_OF_SCOPE" or bool(UNRECORDED.search(a))
    raise ValueError(f"unknown check type {kind!r}")


def file_for(item: dict[str, Any], names: list[str]) -> str | None:
    """The file open while the item is asked; None if it can't be asked with one file."""
    if len(item["files"]) == 1:
        return item["files"][0]
    if not item["files"]:
        return names[int(hashlib.sha256(item["id"].encode()).hexdigest(), 16) % len(names)]
    return None


def upload_all(server: Any, names: list[str], data_dir: Path = DATA) -> dict[str, dict[str, Any]]:
    """Each file over POST /files, as the widget uploads it: the PROV file alone, under its
    own file name; returns what the server said."""
    opened = {}
    http = TestClient(http_app(server, origins=[]), base_url=LOCAL)
    for name in names:
        response = http.post(
            "/files", params={"name": Path(name).name}, content=(data_dir / name).read_bytes()
        )
        if response.status_code != 200:
            raise SystemExit(f"upload of {name} failed: {response.status_code} {response.text}")
        opened[name] = response.json()
    return opened


async def ask_all(server: Any, cases: list[dict[str, Any]], repeat: int) -> list[dict[str, Any]]:
    rows = []
    async with Client(server) as client:
        for case in cases:
            result = await client.call_tool(
                "ask", {"question": case["question"], "file_id": case["file_id"]}
            )
            if result.is_error:
                raise SystemExit(f"{case['id']}: {result.content[0].text}")
            out = result.structured_content
            record = json.loads(Path(out["record"]).read_text(encoding="utf-8"))
            respond = next(a for a in record["activity"].values() if "yp:model" in a)
            g = out["grounding"]
            row = {
                "repeat": repeat,
                "id": case["id"],
                "type": case["type"],
                "category": case["category"],
                "file": case["file"],
                "phrasing": case["phrasing"],
                "turn": 1,
                "question": case["question"],
                "gold": case["gold"],
                "check": case["check"],
                "expect": case["expect"],
                "answer": out["answer"],
                "verdict": out["verdict"],
                "correct": check_for(case["check"])(out["answer"], out["verdict"]),
                "declined": out["declined"],
                "figures_checked": g["figures_checked"],
                "ungrounded": g["ungrounded_figures"],
                "data_lookups": g["data_lookups"],
                "repeat_similarity": g["repeat_similarity"],
                "reasons": g["reasons"],
                "tools": [c["tool"] for c in out["tool_calls"]],
                "model_calls": respond["yp:model_calls"],
                "prompt_tokens": respond["yp:prompt_tokens"],
                "completion_tokens": respond["yp:completion_tokens"],
                "seconds": out["seconds"],
                "record": Path(out["record"]).name,
            }
            rows.append(row)
            mark = "ok " if row["correct"] else "BAD"
            print(
                f"{mark} {row['verdict']:12} {case['id'][:60]:60} p{case['phrasing']}"
                f" {row['seconds']:5.1f}s {row['data_lookups']} lookups",
                flush=True,
            )
    return rows


def run_repeat(
    model_name: str | None,
    records: Path,
    cases: list[dict[str, Any]],
    names: list[str],
    n: int,
    data_dir: Path = DATA,
) -> list[dict[str, Any]]:
    # A fresh server and store per repeat: nothing from one repeat is visible to the next.
    store = ProvStore()
    settings = Settings()
    settings.records = records
    settings.cache = records.parent / f"uploads-r{n}"
    model = default_model(model_name)
    server = build_server(
        store, settings, agent_factory=lambda: ProvenanceAgent(store, model, records_dir=records)
    )
    opened = upload_all(server, names, data_dir)
    for case in cases:
        case["file_id"] = opened[case["file"]]["file_id"]
    return anyio.run(ask_all, server, cases, n)


def by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return {k: rate(v, lambda r: r["correct"]) for k, v in sorted(groups.items())}


def rescore(path: Path) -> None:
    """The current checks applied to an earlier run's stored answers; no model calls. The
    original file is left untouched; the result, with what changed, goes next to it."""
    report = json.loads(path.read_text(encoding="utf-8"))
    changed = []
    for row in report["rows"]:
        correct = check_for(row["check"])(row["answer"], row["verdict"])
        if correct != row["correct"]:
            changed.append(
                {
                    "id": row["id"],
                    "phrasing": row["phrasing"],
                    "now": correct,
                    "answer": row["answer"],
                }
            )
            row["correct"] = correct
    report["summary"] = summarise(report["rows"])
    for key, field in (("category", "category"), ("type", "type"), ("file", "file")):
        report[f"correct_by_{key}"] = by(report["rows"], field)
    report["rescored"] = {
        "at": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "scorer": SCORER,
        "scored_before": report["manifest"].get("scorer", "data-scorer/v1"),
        "changed": changed,
    }
    out = path.with_name("results.rescored.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    s = report["summary"]
    print(
        f"{path.parent.name}: in-scope correct {s['in_scope_correct']['k']}/"
        f"{s['in_scope_correct']['n']}, accepted but wrong {s['accepted_but_wrong']['k']}/"
        f"{s['accepted_but_wrong']['n']}; changed: {changed}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--split", choices=["dev", "used", "test"], default="dev")
    parser.add_argument("--phrasing", default="0", help="0, 1, 2 or all")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--only", default=None, help="items whose id contains this")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--frozen", action="store_true", help="required for --split test: the run is final"
    )
    parser.add_argument("--rescore", default=None, metavar="RESULTS_JSON")
    parser.add_argument(
        "--bank", default=None, help="question bank in eval/questions/ (default data_v3.json)"
    )
    args = parser.parse_args()
    if args.rescore:
        rescore(Path(args.rescore))
        return
    allowed_model(args.model)
    if args.split == "test" and not args.frozen:
        sys.exit("the test split is run once, as the confirmatory run: add --frozen")
    _load_env(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (environment or .env); the evaluation calls the model")

    bank_path = BANK.with_name(args.bank) if args.bank else BANK
    bank_text = bank_path.read_text(encoding="utf-8")
    bank = json.loads(bank_text)
    # Files are named by their path under the bank's data directory (data/ is flat).
    data_dir = ROOT / bank.get("data_dir", "data")
    names = sorted(p.relative_to(data_dir).as_posix() for p in data_dir.rglob("*.json"))
    items = [
        i
        for i in bank["items"]
        if i["split"] == args.split and (not args.only or args.only in i["id"])
    ]
    phrasings = [0, 1, 2] if args.phrasing == "all" else [int(args.phrasing)]
    cases, excluded = [], []
    for item in items:
        name = file_for(item, names)
        if name is None:
            excluded.append(item["id"])
            continue
        for p in phrasings:
            if p < len(item["phrasings"]):
                cases.append(
                    {k: item[k] for k in ("id", "type", "category", "gold", "check", "expect")}
                    | {"file": name, "phrasing": p, "question": item["phrasings"][p]}
                )
    needed = sorted({c["file"] for c in cases})
    print(f"{len(cases)} questions on {len(needed)} files; excluded (several files): {excluded}")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    default = ROOT / "eval" / "results" / f"{stamp}-{data_dir.name}-{args.split}"
    out_dir = Path(args.out) if args.out else default
    rows: list[dict[str, Any]] = []
    for n in range(1, args.repeats + 1):
        rows += run_repeat(args.model, out_dir / "records" / f"r{n}", cases, needed, n, data_dir)

    by_case: dict[tuple[str, int], set[tuple[bool, str]]] = {}
    for r in rows:
        by_case.setdefault((r["id"], r["phrasing"]), set()).add((r["correct"], r["verdict"]))
    unstable = sorted(f"{i} p{p}" for (i, p), o in by_case.items() if len(o) > 1)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", ":!eval/results", "yprov_ask", "eval"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    )
    agent_model = default_model(args.model)
    manifest = {
        "created": stamp,
        "mode": "single file (file_id from POST /files), fresh conversation per question",
        "commit": commit or None,
        "source_modified_since_commit": dirty,
        "package_version": __version__,
        "model": str(getattr(agent_model, "model_name", args.model)),
        "temperature": 0.0,
        "prompt_sha": PROMPT_SHA,
        # The prompt single-file mode uses, the one these questions are asked with.
        "file_prompt_sha": FILE_PROMPT_SHA,
        "grounding_rule": GROUNDING_RULE.name,
        "scorer": SCORER,
        "bank": bank_path.name,
        "bank_version": bank["version"],
        "bank_sha": "sha256:" + hashlib.sha256(bank_text.encode()).hexdigest(),
        "split": args.split,
        "phrasings": phrasings,
        "only": args.only,
        "repeats": args.repeats,
        "excluded_several_files": excluded,
        "data_dir": data_dir.name,
        "data_sha": _digest([data_dir / n for n in names]),
        "price_per_million_usd": PRICE_PER_MILLION,
        "python": platform.python_version(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "manifest": manifest,
        "summary": summarise(rows),
        "correct_by_category": by(rows, "category"),
        "correct_by_type": by(rows, "type"),
        "correct_by_file": by(rows, "file"),
        "unstable": unstable,
        "rows": rows,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["correct_by_category"], indent=2))
    print(f"unstable across repeats: {unstable or 'none'}")
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
