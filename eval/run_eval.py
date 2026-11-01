"""Evaluate the agent through its MCP interface, on questions whose answers are known.

Every expected answer is computed from the store with the deterministic tools at run time,
never typed in, so the gold standard cannot drift from the data it describes. Every question
goes through the server's ``ask`` tool over an in-process MCP session, the path a client
uses, conversation memory included.

Three kinds of case:

    single        one question in a fresh conversation, answerable from the records
    out-of-scope  one question the records cannot answer (general knowledge, sport, places,
                  a fact the records do not hold); the right outcome is OUT_OF_SCOPE
    dialogue      several questions in one conversation: follow-ups that need the earlier
                  turns ("and its minimum?"), topic switches, and off-topic turns in between

Each turn scores two things separately: whether the answer is correct, and the verdict the
agent's own check gave. Keeping them apart is the point: the check is one-sided, so the cell
that matters most is an ACCEPTed answer that is wrong.

    python -m eval.run_eval                       # gpt-4o-mini snapshot, temperature 0
    python -m eval.run_eval --repeats 3           # the same cases three times

Evaluations always use gpt-4o-mini, the cheapest model; ``--model`` takes only a gpt-4o-mini
snapshot (``allowed_model``).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
from mcp import Client

from yprov_ask import __version__
from yprov_ask import tools as T
from yprov_ask.agent import PROMPT_SHA, ProvenanceAgent, default_model
from yprov_ask.conversation import TURNS_FOR_MODEL
from yprov_ask.grounding import GROUNDING_RULE, NUMERAL, REPEAT_THRESHOLD
from yprov_ask.server import Settings, build_server
from yprov_ask.store import ProvStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"

#: US dollars per million tokens, for the cost estimate only (gpt-4o-mini list price,
#: 2026-10). The token counts themselves are measured and recorded.
PRICE_PER_MILLION = {"input": 0.15, "output": 0.60}

#: (answer, verdict) -> correct?
Check = Callable[[str, str], bool]


def text(check: Callable[[str], bool]) -> Check:
    return lambda answer, verdict: check(answer)


def has_all(*words: str) -> Check:
    return text(lambda a: all(w.lower() in a.lower() for w in words))


def has_any(*words: str) -> Check:
    return text(lambda a: any(w.lower() in a.lower() for w in words))


def has_figure(value: float, rel: float = 0.005) -> Check:
    def check(answer: str) -> bool:
        for m in NUMERAL.finditer(answer):
            try:
                x = float(m.group().replace(",", ""))
            except ValueError:
                continue
            if abs(x - value) <= max(abs(value) * rel, 1e-12):
                return True
        return False

    return text(check)


def both(*checks: Check) -> Check:
    return lambda a, v: all(c(a, v) for c in checks)


def out_of_scope() -> Check:
    """The agent declined: the only right outcome for a question the records can't answer."""
    return lambda answer, verdict: verdict == "OUT_OF_SCOPE"


def absent(fact: str) -> Check:
    """The records lack ``fact``: declining is right, and so is an answer that says it is not
    recorded, as long as it gives no value for that fact. Other figures are allowed: saying
    what the records hold instead is what the agent is told to do."""
    negation = re.compile(r"\b(not|no|doesn't|does not|isn't|unavailable)\b", re.I)
    # A value given for the fact: a number in the same clause as its name.
    # (?<![\w.]) keeps out digits inside an identifier: "test accuracy for example_GR0_0 is
    # not recorded" gives no value.
    claimed = re.compile(re.escape(fact) + r"[^.;,\n]*?(?<![\w.])\d", re.I)
    return lambda a, v: v == "OUT_OF_SCOPE" or (bool(negation.search(a)) and not claimed.search(a))


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    check: Check
    gold: str
    #: The verdict a correct answer should get: ACCEPT (in scope), OUT_OF_SCOPE (the records
    #: can't answer it), or ABSENT (about the records, but the fact isn't in them: declining
    #: and saying "not recorded" are both right, so only correctness is scored).
    expect: str = "ACCEPT"


@dataclass(frozen=True)
class Dialogue:
    id: str
    category: str
    turns: Sequence[Case] = field(default_factory=tuple)


def dialogues(store: ProvStore) -> list[Dialogue]:
    def metric(name: str, context: str, doc: str) -> dict[str, Any]:
        return T.metric_summary(store, name, context, doc)["metrics"][0]

    val0 = metric("MSELoss", "Validation", "prov_example_GR0_0")
    train2 = metric("CrossEntropyLoss", "Training", "prov_example_GR0_2")
    doc0 = T.describe_document(store, "prov_example_GR0_0")
    doc2 = T.describe_document(store, "prov_example_GR0_2")
    all_docs = T.list_documents(store)["total"]
    faults = T.list_documents(store, collection="faults")["total"]
    escalations = T.list_decisions(store, collection="faults", verdict="ESCALATE")["total"]
    run0 = doc0["run_attributes"]
    started = datetime.fromisoformat(run0["prov:startedAtTime"])
    ended = datetime.fromisoformat(run0["prov:endedAtTime"])
    duration = (ended - started).total_seconds()

    count_faults = Case(
        "count-faults",
        "How many documents are in the faults collection?",
        has_figure(faults, 0) if faults >= 10 else has_any(str(faults), "nine"),
        str(faults),
    )
    escalated_why = Case(
        "escalated-why",
        "which decisions were escalted and why?",
        both(
            has_any(str(escalations), "three"),
            has_all("timeout", "integrity"),
            has_any("missing", "incomplete", "never arrived"),
        ),
        f"{escalations}: ca1-timeout, ca1-integrity, ca1-incomplete, with reasons",
    )
    singles = [
        count_faults,
        Case(
            "final-val-loss",
            "What is the final validation MSELoss of the run example_GR0_0?",
            has_figure(val0["final"]),
            f"{val0['final']:.4f}",
        ),
        Case(
            "best-val-loss",
            "What was the lowest validation MSELoss of example_GR0_0, and at which epoch?",
            both(has_figure(val0["min"]), has_any(str(val0["min_at_epoch"]))),
            f"{val0['min']:.4f} at epoch {val0['min_at_epoch']}",
        ),
        Case(
            "final-train-loss-2",
            "What was the final training CrossEntropyLoss of example_GR0_2?",
            has_figure(train2["final"]),
            f"{train2['final']:.4f}",
        ),
        Case(
            "why-integrity",
            "Why did the storage agent escalate in the ca1-integrity scenario?",
            both(has_any("integrity"), has_any("false", "fail", "mismatch")),
            "integrity check failed (integrity_ok false)",
        ),
        Case(
            "why-timeout",
            "Why did the storage agent escalate in the ca1-timeout scenario?",
            has_any("timeout", "timed out", "time out"),
            "persisted on timeout, not quorum",
        ),
        Case(
            "why-incomplete",
            "Why did the storage agent escalate in the ca1-incomplete scenario?",
            has_any("missing", "incomplete"),
            "one envelope missing",
        ),
        Case(
            "rule-mc1",
            "Which rule produced the compliance checker's verdict in the "
            "mc1-unjustified scenario, and what was the verdict?",
            has_all("mc1.compliance/v1", "FAIL"),
            "mc1.compliance/v1, FAIL",
        ),
        Case(
            "why-revise",
            "Why was the summary sent back for revision in the cax-revise scenario?",
            has_any("ungrounded", "not grounded", "unsupported", "not supported", "not in the"),
            "2 ungrounded figures",
        ),
        Case(
            "count-escalate",
            "How many decisions in the fault scenarios have the verdict ESCALATE?",
            has_any(str(escalations), "three"),
            str(escalations),
        ),
        Case(
            "lineage",
            "What was the model mnist_model.pt/Training in example_GR0_0 derived from?",
            has_any("Original_mnist_model.pt"),
            "Original_mnist_model.pt/Training",
        ),
        Case(
            "environment",
            "What operating system and machine type ran example_GR0_0?",
            has_all(str(run0["yprov:operating_system"]), str(run0["yprov:machine"])),
            f"{run0['yprov:operating_system']}, {run0['yprov:machine']}",
        ),
        Case(
            "entities",
            "How many entities does the run example_GR0_0 record?",
            has_figure(doc0["counts"]["entities"], 0),
            str(doc0["counts"]["entities"]),
        ),
        Case(
            "optimizer-2",
            "Which optimizer did example_GR0_2 use?",
            has_any(str(doc2["parameters"]["optimizer"])),
            str(doc2["parameters"]["optimizer"]),
        ),
        Case(
            "absent",
            "What test accuracy did example_GR0_0 reach?",
            absent("test accuracy"),
            "not recorded",
            expect="ABSENT",
        ),
        Case(
            "derived-duration",
            "How long did the run example_GR0_0 take from start to end?",
            has_any(f"{duration:.1f}", f"{duration:.2f}", run0["prov:endedAtTime"][11:19]),
            f"{duration:.2f} s (computed from the start and end times)",
        ),
        Case(
            "documents-loaded",
            "What documents are loaded?",
            has_figure(all_docs, 0),
            f"{all_docs} documents (the agent's own answer records not counted)",
        ),
    ]

    def off(case_id: str, question: str) -> Case:
        return Case(case_id, question, out_of_scope(), "OUT_OF_SCOPE", expect="OUT_OF_SCOPE")

    off_topic = [
        off("pizza", "best pizza in trento?"),
        off("la-liga", "best teams in la liga?"),
        off("messi", "lionel messi best season in terms of goals?"),
        off("italy-pm", "who was Italy prime minister in 2022"),
        off("capital", "What is the capital of France?"),
        off("arithmetic", "What is 17 times 23?"),
        off("greeting", "Hello, how are you today?"),
        off("poem", "Write me a short poem about the sea."),
        off("gpu-price", "How much does an NVIDIA H100 GPU cost?"),
        off("weather", "Will it rain in Trento tomorrow?"),
    ]

    optimizer0 = str(doc0["parameters"]["optimizer"])
    one = {c.id: c for c in singles}
    other = {c.id: c for c in off_topic}
    dialogues_ = [Dialogue(f"single/{c.id}", "single", (c,)) for c in singles]
    dialogues_ += [Dialogue(f"off-topic/{c.id}", "out-of-scope", (c,)) for c in off_topic]
    dialogues_ += [
        # A conversation that exposed earlier faults, replayed as it was typed.
        Dialogue(
            "dialogue/reported-session",
            "dialogue",
            (
                one["documents-loaded"],
                escalated_why,
                other["pizza"],
                other["la-liga"],
                other["messi"],
                other["italy-pm"],
            ),
        ),
        Dialogue(
            "dialogue/pronouns-ml",
            "dialogue",
            (
                one["final-val-loss"],
                Case(
                    "follow-min",
                    "And what was its lowest value, at which epoch?",
                    both(has_figure(val0["min"]), has_any(str(val0["min_at_epoch"]))),
                    f"{val0['min']:.4f} at epoch {val0['min_at_epoch']}",
                ),
                Case(
                    "follow-optimizer",
                    "Which optimizer did that run use?",
                    has_any(optimizer0),
                    optimizer0,
                ),
            ),
        ),
        Dialogue(
            "dialogue/pronouns-agents",
            "dialogue",
            (
                one["why-timeout"],
                Case(
                    "follow-integrity",
                    "And in the integrity scenario?",
                    both(has_any("integrity"), has_any("false", "fail", "mismatch")),
                    "integrity check failed",
                ),
            ),
        ),
        Dialogue(
            "dialogue/off-topic-between",
            "dialogue",
            (
                one["optimizer-2"],
                other["capital"],
                Case(
                    "after-off-topic",
                    "And which optimizer did example_GR0_0 use?",
                    has_any(optimizer0),
                    optimizer0,
                ),
            ),
        ),
        Dialogue(
            "dialogue/topic-switch",
            "dialogue",
            (one["optimizer-2"], count_faults),
        ),
    ]
    return dialogues_


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """The Wilson score interval for k successes in n trials (95% by default)."""
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def rate(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    k = sum(1 for r in rows if key(r))
    return {
        "k": k,
        "n": len(rows),
        "rate": round(k / len(rows), 3) if rows else None,
        "ci95": wilson(k, len(rows)),
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    in_scope = [r for r in rows if r["expect"] == "ACCEPT"]
    out = [r for r in rows if r["expect"] == "OUT_OF_SCOPE"]
    missing = [r for r in rows if r["expect"] == "ABSENT"]
    follow = [r for r in in_scope if r["turn"] > 1]
    accepted = [r for r in rows if r["verdict"] == "ACCEPT"]
    tokens_in = sum(r["prompt_tokens"] for r in rows)
    tokens_out = sum(r["completion_tokens"] for r in rows)
    verdicts: dict[str, dict[str, int]] = {}
    for r in rows:
        cell = verdicts.setdefault(r["expect"], {})
        cell[r["verdict"]] = cell.get(r["verdict"], 0) + 1
    return {
        "turns": len(rows),
        "in_scope_correct": rate(in_scope, lambda r: r["correct"]),
        "in_scope_accepted": rate(in_scope, lambda r: r["verdict"] == "ACCEPT"),
        "in_scope_falsely_declined": rate(in_scope, lambda r: r["verdict"] == "OUT_OF_SCOPE"),
        "follow_up_correct": rate(follow, lambda r: r["correct"]),
        "out_of_scope_declined": rate(out, lambda r: r["verdict"] == "OUT_OF_SCOPE"),
        "out_of_scope_answered_and_accepted": rate(out, lambda r: r["verdict"] == "ACCEPT"),
        "absent_fact_handled": rate(missing, lambda r: r["correct"]),
        # The cell that matters most: the check let a wrong answer through.
        "accepted_but_wrong": rate(accepted, lambda r: not r["correct"]),
        # The check's false alarms: a correct in-scope answer it did not accept.
        "correct_but_not_accepted": rate(
            [r for r in in_scope if r["correct"]], lambda r: r["verdict"] != "ACCEPT"
        ),
        "repeat_flagged": sum(1 for r in rows if r["repeat_similarity"] >= REPEAT_THRESHOLD),
        "verdicts_by_expected": verdicts,
        "mean_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 2),
        "mean_lookups": round(sum(r["data_lookups"] for r in rows) / len(rows), 2),
        "prompt_tokens": tokens_in,
        "completion_tokens": tokens_out,
        "estimated_usd": round(
            (tokens_in * PRICE_PER_MILLION["input"] + tokens_out * PRICE_PER_MILLION["output"])
            / 1e6,
            4,
        ),
    }


def allowed_model(name: str | None) -> None:
    """Evaluations run on gpt-4o-mini only: many calls, and the cheapest model (a project
    rule). The cost estimate assumes its price too."""
    if name is not None and not name.startswith("gpt-4o-mini"):
        sys.exit(f"evaluations use gpt-4o-mini only, not {name!r}")


def _digest(paths: Sequence[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        h.update(path.read_bytes())
    return "sha256:" + h.hexdigest()


def _load_env(path: Path) -> None:
    """KEY=VALUE lines from .env, without overriding the environment or printing them."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


async def run_once(
    store: ProvStore, model_name: str | None, records: Path, repeat: int, chosen: list[Dialogue]
) -> list[dict[str, Any]]:
    settings = Settings()
    settings.records = records
    model = default_model(model_name)
    server = build_server(
        store,
        settings,
        agent_factory=lambda: ProvenanceAgent(store, model, records_dir=records),
    )
    rows = []
    async with Client(server) as client:
        for dialogue in chosen:
            conversation = None
            for n, case in enumerate(dialogue.turns, start=1):
                args: dict[str, Any] = {"question": case.question}
                if conversation:
                    args["conversation_id"] = conversation
                result = await client.call_tool("ask", args)
                if result.is_error:
                    raise SystemExit(f"{dialogue.id} turn {n}: {result.content[0].text}")
                out = result.structured_content
                conversation = out["conversation_id"]
                record = json.loads(Path(out["record"]).read_text(encoding="utf-8"))
                respond = next(a for a in record["activity"].values() if "yp:model" in a)
                g = out["grounding"]
                row = {
                    "repeat": repeat,
                    "dialogue": dialogue.id,
                    "category": dialogue.category,
                    "turn": n,
                    "id": case.id,
                    "question": case.question,
                    "gold": case.gold,
                    "expect": case.expect,
                    "answer": out["answer"],
                    "verdict": out["verdict"],
                    "correct": case.check(out["answer"], out["verdict"]),
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
                    f"{mark} {row['verdict']:12} {dialogue.id:34} t{n} {case.id:18}"
                    f" {row['seconds']:5.1f}s {row['data_lookups']} lookups",
                    flush=True,
                )
    return rows


def rescore(path: Path) -> None:
    """Apply the current checks and expectations to the answers an earlier run stored.

    For a fault in the evaluation itself (a check too strict, a case in the wrong category):
    the model is not called again, so the answers are exactly those of the run. The result
    goes next to the original, which is left untouched, with a note of what was re-scored.
    """
    report = json.loads(path.read_text(encoding="utf-8"))
    store = ProvStore()
    store.ingest_path(SAMPLES)
    cases = {(d.id, n): c for d in dialogues(store) for n, c in enumerate(d.turns, start=1)}
    changed = []
    for row in report["rows"]:
        case = cases[(row["dialogue"], row["turn"])]
        new = {"expect": case.expect, "correct": case.check(row["answer"], row["verdict"])}
        if any(row[k] != v for k, v in new.items()):
            changed.append(
                {"repeat": row["repeat"], "dialogue": row["dialogue"], "turn": row["turn"]}
                | {f"was_{k}": row[k] for k in new}
                | new
            )
            row.update(new)
    repeats = sorted({r["repeat"] for r in report["rows"]})
    report["summary"] = summarise(report["rows"])
    report["per_repeat"] = {
        r: summarise([x for x in report["rows"] if x["repeat"] == r]) for r in repeats
    }
    report["rescored"] = {
        "at": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "cases_sha": "sha256:" + hashlib.sha256(inspect.getsource(dialogues).encode()).hexdigest(),
        "changed": changed,
    }
    out = path.with_name("results.rescored.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"{len(changed)} row(s) re-scored; written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--only", default=None, help="run dialogues whose id contains this")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--rescore",
        default=None,
        metavar="RESULTS_JSON",
        help="re-score an earlier run's stored answers with the current checks; no model calls",
    )
    args = parser.parse_args()
    if args.rescore:
        rescore(Path(args.rescore))
        return
    allowed_model(args.model)
    _load_env(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (environment or .env); the evaluation calls the model")

    store = ProvStore()
    store.ingest_path(SAMPLES)
    chosen = [d for d in dialogues(store) if not args.only or args.only in d.id]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) if args.out else ROOT / "eval" / "results" / stamp

    rows: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        # A fresh store per repeat: one repeat's answer records must not be visible to the next.
        store = ProvStore()
        store.ingest_path(SAMPLES)
        rows += anyio.run(
            run_once, store, args.model, out_dir / "records" / f"r{repeat}", repeat, chosen
        )

    per_repeat = {
        r: summarise([x for x in rows if x["repeat"] == r]) for r in range(1, args.repeats + 1)
    }
    # Turns whose outcome differed between repeats: the model is not deterministic even at
    # temperature 0, and a rate means little without knowing how much of it moves.
    by_turn: dict[tuple[str, int], set[tuple[bool, str]]] = {}
    for r in rows:
        by_turn.setdefault((r["dialogue"], r["turn"]), set()).add((r["correct"], r["verdict"]))
    unstable = sorted(f"{d} t{t}" for (d, t), outcomes in by_turn.items() if len(outcomes) > 1)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", "yprov_ask", "eval/run_eval.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    )
    agent_model = default_model(args.model)
    manifest = {
        "created": stamp,
        "commit": commit or None,
        "source_modified_since_commit": dirty,
        "package_version": __version__,
        "model": str(getattr(agent_model, "model_name", args.model)),
        "temperature": 0.0,
        "prompt_sha": PROMPT_SHA,
        "grounding_rule": GROUNDING_RULE.name,
        "repeat_threshold": REPEAT_THRESHOLD,
        "turns_for_model": TURNS_FOR_MODEL,
        "repeats": args.repeats,
        "only": args.only,
        "cases_sha": "sha256:" + hashlib.sha256(inspect.getsource(dialogues).encode()).hexdigest(),
        "samples_sha": _digest([p for p in SAMPLES.rglob("*") if p.is_file()]),
        "price_per_million_usd": PRICE_PER_MILLION,
        "python": platform.python_version(),
        "samples": "samples/ (copied from yprov4agents-demonstrator at commit 0bfcfda)",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "manifest": manifest,
        "summary": summarise(rows),
        "per_repeat": per_repeat,
        "unstable_turns": unstable,
        "rows": rows,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"unstable across repeats: {unstable or 'none'}")
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
