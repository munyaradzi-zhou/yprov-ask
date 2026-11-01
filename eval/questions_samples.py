"""Single-file questions about ``samples/``: yProv4ML runs and yProv4Agents records.

The ``data/`` banks cover workflow files only. This bank covers the two producers the
Explorer will also show, asked the way the Explorer asks: the file uploaded alone, one
question per fresh conversation. Gold answers come from ``eval.gold_samples`` (the raw JSON).

Written before any fix it measures, after a session on prov_example_GR0_0
in which the agent (a) gave the whole run's duration as "the step that took longest" and
(b) answered "0 failed tasks" for a file that records no status at all. Both are facts the
file does not record; this bank asks about such facts on purpose, with the check type
``not_recorded``: right when the answer says the file does not hold the fact, or declines
(see eval/run_data.py). The two questions of that session, on that file,
are in the dev split by hand: they have been seen. The rest are split by a hash of the id.

Note on metric values: yProv4ML writes them to CSV files beside the PROV file. An upload
carries the PROV file alone, so "what was the final loss?" is not answerable from it, and
the right answer says the values are in separate files.

    python -m eval.questions_samples         # writes eval/questions/samples_v1.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import gold_samples as G
from .questions_data import DEV_SHARE

OUT = Path(__file__).resolve().parent / "questions" / "samples_v1.json"
VERSION = "samples_v1"

ML = ["yprov4ml/example_0/prov_example_GR0_0.json", "yprov4ml/example_2/prov_example_GR0_2.json"]
#: Every fault record, and one reference record of each kind (the 18 left out repeat
#: these with other numbers: the same agents on example_2, and more monitor records).
AGENTS = [
    "agents/faults/ca1-incomplete__ca1_prov.json",
    "agents/faults/ca1-integrity__ca1_prov.json",
    "agents/faults/ca1-timeout__ca1_prov.json",
    "agents/faults/ca2-accept__ca2_prov.json",
    "agents/faults/ca3-ok__ca3_prov.json",
    "agents/faults/cax-revise__cax_prov.json",
    "agents/faults/mc1-unjustified__mc1_prov.json",
    "agents/faults/mc3-untimed__mc3_prov.json",
    "agents/faults/stream-gap__cax_prov.json",
    "agents/reference/example_0__ca1_prov.json",
    "agents/reference/example_0__ca2_prov.json",
    "agents/reference/example_0__ca3_prov.json",
    "agents/reference/example_0__cax_prov.json",
    "agents/reference/example_0__ca1__mc1_prov.json",
    "agents/reference/example_0__ca1__mc3_prov.json",
]

#: Seen in the session that prompted this bank: tuning on them is allowed.
SEEN = {"ml-step-duration/prov_example_GR0_0", "ml-failures/prov_example_GR0_0"}

#: Verdicts that are the normal outcome; "why" is asked only of the others.
ROUTINE = {"ACCEPT", "PASS", "COMPLETE", "OK"}


def _name(rel: str) -> str:
    return Path(rel).stem.removesuffix("_prov")


def _split(item_id: str) -> str:
    if item_id in SEEN:
        return "dev"
    h = int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if h < DEV_SHARE else "test"


def _item(qtype, rel, category, phrasings, gold, check, needs, expect="ACCEPT"):
    item_id = f"{qtype}/{_name(rel)}" if rel else qtype
    return {
        "id": item_id,
        "split": _split(item_id),
        "type": qtype,
        "category": category,
        "files": [rel] if rel else [],
        "producer": ["yprov4ml" if rel.startswith("yprov4ml") else "yprov4agents"] if rel else [],
        "phrasings": phrasings,
        "gold": gold,
        "check": check,
        "expect": expect,
        "needs": needs,
    }


def _fig(value: float, rel: float = 0.0) -> dict[str, Any]:
    return {"type": "figure", "value": value, "rel": rel}


def _unrecorded() -> dict[str, Any]:
    return {"type": "not_recorded"}


def ml_items(rel: str) -> list[dict[str, Any]]:
    doc = G.load(rel)
    d = _name(rel)
    run = G.ml_run(doc)
    params = G.ml_parameters(doc)
    datasets = [x.split("/")[0] for x in G.ml_entities(doc, "provml:Dataset")]
    metrics = G.ml_metric_files(doc)
    losses = sorted(m for m in metrics if "Loss" in m)
    loss = next((m for m in losses if m.endswith("Validation")), losses[0])
    phase = loss.rsplit("/", 1)[-1].lower()
    hms = run["start"].split(" ")[1][:8]
    assert not G.has_status(doc)
    assert all(s["end"] is None for s in G.ml_steps(doc).values())
    out = [
        _item(
            "ml-duration",
            rel,
            "duration",
            [
                f"How long did the run {d} take?",
                f"What was the total duration of the training run {d}?",
                f"runtime of {d}",
            ],
            f"{run['seconds']} s ({run['start']} to {run['end']})",
            _fig(run["seconds"], 0.02),
            ["describe"],
        ),
        _item(
            "ml-step-duration",
            rel,
            "absent",
            [
                f"Which step of {d} took the longest to run?",
                f"How long did the Training step of {d} take?",
                f"slowest step in {d}",
            ],
            "not recorded: Training, Validation and the other steps have a start time only;"
            f" only the whole run has a duration ({run['seconds']} s)",
            _unrecorded(),
            ["describe"],
            expect="ABSENT",
        ),
        _item(
            "ml-failures",
            rel,
            "absent",
            [
                f"Did anything fail in {d}?",
                f"Were there any errors during the run {d}?",
                f"failures in {d}?",
            ],
            "not recorded: no activity of the run records a status",
            _unrecorded(),
            ["describe"],
            expect="ABSENT",
        ),
        _item(
            "ml-loss-value",
            rel,
            "absent",
            [
                f"What was the final {phase} loss of {d}?",
                f"What {phase} loss did {d} end with?",
                f"final {phase} loss {d}",
            ],
            f"not in this file: the values of {loss} are in {metrics[loss]}, a separate CSV",
            _unrecorded(),
            ["describe"],
            expect="ABSENT",
        ),
        _item(
            "ml-metric-count",
            rel,
            "count",
            [
                f"How many metrics did {d} log?",
                f"What is the number of metric series recorded for {d}?",
                f"number of metrics in {d}",
            ],
            str(len(metrics)),
            _fig(len(metrics)),
            ["describe"],
        ),
        _item(
            "ml-model-count",
            rel,
            "count",
            [
                f"How many model entities does {d} record?",
                f"How many models or checkpoints were saved in {d}?",
                f"model count {d}",
            ],
            str(len(G.ml_entities(doc, "provml:Model"))),
            _fig(len(G.ml_entities(doc, "provml:Model"))),
            ["describe"],
        ),
        _item(
            "ml-optimizer",
            rel,
            "attribute",
            [
                f"Which optimizer did {d} use?",
                f"What optimisation algorithm trained the model in {d}?",
                f"optimizer of {d}",
            ],
            params["optimizer"],
            {"type": "all_terms", "terms": [params["optimizer"]]},
            ["describe"],
        ),
        _item(
            "ml-datasets",
            rel,
            "lineage",
            [
                f"Which datasets did {d} use?",
                f"What data went into the run {d}?",
                f"datasets of {d}",
            ],
            ", ".join(datasets),
            {"type": "all_terms", "terms": datasets},
            ["describe", "lineage"],
        ),
        _item(
            "ml-python",
            rel,
            "attribute",
            [
                f"Which Python version ran {d}?",
                f"What version of Python was {d} executed with?",
                f"python version {d}",
            ],
            run["python"],
            {"type": "all_terms", "terms": [run["python"]]},
            ["describe"],
        ),
        _item(
            "ml-start",
            rel,
            "time",
            [
                f"When did the run {d} start?",
                f"At what time did {d} begin?",
                f"start time of {d}",
            ],
            run["start"],
            {"type": "all_terms", "terms": [hms]},
            ["describe"],
        ),
        _item(
            "ml-os",
            rel,
            "attribute",
            [
                f"On which operating system did {d} run?",
                f"What OS was {d} executed on?",
                f"os of {d}",
            ],
            run["os"],
            {"type": "any_terms", "terms": [run["os"], "macOS"]},
            ["describe"],
        ),
        _item(
            "ml-script",
            rel,
            "lineage",
            [
                f"Which script was executed for {d}?",
                f"What command started the run {d}?",
                f"script of {d}",
            ],
            params["execution_command"],
            {"type": "all_terms", "terms": [G.ml_script(doc)]},
            ["describe"],
        ),
    ]
    if "scheduler" in params:
        out.append(
            _item(
                "ml-scheduler",
                rel,
                "attribute",
                [
                    f"Which learning-rate scheduler did {d} use?",
                    f"Was a scheduler used in {d}, and which?",
                    f"scheduler {d}",
                ],
                params["scheduler"],
                {"type": "all_terms", "terms": [params["scheduler"]]},
                ["describe"],
            )
        )
    return out


def why_check(dec: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """What a correct "why" answer must mention, read from the decision's rule inputs."""
    inputs = dec["rule_inputs"]
    if inputs.get("integrity_ok") is False:
        return "the integrity check failed", {"type": "any_terms", "terms": ["integrity"]}
    if (inputs.get("envelopes_missing") or 0) > 0:
        terms = ["missing", "never arrived", "did not arrive", "didn't arrive", "incomplete"]
        return "an envelope of the record never arrived", {"type": "any_terms", "terms": terms}
    if inputs.get("persist_trigger") == "timeout":
        return "a timeout: no consumer acknowledged", {"type": "any_terms", "terms": ["timeout"]}
    if (inputs.get("ungrounded_figures") or 0) > 0:
        return "figures 4096 and 512 not in the graph", {
            "type": "any_terms",
            "terms": ["4096", "512", "ungrounded"],
        }
    if (inputs.get("failed_checks") or 0) > 0:
        terms = ["justification", "unjustified", "not justified", "decision-justified"]
        return "a decision without a justification", {"type": "any_terms", "terms": terms}
    if (inputs.get("trail_gaps") or 0) > 0:
        terms = ["time", "times", "timestamp", "timestamps", "untimed", "timing"]
        return "an activity with no timestamps", {"type": "any_terms", "terms": terms}
    if inputs.get("largest_gap_s"):
        return f"a timing gap of {inputs['largest_gap_s']} s", _fig(inputs["largest_gap_s"], 0.02)
    if inputs.get("chart_subject") == "graph-composition":
        terms = ["no numeric", "composition", "graph composition"]
        return "no numeric values to plot", {"type": "any_terms", "terms": terms}
    return None


def agent_items(rel: str) -> list[dict[str, Any]]:
    doc = G.load(rel)
    d = _name(rel)
    dec = G.decision(doc)
    agent = G.agent_name(doc)
    out = [
        _item(
            "ag-verdict",
            rel,
            "decision",
            [
                f"What did the agent decide in {d}?",
                f"What verdict does the record {d} contain?",
                f"verdict of {d}",
            ],
            dec["verdict"],
            {"type": "all_terms", "terms": [dec["verdict"]]},
            ["decision"],
        ),
        _item(
            "ag-agent",
            rel,
            "attribute",
            [
                f"Which agent produced the record {d}?",
                f"Who made the decision recorded in {d}?",
                f"agent in {d}",
            ],
            agent,
            {"type": "all_terms", "terms": [agent]},
            ["describe"],
        ),
        _item(
            "ag-rule",
            rel,
            "decision",
            [
                f"Which rule produced the decision in {d}?",
                f"By what rule was the verdict of {d} reached?",
                f"rule of {d}",
            ],
            dec["rule"],
            {"type": "all_terms", "terms": [dec["rule"]]},
            ["decision"],
        ),
    ]
    why = why_check(dec) if dec["verdict"] not in ROUTINE else None
    if why:
        out.append(
            _item(
                "ag-why",
                rel,
                "decision",
                [
                    f"Why did the agent decide {dec['verdict']} in {d}?",
                    f"What caused the verdict {dec['verdict']} in {d}?",
                    f"why {dec['verdict'].lower()} in {d}",
                ],
                why[0],
                why[1],
                ["decision"],
            )
        )
    store = G.stored(doc)
    if store and store["size_bytes"]:
        out.append(
            _item(
                "ag-stored",
                rel,
                "lineage",
                [
                    f"Under which PID was the record stored in {d}, and how big was it?",
                    f"Where did the storage agent put the record in {d}, and what size?",
                    f"pid and size in {d}",
                ],
                f"{store['pid']}, {store['size_bytes']} bytes",
                {
                    "type": "figure_and_terms",
                    "value": store["size_bytes"],
                    "rel": 0,
                    "terms": [store["pid"]],
                },
                ["describe"],
            )
        )
    if "faults" in rel:
        out.append(
            _item(
                "ag-cost",
                rel,
                "absent",
                [
                    f"How much did the agent's work in {d} cost?",
                    f"What was the monetary cost of the run in {d}?",
                    f"cost of {d}",
                ],
                "not recorded",
                _unrecorded(),
                ["decline"],
                expect="ABSENT",
            )
        )
    return out


def out_of_scope_items() -> list[dict[str, Any]]:
    def off(qid: str, phrasings: list[str]) -> dict[str, Any]:
        return _item(
            qid,
            "",
            "out-of-scope",
            phrasings,
            "OUT_OF_SCOPE",
            {"type": "verdict", "value": "OUT_OF_SCOPE"},
            ["decline"],
            expect="OUT_OF_SCOPE",
        )

    return [
        off(
            "world-cup",
            [
                "Who won the 2022 World Cup?",
                "Which team won the last football World Cup?",
                "world cup winner",
            ],
        ),
        off(
            "backprop",
            [
                "Explain backpropagation.",
                "How does gradient descent train a network?",
                "what is backprop",
            ],
        ),
        off(
            "tokyo-time",
            ["What time is it in Tokyo?", "What is the time zone of Japan?", "time in tokyo"],
        ),
        off(
            "laptop",
            [
                "Recommend a laptop for deep learning.",
                "Which GPU should I buy for training?",
                "best laptop for ml",
            ],
        ),
    ]


def build() -> list[dict[str, Any]]:
    items = []
    for rel in ML:
        items += ml_items(rel)
    for rel in AGENTS:
        items += agent_items(rel)
    items += out_of_scope_items()
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    return items


def main() -> None:
    from collections import Counter

    items = build()
    body = json.dumps(
        {"version": VERSION, "data_dir": "samples", "items": items}, indent=1, default=str
    )
    OUT.write_text(body, encoding="utf-8", newline="\n")
    sha = hashlib.sha256(body.encode()).hexdigest()
    print(f"{len(items)} items -> {OUT.name} sha256:{sha[:16]}")
    print("split:", dict(Counter(i["split"] for i in items)))
    print("category:", dict(Counter(i["category"] for i in items).most_common()))
    print("expect:", dict(Counter(i["expect"] for i in items)))


if __name__ == "__main__":
    main()
