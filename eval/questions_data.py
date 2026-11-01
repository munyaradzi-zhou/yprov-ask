"""The question bank for the yProv files in ``data/``: what users are likely to ask.

Each item is a question type applied to one file (or several), with three phrasings, a gold
answer computed from the raw JSON by ``eval.gold_data`` (independent of the agent's tools),
a check that scores an answer against it, and the capability the agent needs to answer it.
Splits (``item["split"]``):
    dev    tuning allowed (a hash of the id puts 30% of the data_v2 items here)
    used   the rest of data_v2: its test split, run once as the frozen test
           (eval/results/20261024T143207Z-data-test). Seen since, so tuning is allowed.
    test   new in data_v3: question types never asked before, frozen by the hash of the
           file this module writes, run once.

    python -m eval.questions_data            # writes eval/questions/data_v3.json, prints a summary

Earlier banks stay in eval/questions/ unchanged, beside the runs that used them.

Check types (``check["type"]``):
    figure       a number within ``rel`` (relative) of ``value``
    all_terms    every term appears (case-insensitive)
    any_terms    at least one term appears
    figure_and_terms  a number and every term
    figure_and_any    a number and at least one term (ties: any tied step is right)
    verdict      the agent's verdict equals ``value`` (OUT_OF_SCOPE)
    absent       declined, or says the fact is not recorded without giving a value for it
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import gold_data as G

OUT = Path(__file__).resolve().parent / "questions" / "data_v3.json"

#: data_v2 replaces data_v1 before any run on its test split: the gold answers
#: of "slowest task", "most frequent step" and "most time-consuming step" accept every tied
#: step. Checking the aggregation tool against this gold showed that in 6 of 18 files
#: several tasks share the longest duration (dask fuses tasks with identical times), and
#: Ophidia has ties in step counts and summed time; v1 accepted only the one met first.
#: data_v3 (after the frozen run of v2's test split): the v2 items unchanged but
#: for two more "no failure" phrases (a correct "did not have any errors" failed); v2's test
#: split becomes "used"; new question types form the new test split.
VERSION = "data_v3"

#: Share of items in the development set; the rest is the frozen test set.
DEV_SHARE = 0.3

#: Phrases that state no step failed. Deliberately not a bare "all" or "successful", which a
#: wrong answer ("all tasks failed", "not successful") also contains.
NO_FAILURE = [
    "no task failed",
    "no tasks failed",
    "none failed",
    "none of the tasks failed",
    "no failed",
    "no failures",
    "did not fail",
    "didn't fail",
    "no errors",
    "all tasks succeeded",
    "all tasks completed",
    "all succeeded",
    "completed successfully",
    "status success",
    'status "success"',
    "status 'success'",
    # data_v3
    "not have any errors",
    "not have any failed",
    "0 failed",
    "zero failed",
]

WFS = [
    "climatology_anomaly_jt.json",
    "climatology_anomaly_jt2.json",
    "climatology_anomaly_jt2_semantic_enriched_full.json",
    "dataset_mean.json",
    "dataset_mean_jt.json",
    "machine_learning.json",
    "machine_learning_jt.json",
    "prophet_jt.json",
    "spearman_correlation.json",
    "spearman_correlation_jt.json",
    "test_netcdf.json",
    "yprov4wfs.json",
]
OPENEO = ["yprov4openEO.json", "openEO_flood_mapper.json"]
OPHIDIA = ["Fires_0.json"]
ANALYTICS = ["climatology_detail.json", "climatology_detail_red.json"]


def _doc(name: str) -> str:
    return name.removesuffix(".json")


def _split(item_id: str) -> str:
    h = int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if h < DEV_SHARE else "test"


def _item(
    qtype: str,
    files: list[str],
    category: str,
    phrasings: list[str],
    gold: str,
    check: dict[str, Any],
    needs: list[str],
    expect: str = "ACCEPT",
    split: str | None = None,
) -> dict[str, Any]:
    item_id = f"{qtype}/{'+'.join(_doc(f) for f in files)}"
    return {
        "id": item_id,
        # data_v2 items keep the split their hash gave them; its test split is now "used".
        "split": split or ("dev" if _split(item_id) == "dev" else "used"),
        "type": qtype,
        "category": category,
        "files": files,
        "producer": sorted({G.producer(G.load(f)) for f in files}) if files else [],
        "phrasings": phrasings,
        "gold": gold,
        "check": check,
        "expect": expect,
        "needs": needs,
    }


def _fig(value: float, rel: float = 0.01) -> dict[str, Any]:
    return {"type": "figure", "value": value, "rel": rel}


def wfs_items(name: str) -> list[dict[str, Any]]:
    doc = G.load(name)
    d = _doc(name)
    out = []
    n = G.counts(doc)["activity"]
    span = G.span(doc)
    total = round(sum(G.time_by_step(doc).values()), 3)
    slow = G.slowest(doc)
    top_time = next(iter(G.time_by_step(doc).items()))
    top_count = next(iter(G.steps(doc).items()))
    hosts = G.hosts(doc)
    mods = G.modules(doc)
    big = G.largest_entity(doc)
    cells = G.cells(doc)
    data = G.datasets_opened(doc)

    out.append(
        _item(
            "tasks-count",
            [name],
            "count",
            [
                f"How many tasks were executed in the workflow {d}?",
                f"How many activities does {d} record?",
                f"what's the number of tasks in {d}",
            ],
            str(n),
            _fig(n, 0),
            ["describe"],
        )
    )
    out.append(
        _item(
            "wall-clock",
            [name],
            "duration",
            [
                f"How long did the workflow {d} take from its first task to its last?",
                f"What was the wall-clock duration of {d}?",
                f"how long did {d} run end to end",
            ],
            f"{span['seconds']} s ({span['start']} to {span['end']})",
            _fig(span["seconds"], 0.02),
            ["aggregate-time"],
        )
    )
    out.append(
        _item(
            "summed-time",
            [name],
            "duration",
            [
                f"What is the total time of all tasks in {d} added together?",
                f"Summed over every task, how much compute time did {d} use?",
                f"total task time in {d}, summed across tasks",
            ],
            f"{total} s (summed over {n} tasks; wall clock {span['seconds']} s)",
            _fig(total, 0.02),
            ["aggregate-time"],
        )
    )
    out.append(
        _item(
            "slowest-task",
            [name],
            "extremum",
            [
                f"Which task in {d} took the longest, and how long did it take?",
                f"What was the slowest single task of {d}?",
                f"longest running task in {d}?",
            ],
            f"{' / '.join(slow['tied_steps'])}, {slow['seconds']} s",
            {
                "type": "figure_and_any",
                "value": slow["seconds"],
                "rel": 0.02,
                "terms": slow["tied_steps"],
            },
            ["aggregate-time"],
        )
    )
    out.append(
        _item(
            "costliest-step",
            [name],
            "extremum",
            [
                f"Which kind of task used the most time in total in {d}?",
                f"In {d}, which step accounts for the most summed task time?",
                f"where did {d} spend most of its time",
            ],
            f"{' / '.join(G.top_ties(G.time_by_step(doc)))}, {top_time[1]} s summed",
            {"type": "any_terms", "terms": G.top_ties(G.time_by_step(doc))},
            ["aggregate-time", "group-steps"],
        )
    )
    out.append(
        _item(
            "commonest-step",
            [name],
            "count",
            [
                f"Which operation was executed most often in {d}, and how many times?",
                f"What is the most frequent task type in {d}?",
                f"most common step in {d}",
            ],
            f"{' / '.join(G.top_ties(G.steps(doc)))} ({top_count[1]} tasks)",
            {
                "type": "figure_and_any",
                "value": top_count[1],
                "rel": 0,
                "terms": G.top_ties(G.steps(doc)),
            },
            ["group-steps"],
        )
    )
    out.append(
        _item(
            "failures",
            [name],
            "status",
            [
                f"Did any task fail in {d}?",
                f"Were there errors or failed tasks in the workflow {d}?",
                f"did {d} complete successfully",
            ],
            "no; every task has status success" if not G.failed(doc) else f"{G.failed(doc)}",
            # Every file of this kind has no failure; a "yes" answer names a failure, so the
            # check looks for an explicit "no failure" or "all succeeded" statement.
            {"type": "any_terms", "terms": NO_FAILURE},
            ["aggregate-status"],
        )
    )
    out.append(
        _item(
            "workers",
            [name],
            "count",
            [
                f"On how many workers or processes did the tasks of {d} run?",
                f"How many distinct executors processed the tasks of {d}?",
                f"number of workers used by {d}",
            ],
            f"{len(hosts)} ({', '.join(hosts)})",
            _fig(len(hosts), 0),
            ["aggregate-attribute"],
        )
    )
    out.append(
        _item(
            "top-module",
            [name],
            "attribute",
            [
                f"Which Python module did most of the tasks in {d} come from?",
                f"What library produced most tasks of {d}?",
                f"main module behind {d}'s tasks",
            ],
            f"{next(iter(mods))} ({next(iter(mods.values()))} tasks)",
            {"type": "all_terms", "terms": [next(iter(mods))]},
            ["aggregate-attribute"],
        )
    )
    out.append(
        _item(
            "run-date",
            [name],
            "time",
            [
                f"When was {d} run?",
                f"On which date did the workflow {d} execute?",
                f"date of the {d} run",
            ],
            span["start"][:10],
            {"type": "any_terms", "terms": _date_forms(span["start"][:10])},
            ["aggregate-time"],
        )
    )
    if data:
        out.append(
            _item(
                "dataset-variable",
                [name],
                "lineage",
                [
                    f"Which dataset variable did {d} read?",
                    f"What data did the workflow {d} open?",
                    f"which variable was loaded in {d}",
                ],
                ", ".join(data),
                {"type": "all_terms", "terms": data},
                ["search"],
            )
        )
    if big:
        out.append(
            _item(
                "largest-data",
                [name],
                "extremum",
                [
                    f"What was the largest piece of data in {d}, in bytes?",
                    f"Which intermediate result of {d} was the biggest?",
                    f"biggest data object in {d}?",
                ],
                f"{big['entity']}, {big['nbytes']} bytes",
                _fig(big["nbytes"], 0.01),
                ["aggregate-attribute"],
            )
        )
    if cells:
        idx, cell = next(iter(cells.items()))
        out.append(
            _item(
                "busiest-cell",
                [name],
                "notebook",
                [
                    f"Which notebook cell in {d} triggered the most tasks?",
                    f"In {d}, which Jupyter cell launched most of the computation?",
                    f"which cell ran the most tasks in {d}",
                ],
                f"cell {idx} ({str(cell['line'])[:40]!r}), {cell['tasks']} tasks",
                {"type": "figure_and_terms", "value": cell["tasks"], "rel": 0, "terms": [idx]},
                ["aggregate-attribute"],
            )
        )
    out.append(
        _item(
            "absent-version",
            [name],
            "absent",
            [
                f"Which version of dask was used in {d}?",
                f"What dask version ran {d}?",
                f"dask version for {d}",
            ],
            "not recorded",
            {"type": "absent", "fact": "version"},
            ["decline"],
            expect="ABSENT",
        )
    )
    if "semantic" in name:
        sem = G.semantic(doc)
        out.append(
            _item(
                "semantic-domain",
                [name],
                "semantic",
                [
                    f"What scientific domain does {d} belong to?",
                    f"According to its semantic annotations, what is {d} about?",
                    f"domain and subject of {d}",
                ],
                f"{sem['domain'][0]}; {sem['subdomain'][0]}",
                {"type": "all_terms", "terms": ["climate", "temperature"]},
                ["search"],
            )
        )
        out.append(
            _item(
                "semantic-source",
                [name],
                "semantic",
                [
                    f"Where does the data processed in {d} come from?",
                    f"What is the data source of {d}?",
                    f"data source of {d}",
                ],
                sem["data_source"][0],
                {"type": "all_terms", "terms": ["CMIP6"]},
                ["search"],
            )
        )
    return out


def _date_forms(iso: str) -> list[str]:
    from datetime import date

    day = date.fromisoformat(iso)
    return [
        iso,
        day.strftime("%B %d, %Y").replace(" 0", " "),
        day.strftime("%d %B %Y").lstrip("0"),
        day.strftime("%B %d").replace(" 0", " "),
    ]


def openeo_items(name: str) -> list[dict[str, Any]]:
    doc = G.load(name)
    d = _doc(name)
    res = G.resources(doc)
    fails = G.failed(doc)
    steps = G.steps(doc)
    span = G.span(doc)
    carbon = res["carbon_footprint_kg"]
    ram = res["ram_usage_mb"]
    out = [
        _item(
            "process-count",
            [name],
            "count",
            [
                f"How many openEO processes ran in {d}?",
                f"How many process nodes does the openEO graph {d} contain?",
                f"number of processes in {d}",
            ],
            str(G.counts(doc)["activity"]),
            _fig(G.counts(doc)["activity"], 0),
            ["describe"],
        ),
        _item(
            "failed-processes",
            [name],
            "status",
            [
                f"Which processes failed in {d}?",
                f"Did anything fail in the openEO workflow {d}? Which steps?",
                f"failed steps in {d}",
            ],
            f"{len(fails)} failed: " + ", ".join(sorted(set(fails))),
            {
                "type": "figure_and_terms",
                "value": len(fails),
                "rel": 0,
                "terms": sorted(set(fails)),
            },
            ["aggregate-status"],
        ),
        _item(
            "carbon-total",
            [name],
            "resources",
            [
                f"What was the total carbon footprint of {d}?",
                f"How much CO2 did the workflow {d} emit in total?",
                f"total carbon of {d}",
            ],
            f"{carbon['sum']:.4g} kg, from {carbon['n']} of {G.counts(doc)['activity']} processes"
            " that report it",
            _fig(carbon["sum"], 0.02),
            ["aggregate-numeric"],
        ),
        _item(
            "carbon-coverage",
            [name],
            "resources",
            [
                f"How many processes in {d} report a carbon footprint?",
                f"Is the carbon footprint recorded for every process of {d}?",
                f"carbon reported for all steps in {d}?",
            ],
            f"{carbon['n']} of {G.counts(doc)['activity']}",
            _fig(carbon["n"], 0),
            ["aggregate-numeric"],
        ),
        _item(
            "ram-peak",
            [name],
            "resources",
            [
                f"Which process in {d} used the most memory, and how much?",
                f"What was the peak RAM usage of a single process in {d}?",
                f"max ram in {d}",
            ],
            f"{ram['max_step']} ({ram['max_node']}), {ram['max']} MB",
            {
                "type": "figure_and_terms",
                "value": ram["max"],
                "rel": 0.01,
                "terms": [ram["max_step"]],
            },
            ["aggregate-numeric"],
        ),
        _item(
            "exec-vs-wall",
            [name],
            "duration",
            [
                f"How much execution time did the processes of {d} take in total, and how long was"
                " the whole run?",
                f"Compare the summed execution time of {d}'s processes with its overall duration.",
                f"{d}: total process execution time vs wall clock",
            ],
            f"{res['execution_time_sec']['sum']:.3f} s of execution;"
            f" {span['seconds']} s wall clock",
            {
                "type": "figure_and_terms",
                "value": res["execution_time_sec"]["sum"],
                "rel": 0.02,
                "terms": [str(round(span["seconds"]))],
            },
            ["aggregate-numeric", "aggregate-time"],
        ),
        _item(
            "commonest-process",
            [name],
            "count",
            [
                f"Which openEO process was used most often in {d}?",
                f"What is the most frequent process in {d}, and how many times does it occur?",
                f"most used process in {d}",
            ],
            f"{' / '.join(G.top_ties(steps))} ({next(iter(steps.values()))})",
            {
                "type": "figure_and_any",
                "value": next(iter(steps.values())),
                "rel": 0,
                "terms": G.top_ties(steps),
            },
            ["group-steps"],
        ),
        _item(
            "engine",
            [name],
            "attribute",
            [
                f"Which engine executed {d}?",
                f"What backend ran the openEO graph {d}?",
                f"engine of {d}",
            ],
            ", ".join(G.engine(doc)),
            {"type": "all_terms", "terms": ["LocalProcessing"]},
            ["search"],
        ),
        _item(
            "input-files",
            [name],
            "lineage",
            [
                f"Which input files did {d} load?",
                f"What satellite data files went into {d}?",
                f"inputs of {d}",
            ],
            ", ".join(G.input_files(doc)),
            {"type": "all_terms", "terms": ["SIG0", "PLIA"]},
            ["search", "lineage"],
        ),
    ]
    return out


def ophidia_items(name: str) -> list[dict[str, Any]]:
    doc = G.load(name)
    d = _doc(name)
    o = G.ophidia_runs(doc)
    span = G.span(doc)
    slow = G.slowest(doc)
    return [
        _item(
            "ophidia-scenario",
            [name],
            "attribute",
            [
                f"Which climate model and scenario did {d} process?",
                f"What CMIP6 model and experiment are the inputs of {d} from?",
                f"model and scenario in {d}",
            ],
            f"{o['models'][0]}, {o['scenarios'][0]}",
            {"type": "all_terms", "terms": [o["models"][0], o["scenarios"][0]]},
            ["search"],
        ),
        _item(
            "ophidia-variables",
            [name],
            "lineage",
            [
                f"Which variables did {d} import?",
                f"What climate variables were read by {d}?",
                f"variables used in {d}",
            ],
            ", ".join(o["variables"]),
            {"type": "all_terms", "terms": o["variables"]},
            ["search", "lineage"],
        ),
        _item(
            "ophidia-operators",
            [name],
            "count",
            [
                f"Which Ophidia operators were run in {d}, and how often?",
                f"List the operators executed by {d}.",
                f"operators in {d}",
            ],
            ", ".join(f"{k} {v}" for k, v in o["operators"].items()),
            {
                "type": "all_terms",
                "terms": [
                    "Import_variable",
                    "Regrid_variable",
                    "Ensemble_operation",
                    "Merge_models",
                ],
            },
            ["group-steps"],
        ),
        _item(
            "ophidia-duration",
            [name],
            "duration",
            [
                f"How long did {d} take from start to end?",
                f"What was the duration of the Ophidia run {d}?",
                f"runtime of {d}",
            ],
            f"{span['seconds']} s",
            _fig(span["seconds"], 0.02),
            ["aggregate-time"],
        ),
        _item(
            "ophidia-slowest",
            [name],
            "extremum",
            [
                f"Which operator in {d} took the longest?",
                f"What was the slowest step of {d}?",
                f"slowest operator in {d}",
            ],
            f"{' / '.join(slow['tied_steps'])} ({slow['seconds']} s)",
            {
                "type": "figure_and_any",
                "value": slow["seconds"],
                "rel": 0.02,
                "terms": slow["tied_steps"],
            },
            ["aggregate-time"],
        ),
        _item(
            "ophidia-status",
            [name],
            "status",
            [
                f"Did every operator of {d} complete?",
                f"Were there any failed operators in {d}?",
                f"all operators ok in {d}?",
            ],
            "yes, all 43 completed (OPH_STATUS_COMPLETED)",
            {
                "type": "any_terms",
                "terms": [
                    *NO_FAILURE,
                    "all completed",
                    "all 43 completed",
                    "all operators completed",
                ],
            },
            ["aggregate-status"],
        ),
    ]


def analytics_items() -> list[dict[str, Any]]:
    red = G.load("climatology_detail_red.json")
    pipes = G.pipelines(red)
    return [
        _item(
            "pipelines-count",
            ["climatology_detail_red.json"],
            "analytics",
            [
                "How many reduced pipelines does climatology_detail_red contain?",
                "In the reduced graph climatology_detail_red, how many pipelines were formed?",
                "pipelines in climatology_detail_red",
            ],
            f"{len(pipes)}: " + ", ".join(pipes),
            _fig(len(pipes), 0),
            ["search"],
        ),
        _item(
            "pipeline-size",
            ["climatology_detail_red.json"],
            "analytics",
            [
                "How many tasks does pipeline_1_1 aggregate in climatology_detail_red?",
                "Which tasks were merged into pipeline_1_1 of climatology_detail_red?",
                "size of pipeline_1_1",
            ],
            f"{len(pipes['pipeline_1_1'])}",
            _fig(len(pipes["pipeline_1_1"]), 0),
            ["node"],
        ),
        _item(
            "detail-vs-reduced",
            ANALYTICS,
            "analytics",
            [
                "How many activities are in climatology_detail and in climatology_detail_red?",
                "Compare the size of climatology_detail with its reduced version.",
                "climatology_detail vs climatology_detail_red: activity counts",
            ],
            "16 and 5",
            {"type": "all_terms", "terms": ["16", "5"]},
            ["describe", "compare"],
        ),
    ]


def cross_items() -> list[dict[str, Any]]:
    a, b = (
        G.span(G.load("climatology_anomaly_jt.json")),
        G.span(G.load("climatology_anomaly_jt2.json")),
    )
    m, mj = G.span(G.load("dataset_mean.json")), G.span(G.load("dataset_mean_jt.json"))
    air = sorted(_doc(f) for f in WFS if "air" in G.datasets_opened(G.load(f)))
    sizes = {_doc(f): G.counts(G.load(f))["activity"] for f in WFS + OPENEO + OPHIDIA}
    biggest = max(sizes, key=sizes.get)
    return [
        _item(
            "faster-climatology",
            ["climatology_anomaly_jt.json", "climatology_anomaly_jt2.json"],
            "compare",
            [
                "Which run was faster, climatology_anomaly_jt or climatology_anomaly_jt2?",
                "Compare the durations of climatology_anomaly_jt and climatology_anomaly_jt2.",
                "jt vs jt2 climatology runtime",
            ],
            f"climatology_anomaly_jt ({a['seconds']} s vs {b['seconds']} s)",
            {
                "type": "figure_and_terms",
                "value": a["seconds"],
                "rel": 0.02,
                "terms": ["climatology_anomaly_jt"],
            },
            ["aggregate-time", "compare"],
        ),
        _item(
            "mean-vs-mean-jt",
            ["dataset_mean.json", "dataset_mean_jt.json"],
            "compare",
            [
                "How do the run times of dataset_mean and dataset_mean_jt compare?",
                "Which took longer, dataset_mean or dataset_mean_jt, and by how much?",
                "dataset_mean vs dataset_mean_jt duration",
            ],
            f"dataset_mean {m['seconds']} s, dataset_mean_jt {mj['seconds']} s",
            {
                "type": "figure_and_terms",
                "value": mj["seconds"],
                "rel": 0.02,
                "terms": ["dataset_mean_jt"],
            },
            ["aggregate-time", "compare"],
        ),
        _item(
            "same-openeo",
            ["yprov4openEO.json", "yprov4wfs_openeo_complex.json"],
            "compare",
            [
                "Are yprov4openEO and yprov4wfs_openeo_complex the same record?",
                "Do yprov4openEO and yprov4wfs_openeo_complex describe the same workflow run?",
                "is yprov4wfs_openeo_complex a copy of yprov4openEO",
            ],
            "yes: identical content",
            {"type": "any_terms", "terms": ["identical", "same"]},
            ["compare"],
        ),
        _item(
            "air-users",
            WFS,
            "cross",
            [
                "Which workflows read the 'air' dataset?",
                "List the records whose workflow opened the air variable.",
                "which runs used air",
            ],
            ", ".join(air),
            {"type": "all_terms", "terms": air},
            ["search", "cross-document"],
        ),
        _item(
            "largest-workflow",
            WFS + OPENEO + OPHIDIA,
            "cross",
            [
                "Which workflow record has the most tasks?",
                "What is the largest workflow among the loaded records, by number of tasks?",
                "biggest run by tasks",
            ],
            f"{biggest} ({sizes[biggest]})",
            {"type": "figure_and_terms", "value": sizes[biggest], "rel": 0, "terms": [biggest]},
            ["describe", "cross-document"],
        ),
        _item(
            "which-failed",
            WFS + OPENEO + OPHIDIA,
            "cross",
            [
                "Which workflows had failed steps?",
                "Did any of the loaded workflows fail somewhere? Which?",
                "runs with failures",
            ],
            "the openEO records (yprov4openEO, openEO_flood_mapper, and the copy"
            " yprov4wfs_openeo_complex)",
            {"type": "any_terms", "terms": ["openEO", "openeo"]},
            ["aggregate-status", "cross-document"],
        ),
    ]


def out_of_scope_items() -> list[dict[str, Any]]:
    def off(qid: str, phrasings: list[str]) -> dict[str, Any]:
        return _item(
            qid,
            [],
            "out-of-scope",
            phrasings,
            "OUT_OF_SCOPE",
            {"type": "verdict", "value": "OUT_OF_SCOPE"},
            ["decline"],
            expect="OUT_OF_SCOPE",
        )

    return [
        off(
            "what-is-dask",
            ["What is dask?", "Explain how dask schedules tasks.", "what does dask do"],
        ),
        off(
            "install-yprov",
            [
                "How do I install yProv4WfS?",
                "Give me the pip command for yProv4WfS.",
                "install yprov4wfs",
            ],
        ),
        off(
            "climate-trend",
            [
                "Is the climate in Italy getting warmer?",
                "What will the temperature anomaly be in 2050?",
                "global warming trend",
            ],
        ),
        off(
            "openeo-docs",
            [
                "What does the openEO process merge_cubes do in general?",
                "Explain the openEO API.",
                "what is openEO",
            ],
        ),
    ]


def build() -> list[dict[str, Any]]:
    items = []
    for f in WFS:
        items += wfs_items(f)
    for f in OPENEO:
        items += openeo_items(f)
    for f in OPHIDIA:
        items += ophidia_items(f)
    items += analytics_items() + cross_items() + out_of_scope_items()
    from .questions_v3 import new_items

    items += new_items()
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    return items


def main() -> None:
    from collections import Counter

    items = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"version": VERSION, "items": items}, indent=1, default=str)
    OUT.write_text(body, encoding="utf-8")
    sha = hashlib.sha256(body.encode()).hexdigest()
    phr = sum(len(i["phrasings"]) for i in items)
    print(f"{len(items)} items, {phr} phrasings -> {OUT.name} sha256:{sha[:16]}")
    print("split:", dict(Counter(i["split"] for i in items)))
    print("category:", dict(Counter(i["category"] for i in items).most_common()))
    print("producer:", dict(Counter(p for i in items for p in i["producer"] or ["none"])))
    print("needs:", dict(Counter(n for i in items for n in i["needs"]).most_common()))


if __name__ == "__main__":
    main()
