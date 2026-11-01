"""The new test split of data_v3: question types never asked before, on the same files.

Written after the frozen run of data_v2's test split and before any run of
these. The fixes made after that run (inputs, data sizes, pipelines, task status) were found
on the v2 questions; these measure whether they, and everything else, hold on questions the
agent was never tuned on. A few types touch the same weak areas from a new angle (how many
input files, a size in megabytes, a pipeline's size); most cover facts not asked before.

Gold answers come from ``eval.gold_data`` (the raw JSON), as for v2.
"""

from __future__ import annotations

from typing import Any

from . import gold_data as G
from .questions_data import NO_FAILURE, OPENEO, OPHIDIA, WFS, _doc, _fig, _item

TEST = "test"


def _hms(moment: str) -> str:
    """'2025-09-23 15:11:53.368494' -> '15:11:53'."""
    return moment.split(" ")[1][:8]


def wfs_items(name: str) -> list[dict[str, Any]]:
    doc = G.load(name)
    d = _doc(name)
    steps = G.steps(doc)
    span = G.span(doc)
    loads = G.worker_loads(doc)
    top = next(iter(loads.values()))
    busiest = [w for w, n in loads.items() if n == top]
    big = G.largest_entity(doc)
    cells = G.cells(doc)
    out = [
        _item(
            "step-kinds",
            [name],
            "count",
            [
                f"How many different kinds of task (steps) does {d} contain?",
                f"Into how many distinct steps do the tasks of {d} fall?",
                f"number of distinct steps in {d}",
            ],
            f"{len(steps)}",
            _fig(len(steps), 0),
            ["describe"],
            split=TEST,
        ),
        _item(
            "run-end",
            [name],
            "time",
            [
                f"At what time did the last task of {d} finish?",
                f"When did the workflow {d} end?",
                f"end time of {d}",
            ],
            span["end"],
            {"type": "all_terms", "terms": [_hms(span["end"])]},
            ["aggregate-time"],
            split=TEST,
        ),
        _item(
            "busiest-worker",
            [name],
            "count",
            [
                f"Which worker ran the most tasks in {d}, and how many?",
                f"In {d}, which executor processed the largest number of tasks?",
                f"busiest worker of {d}",
            ],
            f"{' / '.join(busiest)} ({top} tasks)",
            {
                "type": "figure_and_any",
                "value": top,
                "rel": 0,
                # The full address, or its port and index as an answer may shorten it.
                "terms": busiest + [w.rsplit(":", 1)[-1] for w in busiest],
            },
            ["aggregate-attribute"],
            split=TEST,
        ),
        _item(
            "absent-user",
            [name],
            "absent",
            [
                f"Which user ran {d}?",
                f"Who was the person that launched the workflow {d}?",
                f"user who ran {d}",
            ],
            "not recorded",
            {"type": "absent", "fact": "user"},
            ["decline"],
            expect="ABSENT",
            split=TEST,
        ),
    ]
    # Only objects of 1 MB or more: 22379 bytes is "0.02 MB" rounded, 10% off the gold.
    if big and big["nbytes"] >= 1e6:
        out.append(
            _item(
                "data-size-mb",
                [name],
                "extremum",
                [
                    f"How large, in megabytes, was the biggest data object in {d}?",
                    f"What is the size of the largest result {d} produced?",
                    f"size of the largest array in {d}",
                ],
                f"{big['nbytes']} bytes ({big['nbytes'] / 1e6:.2f} MB), {big['entity']}",
                _fig(big["nbytes"], 0.01),
                ["aggregate-attribute"],
                split=TEST,
            )
        )
    if len(steps) > 1:
        second, count = list(steps.items())[1]
        out.append(
            _item(
                "step-count",
                [name],
                "count",
                [
                    f"How many {second} tasks ran in {d}?",
                    f"In {d}, how many tasks belong to the step {second}?",
                    f"{second} task count in {d}",
                ],
                str(count),
                _fig(count, 0),
                ["group-steps"],
                split=TEST,
            )
        )
    if cells:
        out.append(
            _item(
                "cell-count",
                [name],
                "notebook",
                [
                    f"How many notebook cells launched tasks in {d}?",
                    f"From how many different Jupyter cells did the tasks of {d} come?",
                    f"number of cells in {d}",
                ],
                f"{len(cells)} ({', '.join(cells)})",
                _fig(len(cells), 0),
                ["aggregate-attribute"],
                split=TEST,
            )
        )
    if "semantic" in name:
        out.append(
            _item(
                "semantic-variable",
                [name],
                "semantic",
                [
                    f"Which physical variable does {d} analyse, according to its annotations?",
                    f"What quantity is studied in {d}?",
                    f"variable studied in {d}",
                ],
                G.semantic(doc)["variable"][0],
                {"type": "any_terms", "terms": ["tas", "near-surface air temperature"]},
                ["search"],
                split=TEST,
            )
        )
    return out


def openeo_items(name: str) -> list[dict[str, Any]]:
    doc = G.load(name)
    d = _doc(name)
    inputs = G.sources(doc)
    fails = G.failed(doc)
    steps = G.steps(doc)
    exe = G.measure_peak(doc, "yprov4wfs:execution_time_sec")
    span = G.span(doc)
    return [
        _item(
            "input-count",
            [name],
            "lineage",
            [
                f"How many input files does {d} start from?",
                f"Count the source data files that {d} read.",
                f"number of inputs of {d}",
            ],
            f"{len(inputs)}: "
            + ", ".join(str(G.nodes(doc, "entity")[i].get("prov:label")) for i in inputs),
            _fig(len(inputs), 0),
            ["lineage"],
            split=TEST,
        ),
        _item(
            "backscatter-file",
            [name],
            "lineage",
            [
                f"Which Sentinel-1 backscatter (SIG0) file went into {d}?",
                f"Name the SIG0 file that {d} loaded.",
                f"SIG0 input of {d}",
            ],
            next(
                str(e.get("prov:label"))
                for e in G.nodes(doc, "entity").values()
                if str(e.get("prov:label", "")).startswith("SIG0_")
            ),
            {"type": "all_terms", "terms": ["SIG0_20180228T043908"]},
            ["lineage"],
            split=TEST,
        ),
        _item(
            "failed-count",
            [name],
            "status",
            [
                f"How many processes failed in {d}?",
                f"What number of steps of {d} ended with an error?",
                f"count of failed processes in {d}",
            ],
            str(len(fails)),
            _fig(len(fails), 0),
            ["aggregate-status"],
            split=TEST,
        ),
        _item(
            "load-count",
            [name],
            "count",
            [
                f"How many load_collection processes did {d} run?",
                f"How often was data loaded with load_collection in {d}?",
                f"load_collection count in {d}",
            ],
            str(steps["load_collection"]),
            _fig(steps["load_collection"], 0),
            ["group-steps"],
            split=TEST,
        ),
        _item(
            "exec-peak",
            [name],
            "resources",
            [
                f"Which process of {d} has the highest recorded execution time, and what is it?",
                f"According to the execution-time measure, what was the most expensive process"
                f" in {d}?",
                f"max execution_time_sec in {d}",
            ],
            f"{' / '.join(exe['steps'])}, {exe['max']} s",
            {"type": "figure_and_any", "value": exe["max"], "rel": 0.02, "terms": exe["steps"]},
            ["aggregate-numeric"],
            split=TEST,
        ),
        _item(
            "run-end",
            [name],
            "time",
            [
                f"At what time did the last process of {d} finish?",
                f"When did the openEO run {d} end?",
                f"end time of {d}",
            ],
            span["end"],
            {"type": "all_terms", "terms": [_hms(span["end"])]},
            ["aggregate-time"],
            split=TEST,
        ),
        _item(
            "absent-gpu",
            [name],
            "absent",
            [
                f"How much GPU memory did {d} use?",
                f"What GPU ran the processes of {d}?",
                f"gpu usage of {d}",
            ],
            "not recorded",
            {"type": "absent", "fact": "GPU"},
            ["decline"],
            expect="ABSENT",
            split=TEST,
        ),
    ]


def ophidia_items(name: str) -> list[dict[str, Any]]:
    doc = G.load(name)
    d = _doc(name)
    o = G.ophidia_runs(doc)
    span = G.span(doc)
    period = G.ophidia_periods(doc)[0]
    start, end = period.split("_")
    return [
        _item(
            "ophidia-period",
            [name],
            "attribute",
            [
                f"Which time period of the climate data did {d} process?",
                f"What date range were the inputs of {d} subset to?",
                f"time window of the data in {d}",
            ],
            f"{start} to {end}",
            {"type": "all_terms", "terms": [start, end]},
            ["search"],
            split=TEST,
        ),
        _item(
            "ophidia-institution",
            [name],
            "attribute",
            [
                f"Which modelling institution's data did {d} use?",
                f"Whose climate model output is the input of {d}?",
                f"institution behind the data of {d}",
            ],
            ", ".join(o["institutions"]),
            {"type": "all_terms", "terms": o["institutions"]},
            ["search"],
            split=TEST,
        ),
        _item(
            "ophidia-import-count",
            [name],
            "count",
            [
                f"How many times was Import_variable run in {d}?",
                f"How many Import_variable operators does {d} record?",
                f"Import_variable count in {d}",
            ],
            str(o["operators"]["Import_variable"]),
            _fig(o["operators"]["Import_variable"], 0),
            ["group-steps"],
            split=TEST,
        ),
        _item(
            "ophidia-end",
            [name],
            "time",
            [
                f"At what time did the last operator of {d} finish?",
                f"When did the Ophidia run {d} end?",
                f"end time of {d}",
            ],
            span["end"],
            {"type": "all_terms", "terms": [_hms(span["end"])]},
            ["aggregate-time"],
            split=TEST,
        ),
        _item(
            "ophidia-no-failure",
            [name],
            "status",
            [
                f"Did any operator of {d} end with an error?",
                f"Is there a failed step anywhere in {d}?",
                f"errors in {d}?",
            ],
            "no failures; every operator has OPH_STATUS_COMPLETED",
            {
                "type": "any_terms",
                "terms": [*NO_FAILURE, "all completed", "all operators completed"],
            },
            ["aggregate-status"],
            split=TEST,
        ),
    ]


def analytics_items() -> list[dict[str, Any]]:
    red = "climatology_detail_red.json"
    pipes = G.pipelines(G.load(red))
    biggest = max(pipes, key=lambda p: len(pipes[p]))
    level0 = [
        i
        for i, a in G.nodes(G.load(red), "activity").items()
        if a.get("analytics4yprov:level") == 0
    ]
    detail = G.counts(G.load("climatology_detail.json"))["activity"]
    return [
        _item(
            "pipeline-size-0",
            [red],
            "analytics",
            [
                "How many tasks does pipeline_1_0 stand for in climatology_detail_red?",
                "Which tasks were folded into pipeline_1_0 of climatology_detail_red? How many?",
                "pipeline_1_0 size in climatology_detail_red",
            ],
            str(len(pipes["pipeline_1_0"])),
            _fig(len(pipes["pipeline_1_0"]), 0),
            ["node"],
            split=TEST,
        ),
        _item(
            "largest-pipeline",
            [red],
            "analytics",
            [
                "Which reduced pipeline in climatology_detail_red aggregates the most tasks?",
                "In climatology_detail_red, which pipeline is the largest, and how many tasks"
                " does it merge?",
                "biggest pipeline in climatology_detail_red",
            ],
            f"{biggest} ({len(pipes[biggest])} tasks)",
            {
                "type": "figure_and_terms",
                "value": len(pipes[biggest]),
                "rel": 0,
                "terms": [biggest],
            },
            ["node"],
            split=TEST,
        ),
        _item(
            "unreduced",
            [red],
            "analytics",
            [
                "How many activities of climatology_detail_red were left as single tasks,"
                " outside any pipeline?",
                "In climatology_detail_red, how many level-0 activities are there?",
                "level 0 activities in climatology_detail_red",
            ],
            f"{len(level0)}",
            _fig(len(level0), 0),
            ["describe"],
            split=TEST,
        ),
        _item(
            "detail-count",
            ["climatology_detail.json"],
            "analytics",
            [
                "How many activities does climatology_detail record?",
                "What is the number of tasks in the unreduced graph climatology_detail?",
                "activities in climatology_detail",
            ],
            str(detail),
            _fig(detail, 0),
            ["describe"],
            split=TEST,
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
            split=TEST,
        )

    return [
        off(
            "weather-today",
            ["What is the weather in Trento today?", "Will it rain tomorrow?", "weather now"],
        ),
        off(
            "write-code",
            [
                "Write a Python function that parses a PROV-JSON file.",
                "Give me a script to plot a dask task graph.",
                "code to read netcdf",
            ],
        ),
        off(
            "who-founded",
            [
                "Who founded the yProv project?",
                "Which university develops yProv?",
                "yprov authors",
            ],
        ),
        off(
            "translate",
            [
                "Translate 'workflow' into Italian.",
                "How do you say 'provenance' in German?",
                "translate provenance to french",
            ],
        ),
    ]


def new_items() -> list[dict[str, Any]]:
    items = []
    for f in WFS:
        items += wfs_items(f)
    for f in OPENEO:
        items += openeo_items(f)
    for f in OPHIDIA:
        items += ophidia_items(f)
    return items + analytics_items() + out_of_scope_items()
