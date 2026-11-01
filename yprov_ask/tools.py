"""Deterministic queries over the store. The only place an answer's facts come from.

Each function takes plain arguments and returns plain JSON-able data, so the same code
serves three callers: the LangGraph agent (as LangChain tools), the MCP server (as MCP
tools a client can call directly, with no model in the loop), and the tests.

Nothing here calls a model or guesses. A question these functions cannot answer gets an
error that says what does exist, so the agent can ask again rather than invent.
"""

from __future__ import annotations

import contextlib
import json
import re
import statistics
from collections import Counter, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import vocabulary as V
from . import workflow as W
from .store import KINDS, ProvStore

#: Longest string value returned to a caller. Command lines and justifications can run to
#: kilobytes; the agent needs to recognise them, not to read them whole.
MAX_TEXT = 400

#: How many items a list-returning tool returns before it says how many it left out.
MAX_ITEMS = 40

#: Metrics that say how a run turned out, shown with their values in a run's overview.
HEADLINE_METRIC = re.compile(r"loss|acc|f1|error|score|auc|precision|recall|perplex", re.I)


class ToolError(ValueError):
    """A query that cannot be answered as asked. The message says what can."""


def _clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_TEXT:
        return value[:MAX_TEXT] + f"… [{len(value) - MAX_TEXT} more characters]"
    if isinstance(value, dict):
        return {k: _clip(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v) for v in value]
    return value


def _capped(items: list[Any], what: str) -> dict[str, Any]:
    out: dict[str, Any] = {what: items[:MAX_ITEMS], "total": len(items)}
    if len(items) > MAX_ITEMS:
        out["note"] = f"showing {MAX_ITEMS} of {len(items)}; narrow the query to see the rest"
    return out


def resolve_doc(store: ProvStore, doc_id: str) -> str:
    """The document a caller meant: an exact id, or the one id that contains the text."""
    if store.document(doc_id) is not None:
        return doc_id
    ids = [row["doc_id"] for row in store.documents()]
    matches = [i for i in ids if doc_id.lower() in i.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ToolError(f"no document {doc_id!r}; ingested documents are: {', '.join(ids)}")
    raise ToolError(f"{doc_id!r} matches several documents: {', '.join(matches)}")


def _node_summary(row: Any) -> dict[str, Any]:
    out = {"doc_id": row["doc_id"], "node_id": row["node_id"], "kind": row["kind"]}
    if row["label"] and row["label"] != row["node_id"]:
        out["label"] = row["label"]
    if row["prov_type"]:
        out["type"] = row["prov_type"]
    return out


# ── documents ─────────────────────────────────────────────────────────────────────────


def collections(store: ProvStore) -> dict[str, str]:
    """Each document's collection: the name of the folder it was ingested from.

    Folders are how people group provenance ("faults", "reference", "run_07"), and a
    question names the group far more often than the files in it.
    """
    return {row["doc_id"]: Path(row["path"]).parent.name for row in store.documents()}


def _answers(store: ProvStore) -> set[str]:
    """The doc_ids of this agent's own answer records."""
    return {row["doc_id"] for row in store.documents() if row["kind"] == "qa"}


def _no_collection(store: ProvStore, collection: str, names: Iterable[str]) -> ToolError:
    """The error for a collection that matches no folder. A scenario name is often part of
    a document id rather than a folder ("cax-revise" in ``cax-revise__cax_prov``); say so,
    or the model concludes the records are not there (seen in the evaluation)."""
    term = collection.lower()
    docs = sorted(r["doc_id"] for r in store.documents() if term in r["doc_id"].lower())
    hint = (
        f"; but these documents' ids contain {collection!r}: {docs[:10]} (use doc_id)"
        if docs
        else ""
    )
    return ToolError(
        f"no collection matches {collection!r}; collections are {sorted(set(names))}{hint}"
    )


def list_documents(
    store: ProvStore, collection: str | None = None, include_answers: bool = False
) -> dict[str, Any]:
    """Every ingested provenance document, with what produced it, the collection (folder)
    it came from, and how big it is. ``collection`` keeps the documents of one folder.
    The agent's own answer records (kind ``qa``) are left out unless ``include_answers``
    is true; set it only for a question about earlier answers.
    """
    folder = collections(store)
    shown = [r for r in store.documents() if include_answers or r["kind"] != "qa"]
    docs = [
        {
            "doc_id": row["doc_id"],
            "title": row["title"],
            "collection": folder[row["doc_id"]],
            "kind": row["kind"],
            "entities": row["n_entities"],
            "activities": row["n_activities"],
            "agents": row["n_agents"],
            "relations": row["n_relations"],
        }
        for row in shown
        if not collection or collection.lower() in folder[row["doc_id"]].lower()
    ]
    names = Counter(folder[r["doc_id"]] for r in shown)
    if not docs and collection:
        raise _no_collection(store, collection, names)
    return _capped(docs, "documents") | {"collections": dict(names)}


def describe_document(store: ProvStore, doc_id: str) -> dict[str, Any]:
    """An overview of one document: its agents, activities, parameters, metrics, decisions.

    For a workflow run (activities with times or a status) it also gives ``workflow``: the
    number of tasks, wall-clock and summed time, the slowest task, status counts and failed
    tasks, the steps (task kinds) with their counts and time, Python modules, workers,
    notebook cells, every numeric measure (CPU, RAM, carbon, execution time) with how many
    tasks report it, its sum and maximum (and the task with the maximum), values all tasks
    share (shared_attributes: e.g. domain, data source, variable), dask task groups, and
    workflow-level activities (e.g. the engine). ``data_flow`` names the run's inputs
    (entities used and never generated: input files, datasets) and final outputs;
    ``largest_size`` the biggest data: its size in bytes, how many entities have it, and
    a few of them as examples. An activity that stands for
    several tasks gives their number in ``list_sizes``. Use activity_stats to group or filter.
    """
    doc_id = resolve_doc(store, doc_id)
    row = store.document(doc_id)
    nodes = store.nodes(doc_id)
    by_kind: dict[str, list[Any]] = {"entity": [], "activity": [], "agent": []}
    for node in nodes:
        by_kind[node["kind"]].append(node)

    activities = []
    for node in by_kind["activity"]:
        attrs = json.loads(node["attrs"])
        item = _node_summary(node)
        start = attrs.get("prov:startedAtTime") or attrs.get("prov:startTime")
        end = attrs.get("prov:endedAtTime") or attrs.get("prov:endTime")
        if start:
            item["started"] = start
        if end:
            item["ended"] = end
        if "yp:verdict" in attrs:
            item["verdict"] = attrs["yp:verdict"]
        # An activity that stands for several (analytics4yprov's reduced pipelines list the
        # tasks they aggregate): how many, without the list itself.
        sizes = {k: len(v) for k, v in attrs.items() if isinstance(v, list)}
        if sizes:
            item["list_sizes"] = sizes
        activities.append(item)

    # A parameter is an entity that carries a value and no type of its own: yProv4ML
    # writes hyperparameters, the command line and dataset transformations this way.
    parameters = {}
    metrics = []
    types: Counter[str] = Counter()
    for node in by_kind["entity"]:
        attrs = json.loads(node["attrs"])
        types[node["prov_type"] or "untyped"] += 1
        if node["prov_type"] == "provml:Metric":
            metrics.append(node["node_id"])
        elif "prov:value" in attrs and not node["prov_type"]:
            parameters[node["node_id"]] = attrs["prov:value"]

    # yProv4ML names the run's top-level activity after the run; its attributes are the
    # execution environment (Python version, OS, start and end time).
    run_attributes = None
    if row["kind"] == "yprov4ml":
        default = (json.loads(row["raw"]).get("prefix") or {}).get("default")
        for node in by_kind["activity"]:
            if node["node_id"] == default:
                run_attributes = json.loads(node["attrs"])

    out: dict[str, Any] = {
        "doc_id": doc_id,
        "title": row["title"],
        "kind": row["kind"],
        # The library that wrote the file; not the engine that ran the work.
        "recorded_with": KINDS.get(row["kind"], row["kind"]),
        # The file's name only: its path on the server is no business of the user's, and
        # the agent gave it as a record's "storage location".
        "file": Path(row["path"]).name,
        "counts": {
            "entities": row["n_entities"],
            "activities": row["n_activities"],
            "agents": row["n_agents"],
            "relations": row["n_relations"],
        },
        "agents": [_node_summary(n) for n in by_kind["agent"]],
        "activities": activities[:MAX_ITEMS],
        "entity_types": dict(types),
        "parameters": dict(list(parameters.items())[:MAX_ITEMS]),
        "metrics": metrics[:MAX_ITEMS],
        "decisions": sum(1 for a in activities if "verdict" in a),
    }
    items, run = _steps_and_run(store, doc_id)
    if run_attributes is not None:
        out["run_attributes"] = run_attributes
    if run is not None:
        out["run"] = run
    if metrics:
        out |= _metric_overview(store, doc_id, metrics)
    if out["decisions"]:
        # A yProv4Agents record is about its decision; give it here rather than a count.
        with contextlib.suppress(ToolError):
            out["decision_details"] = [
                # doc_id and collection (for an upload, the server's "uploads" folder) are
                # not facts of the record.
                {k: v for k, v in d.items() if k not in ("doc_id", "collection")}
                for d in list_decisions(store, doc_id, include_answers=True)["decisions"][:5]
            ]
    identifiers = _identifiers(nodes)
    if identifiers:
        out["identifiers"] = identifiers
    if len(by_kind["entity"]) <= SMALL_RECORD:
        # A small record (an agent's: 2 to 4 entities) is read whole: its facts sit in the
        # entities' attributes (a storage receipt's PID), which no summary above shows.
        out["entity_details"] = [
            _node_summary(n) | {"attributes": json.loads(n["attrs"])} for n in by_kind["entity"]
        ]
    timed = sum(1 for t in items if t.seconds is not None)
    # The workflow overview is for runs of many timed tasks, or tasks with a status. A
    # yProv4ML run has one timed activity (the run itself): read as tasks, it became "the
    # slowest step" and "0 failed tasks".
    if timed >= 2 or any(t.status for t in items):
        out["workflow"] = W.summary(items)
        # Thousands of task ids say less than the summary; keep a few as examples.
        out["activities"] = activities[:10]
        if len(activities) > 10:
            out["activities_note"] = f"10 of {len(activities)} tasks shown; see workflow"
    if items and timed < len(items):
        out["timing_coverage"] = (
            f"{timed} of {len(items)} {'steps' if run else 'activities'} have both a start and"
            " an end time; the durations of the others are not recorded"
            + ("; only the whole run's duration is (see run)" if run else "")
        )
    if items and not any(t.status for t in items):
        out["status_coverage"] = (
            "no activity records a status: whether anything failed is not recorded"
        )
    largest = W.largest_entities(store, doc_id)
    if largest:
        top = largest[0]["bytes"]
        # One field, counted: a list of 3 examples beside a separate count was read as
        # "three entities", then "shared by 64" in the same answer.
        out["largest_size"] = {
            "bytes": top,
            "entities_with_this_size": W.entities_of_size(store, doc_id, top),
            "examples": [e["node_id"] for e in largest if e["bytes"] == top],
            "attribute": largest[0]["attribute"],
        }
        smaller = [e for e in largest if e["bytes"] < top]
        if smaller:
            out["next_largest"] = smaller
    flow = W.data_flow(store, doc_id)
    if flow:
        out["data_flow"] = flow
    out["attributes"] = attribute_catalogue(nodes)
    return _clip(out)


#: Records with at most this many entities have every entity's attributes in the overview.
SMALL_RECORD = 10

#: Attributes that identify something outside the file: a store's PID, a DOI, a URI.
IDENTIFIER = re.compile(r"(?:^|:)(?:pid|doi|uri|url|handle|persistent_id)$", re.IGNORECASE)


#: A key of a key=value argument string (Ophidia's "imp_dim=time,...,subset_filter=...").
ARG_KEY = re.compile(r"^[A-Za-z_][\w.-]*$")


def fields(attrs: dict[str, Any]) -> dict[str, Any]:
    """A node's attributes, with each key=value argument string also split into its parts,
    keyed ``attribute.key``. A fact inside such a string (the period Fires_0 read, as
    ``subset_filter=2090-01-01_2090-01-15``) was in no summary and never found.
    """
    out = dict(attrs)
    for key, value in attrs.items():
        if not isinstance(value, str) or value.count("=") < 2:
            continue
        pairs = [p.split("=", 1) for p in value.split(",") if "=" in p]
        if len(pairs) >= 2 and all(ARG_KEY.match(k) for k, _ in pairs):
            out |= {f"{key}.{k}": v for k, v in pairs}
    return out


#: How many distinct values of an attribute the catalogue lists; beyond it, one example.
CATALOGUE_VALUES = 6

#: How many attributes the catalogue names before it says how many it left out.
CATALOGUE_SIZE = 80


def attribute_catalogue(nodes: list[Any]) -> dict[str, Any]:
    """Every attribute the document's nodes carry: how many nodes have it, and its values
    (with how many nodes have each) when there are few, else how many distinct values and
    one example. A fact no summary picks out is still named here, so "not recorded" can
    be checked against the whole file, not only against the summary."""
    seen: dict[str, tuple[set[str], Counter[str]]] = {}
    for n in nodes:
        for key, value in fields(json.loads(n["attrs"])).items():
            if key in ("prov:label", "prov:type") or isinstance(value, (list, dict)):
                continue
            kinds, values = seen.setdefault(key, (set(), Counter()))
            kinds.add(n["kind"])
            values[str(value)[:120]] += 1
    out: dict[str, Any] = {}
    for key, (kinds, values) in sorted(seen.items()):
        item: dict[str, Any] = {"on": "/".join(sorted(kinds)), "nodes": values.total()}
        if len(values) <= CATALOGUE_VALUES:
            item["values"] = dict(values.most_common())
        else:
            item["distinct_values"] = len(values)
            item["example"] = next(iter(values))
            item["all_values"] = "count_values"
        out[key] = item
    if len(out) > CATALOGUE_SIZE:
        left = len(out) - CATALOGUE_SIZE
        out = dict(list(out.items())[:CATALOGUE_SIZE])
        out["note"] = f"{left} more attributes; search_nodes finds them by name"
    return out


def count_values(
    store: ProvStore, doc_id: str, attribute: str, kind: str | None = None
) -> dict[str, Any]:
    """Every value of one attribute in a document, with how many nodes carry it, most common
    first: which dtype is most common, how many entities are numpy arrays, how many tasks
    ran on each worker. ``attribute`` is the name as describe_document's "attributes" gives
    it, with or without its prefix ("dtype" finds yprov4wfs:dtype). The catalogue lists
    values only when there are few; with more (13 dtypes in one climate workflow) an answer
    guessed the most common from a sample and was wrong."""
    if kind is not None and kind not in ("entity", "activity", "agent"):
        raise ToolError("kind must be 'entity', 'activity' or 'agent'")
    doc_id = resolve_doc(store, doc_id)
    wanted = attribute.strip().lower()
    counts: Counter[str] = Counter()
    keys: set[str] = set()
    names: set[str] = set()
    for node in store.nodes(doc_id, kind):
        for key, value in fields(json.loads(node["attrs"])).items():
            names.add(key)
            if key.lower() == wanted or key.lower().rsplit(":", 1)[-1] == wanted:
                keys.add(key)
                counts[str(value)[:120]] += 1
    if not counts:
        near = sorted(n for n in names if wanted.rsplit(":", 1)[-1] in n.lower())
        hint = ", ".join(near[:10]) or "describe_document lists every attribute"
        raise ToolError(f"no attribute {attribute!r} in {doc_id}; similar: {hint}")
    values = [{"value": v, "nodes": c} for v, c in counts.most_common()]
    out = _capped(values, "values")
    out |= {"attribute": sorted(keys), "nodes_with_it": counts.total(), "distinct": len(counts)}
    return out


def _identifiers(nodes: list[Any]) -> list[dict[str, Any]]:
    """Persistent identifiers recorded anywhere in the document (a storage receipt's PID was
    invented in 36 of 36 answers when only the entity held it)."""
    out = []
    for n in nodes:
        for key, value in json.loads(n["attrs"]).items():
            if IDENTIFIER.search(key) and value not in (None, "", "None"):
                out.append({"node_id": n["node_id"], "attribute": key, "value": value})
    return out[:10]


def _steps_and_run(store: ProvStore, doc_id: str) -> tuple[list[W.Task], dict[str, Any] | None]:
    """The document's tasks, and the run they belong to when the file records one.

    yProv4ML names the activity that encloses a whole run after the run (the document's
    default namespace). It is the run, not one of its steps: counted as a task it was
    reported as "the step that took longest"."""
    items = W.tasks(store, doc_id)
    row = store.document(doc_id)
    if row["kind"] != "yprov4ml":
        return items, None
    default = (json.loads(row["raw"]).get("prefix") or {}).get("default")
    run = next((t for t in items if t.node_id == default), None)
    if run is None:
        return items, None
    return [t for t in items if t is not run], {
        "name": run.node_id,
        "started": run.start.isoformat(" ") if run.start else None,
        "ended": run.end.isoformat(" ") if run.end else None,
        "seconds": W._r(run.seconds) if run.seconds is not None else None,
    }


def _metric_overview(store: ProvStore, doc_id: str, metrics: list[str]) -> dict[str, Any]:
    """The values of a run's metrics, or where they are when the file does not hold them.
    yProv4ML writes each metric's points to a CSV beside the PROV file; a file uploaded on
    its own (the Explorer's case) has the metric's name and the CSV's path, not its values."""
    # The run's outcome is in its loss or accuracy, not in its 15 energy counters: give the
    # headline metrics' values, and leave the rest to metric_summary.
    headline = [m for m in metrics if HEADLINE_METRIC.search(m)] or metrics
    values = {}
    for node_id in headline[:6]:
        stats = _series_stats(store, doc_id, node_id)
        if stats.get("series_available"):
            values[node_id] = {
                k: stats[k] for k in ("final", "min", "min_at_epoch", "max", "max_at_epoch")
            }
    if values:
        out: dict[str, Any] = {"metric_values": values}
        if len(metrics) > len(values):
            out["metric_values_note"] = (
                f"values of {len(values)} of {len(metrics)} metrics shown; metric_summary"
                " gives the others"
            )
        return out
    example = next(
        (
            json.loads(n["attrs"]).get("dcterms:identifier")
            for n in store.nodes(doc_id, "entity")
            if n["node_id"] == metrics[0]
        ),
        None,
    )
    return {
        "metric_values_note": (
            f"the values of these {len(metrics)} metrics are not in this file: yProv4ML"
            f" keeps them in separate CSV files (e.g. {example})"
        )
    }


def activity_stats(
    store: ProvStore,
    doc_id: str,
    group_by: str = "step",
    step: str | None = None,
    status: str | None = None,
    measure: str | None = None,
    sort_by: str = "count",
) -> dict[str, Any]:
    """Aggregate a workflow's tasks (activities): count, time and measures, per group.

    ``group_by``: "step" (the kind of task, e.g. getitem, merge_cubes, Import_variable),
    "status", "module", "cell" (Jupyter cell), "worker", "none" (all tasks as one group), or
    any activity attribute name. ``step`` keeps tasks whose step contains this text;
    ``status`` keeps tasks with this status, or "failed" for every task not ending well.
    ``measure`` names a numeric attribute (e.g. yprov4wfs:ram_usage_mb,
    yprov4wfs:carbon_footprint_kg) to add up per group; ``sort_by`` is "count", "time"
    (summed seconds), "max_time" (slowest task first), "min_time" (fastest task first: for
    "which activity was the shortest"), "measure_sum" or "measure_max" (the group's largest
    single value: use it for "which used the most memory"). The totals give the slowest and
    the fastest single task.
    group_by may name a nested key, e.g. metadata.domain.

    Each group gives tasks, summed_seconds (task durations added up), wall_clock_seconds
    (first start to last end), the slowest task, and the measure's sum, max and how many
    tasks report it; list_sizes, for activities that stand for several tasks (a reduced
    pipeline), how many tasks are behind them. Totals cover every task kept by the filters.
    This is about tasks: the
    size of data (bytes) and a run's input files are in describe_document.
    """
    doc_id = resolve_doc(store, doc_id)
    items, run = _steps_and_run(store, doc_id)
    if not items:
        raise ToolError("this document records no activities")
    if step:
        items = [t for t in items if step.lower() in t.step.lower()]
    if status and not any(t.status for t in items):
        # An answer, not an error: as an error it counted as a failed lookup, and a right
        # "failure is not recorded" was sent back as resting on nothing.
        return {
            "status_filter": status,
            "matching_tasks": None,
            "status_coverage": "no task records a status: whether anything failed is not recorded",
        }
    if status:
        wanted = status.lower()
        items = [
            t
            for t in items
            if (t.failed if wanted == "failed" else (t.status or "").lower() == wanted)
        ]
    if not items:
        raise ToolError("no task matches the filters; describe_document lists steps and statuses")
    if measure is not None and measure not in W.measures(items):
        available = W.measures(items)
        raise ToolError(
            f"no task reports {measure!r} as a number; numeric attributes are: "
            + (", ".join(available) or "none")
            + ". Sizes of data (bytes) are on entities, not tasks: describe_document gives"
            " largest_size."
        )
    sort_by = {"measure": "measure_sum"}.get(sort_by, sort_by)
    sorts = ("count", "time", "max_time", "min_time", "measure_sum", "measure_max")
    if sort_by not in sorts:
        raise ToolError("sort_by must be " + ", ".join(sorts))
    if sort_by.startswith("measure") and measure is None:
        raise ToolError(f"sort_by={sort_by!r} needs a measure")

    def key_of(t: W.Task) -> str:
        if group_by == "step":
            return t.step
        if group_by == "status":
            return t.status or "(no status)"
        if group_by == "module":
            return t.module or "(no module)"
        if group_by == "cell":
            return t.cell if t.cell is not None else "(no cell)"
        if group_by == "worker":
            return t.worker or "(no worker)"
        if group_by == "none":
            return "all tasks"
        value = W.attr(t, group_by)
        return "(missing)" if value is None else str(value)

    groups: dict[str, list[W.Task]] = {}
    for t in items:
        groups.setdefault(key_of(t), []).append(t)
    if group_by not in ("step", "status", "module", "cell", "worker", "none") and set(groups) == {
        "(missing)"
    }:
        raise ToolError(
            f"no task has the attribute {group_by!r}; group_by may be step, status, module,"
            " cell, worker, none, or an attribute name. Facts about the whole run (a model,"
            " scenario, variable, input file) may be in describe_document: shared_attributes,"
            " data_flow"
        )
    # A grouping no task records gives one empty group, which reads as an answer ("all tasks
    # under (no module)"); say so instead.
    if len(groups) == 1 and next(iter(groups)) in (
        "(no status)",
        "(no module)",
        "(no cell)",
        "(no worker)",
    ):
        raise ToolError(
            f"no task records a {group_by}; group_by='step' gives the kind of task (a dask"
            " function, an openEO process, an Ophidia operator)"
        )

    def describe(name: str, group: list[W.Task]) -> dict[str, Any]:
        t = W.timing(group)
        row: dict[str, Any] = {"group": name, "tasks": len(group)}
        if group_by == "cell" and group[0].cell_line:
            row["line"] = str(group[0].cell_line)[:80]
        if t["timed_tasks"]:
            row |= {
                "summed_seconds": t["summed_task_seconds"],
                "wall_clock_seconds": t["wall_clock_seconds"],
                "slowest": t["slowest_task"],
            }
            if name == "all" or sort_by == "min_time":
                row["fastest"] = t["fastest_task"]
        failed = sum(1 for x in group if x.failed)
        if failed:
            row["failed"] = failed
        # Activities that stand for several tasks (a reduced pipeline's aggregates): the
        # number of tasks behind them, which "tasks" (activities) does not show.
        sizes: Counter[str] = Counter()
        for x in group:
            for key, value in x.attrs.items():
                if isinstance(value, list):
                    sizes[key] += len(value)
        if sizes:
            row["list_sizes"] = dict(sizes)
        if measure:
            row[measure] = W.stats(
                [(v, x) for x in group if (v := W.number(x.attrs.get(measure))) is not None]
            )
        return row

    rows = [describe(name, group) for name, group in groups.items()]
    order = {
        "count": lambda r: r["tasks"],
        "time": lambda r: r.get("summed_seconds", 0.0),
        "max_time": lambda r: r.get("slowest", {}).get("seconds", 0.0),
        # Quickest first: the group whose fastest task is shortest.
        "min_time": lambda r: -r.get("fastest", {}).get("seconds", float("inf")),
        "measure_sum": lambda r: r[measure].get("sum", 0.0) if measure else 0.0,
        "measure_max": lambda r: r[measure].get("max", 0.0) if measure else 0.0,
    }[sort_by]
    rows.sort(key=order, reverse=True)
    totals = describe("all", items)
    totals.pop("group")
    out: dict[str, Any] = {"group_by": group_by}
    if sort_by == "min_time" and totals.get("fastest"):
        # First, with every step tied at that duration: from the first group alone the
        # agent named 1 of 4 tied steps.
        out["fastest"] = totals["fastest"]
    if run is not None and not step:
        # The whole run, kept out of the groups. Not given when a step is asked for: its
        # 7.81 s was then reported as the Training step's duration.
        out["run"] = run | {"note": "the whole run, not any of its steps"}
    if measure and totals[measure].get("reported_by"):
        # First, so it is not mistaken for the top group's maximum when sorted otherwise.
        t = totals[measure]
        out["peak"] = {
            "measure": measure,
            "value": t["max"],
            "step": t["max_step"],
            "node_id": t["max_node"],
        }
    out |= {"groups_total": len(rows), "groups": rows[:30], "totals": totals}
    timed = sum(1 for t in items if t.seconds is not None)
    if timed < len(items):
        out["timing_coverage"] = (
            f"{timed} of {len(items)} {'steps' if run else 'tasks'} have both a start and an"
            " end time; the durations of the others are not recorded"
        )
    if not any(t.status for t in items):
        out["status_coverage"] = "no task records a status: whether anything failed is not recorded"
    if len(rows) > 30:
        out["note"] = f"showing 30 of {len(rows)} groups, sorted by {sort_by}"
    return _clip(out)


# ── nodes and lineage ─────────────────────────────────────────────────────────────────


def search_nodes(
    store: ProvStore,
    query: str,
    doc_id: str | None = None,
    kind: str | None = None,
    include_answers: bool = False,
) -> dict[str, Any]:
    """Entities, activities or agents whose id, label, type or attribute values contain
    every word of ``query`` (case-insensitive). If none contains every word, those that
    contain the most of them, with a note saying so. A word also matches what files write
    for it ("range" matches subset_filter, "worker" processed_on); such a hit says so in
    ``matched``, and its ``fields`` give the attributes it named, with their values.
    The agent's own answer records (kind ``qa``) are left out unless ``include_answers``
    is true; set it only for a question about earlier answers.
    """
    if kind is not None and kind not in ("entity", "activity", "agent"):
        raise ToolError("kind must be 'entity', 'activity' or 'agent'")
    if doc_id is not None:
        doc_id = resolve_doc(store, doc_id)
    words = [w.lower() for w in query.split()]
    alternatives = {w: V.alternatives(w) for w in words}
    skip = set() if include_answers or doc_id else _answers(store)
    hits, partial = [], []
    for node in store.nodes(doc_id, kind):
        if node["doc_id"] in skip:
            continue
        haystack = " ".join(
            str(x) for x in (node["node_id"], node["label"], node["prov_type"], node["attrs"])
        ).lower()
        # A word matches as written, or as what files write for it ("range" as
        # subset_filter); a match of the second kind says so.
        via = {}
        for w in words:
            reached = next(((t, why) for t, why in alternatives[w] if t in haystack), None)
            if reached is not None:
                via[w] = reached
        found = len(via)
        if not found:
            continue
        hit = _node_summary(node)
        explained = [why for _, why in via.values() if why]
        if explained:
            hit["matched"] = "; ".join(explained)
        attrs = json.loads(node["attrs"])
        # A parameter's value is usually what was asked ("scheduler" -> CosineAnnealingLR);
        # without it the agent answered "a node called scheduler exists".
        value = attrs.get("prov:value")
        if value is not None:
            hit["value"] = str(value)[:120]
        # The attributes a reached term names, with their values: the hit for "date range"
        # gives the period itself, not only the operator that has it.
        named = {
            k: str(v)[:120]
            for k, v in fields(attrs).items()
            if any(t in k.lower() for t, why in via.values() if why) and not isinstance(v, list)
        }
        if named:
            hit["fields"] = dict(list(named.items())[:3])
        if found == len(words):
            hits.append(hit)
        else:
            partial.append((found, hit))
    if hits or not partial:
        out = _capped(hits, "matches")
        if not hits:
            # A user's words are often not the file's ("learning-rate scheduler" for the
            # parameter "scheduler"; the file's own name, which is no node).
            out["note"] = "nothing matches; describe_document gives an overview of the file"
        return out
    partial.sort(key=lambda x: -x[0])
    out = _capped([n for _, n in partial], "matches")
    out["note"] = "no node contains every word; these contain some of them, most first"
    return out


def get_node(store: ProvStore, doc_id: str, node_id: str) -> dict[str, Any]:
    """One node with all its attributes and every relation it takes part in."""
    doc_id = resolve_doc(store, doc_id)
    rows = store.node(doc_id, node_id)
    if not rows:
        near = search_nodes(store, node_id.split("/")[0].split("_")[0], doc_id)["matches"]
        hint = ", ".join(m["node_id"] for m in near[:10]) or "none"
        raise ToolError(f"no node {node_id!r} in {doc_id}; similar ids: {hint}")
    nodes = []
    for row in rows:
        item = _node_summary(row)
        item["attributes"] = json.loads(row["attrs"])
        nodes.append(item)
    relations = [
        {"relation": r["rel_type"], "subject": r["subject"], "object": r["object"]}
        for r in store.relations_of(doc_id, node_id)
    ]
    out: dict[str, Any] = nodes[0] if len(nodes) == 1 else {"nodes": nodes}
    out["relations"] = relations[:MAX_ITEMS]
    if any(n.get("type") == "provml:Metric" for n in nodes):
        out["metric"] = _series_stats(store, doc_id, node_id)
    return _clip(out)


def trace_lineage(
    store: ProvStore,
    doc_id: str,
    node_id: str,
    direction: str = "upstream",
    depth: int = 3,
) -> dict[str, Any]:
    """Walk the provenance graph from a node.

    Upstream follows what the node depends on (what an entity was generated by or derived
    from, what an activity used or was informed by). Downstream follows what depends on it.
    """
    if direction not in ("upstream", "downstream"):
        raise ToolError("direction must be 'upstream' or 'downstream'")
    doc_id = resolve_doc(store, doc_id)
    if not store.node(doc_id, node_id):
        raise ToolError(f"no node {node_id!r} in {doc_id}")
    depth = max(1, min(int(depth), 10))
    edges = [r for r in store.relations(doc_id) if r["subject"] and r["object"]]
    step_from = "subject" if direction == "upstream" else "object"
    step_to = "object" if direction == "upstream" else "subject"

    seen = {node_id}
    queue = deque([(node_id, 0)])
    found = []
    while queue:
        current, level = queue.popleft()
        if level >= depth:
            continue
        for edge in edges:
            if edge[step_from] != current:
                continue
            target = edge[step_to]
            found.append(
                {"from": current, "relation": edge["rel_type"], "to": target, "hop": level + 1}
            )
            if target not in seen:
                seen.add(target)
                queue.append((target, level + 1))
    return _capped(found, "edges") | {"start": node_id, "direction": direction}


# ── metrics ───────────────────────────────────────────────────────────────────────────


def _series_stats(store: ProvStore, doc_id: str, node_id: str) -> dict[str, Any]:
    points = store.series(doc_id, node_id)
    if not points:
        return {"series_available": False}
    values = [p["value"] for p in points]
    i_min = min(range(len(values)), key=values.__getitem__)
    i_max = max(range(len(values)), key=values.__getitem__)
    return {
        "series_available": True,
        "points": len(values),
        # "final", not "last": a model reading "last" next to "min" picks the wrong one
        # when asked for a run's final value often enough to matter.
        "first": values[0],
        "final": values[-1],
        "min": values[i_min],
        "min_at_epoch": points[i_min]["epoch"],
        "max": values[i_max],
        "max_at_epoch": points[i_max]["epoch"],
        "mean": statistics.fmean(values),
        "epochs": len({p["epoch"] for p in points if p["epoch"] is not None}),
    }


def metric_summary(
    store: ProvStore,
    name: str | None = None,
    context: str | None = None,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """Statistics of logged metrics (yProv4ML ``provml:Metric`` entities).

    ``name`` filters on the metric's label (``MSELoss``, ``cpu_energy``), ``context`` on
    the phase it was logged in (``Training``, ``Validation``). Without ``doc_id`` the
    query runs across every document, which is how runs are compared.
    """
    if doc_id is not None:
        doc_id = resolve_doc(store, doc_id)
    rows = []
    for node in store.nodes(doc_id, "entity"):
        if node["prov_type"] != "provml:Metric":
            continue
        attrs = json.loads(node["attrs"])
        label = str(attrs.get("prov:label") or node["node_id"])
        if name and name.lower() not in label.lower():
            continue
        if context and context.lower() != str(attrs.get("yprov:context", "")).lower():
            continue
        rows.append(
            {
                "doc_id": node["doc_id"],
                "metric": node["node_id"],
                "context": attrs.get("yprov:context"),
                "source": attrs.get("yprov:source"),
                **_series_stats(store, node["doc_id"], node["node_id"]),
            }
        )
    if not rows:
        raise ToolError("no metric matches; call describe_document to see the metrics a run logged")
    return _capped(rows, "metrics")


def metric_series(store: ProvStore, doc_id: str, node_id: str, every: int = 1) -> dict[str, Any]:
    """The raw points of one metric, optionally keeping only every n-th."""
    doc_id = resolve_doc(store, doc_id)
    points = store.series(doc_id, node_id)
    if not points:
        raise ToolError(f"no stored series for {node_id!r} in {doc_id}")
    every = max(1, int(every))
    kept = [{"epoch": p["epoch"], "step": p["step"], "value": p["value"]} for p in points[::every]]
    return {"doc_id": doc_id, "metric": node_id, "points": kept[:200], "total": len(points)}


# ── decisions (yProv4Agents records) ──────────────────────────────────────────────────


def list_decisions(
    store: ProvStore,
    doc_id: str | None = None,
    verdict: str | None = None,
    exclude_verdict: str | None = None,
    agent: str | None = None,
    collection: str | None = None,
    include_answers: bool = False,
) -> dict[str, Any]:
    """Decisions recorded by yProv4Agents agents: the verdict, the rule that produced it,
    the values the rule read, and the justification.

    Each agent has its own verdicts (ACCEPT, ESCALATE, REVISE, OK, ANOMALY, PASS, FAIL,
    COMPLETE, PARTIAL, INCOMPLETE, ...). ``verdict`` keeps one, ``exclude_verdict`` drops
    one ("which decisions were not ACCEPT?"). ``agent`` filters on the deciding agent's
    label (``StorageAgent``, ``CA1``), ``collection`` on the folder the record came from
    (``faults``), which answers a question about a whole group in one call.
    ``verdict_counts`` always counts every verdict before the verdict filters, so nothing a
    filter left out goes unseen.
    The agent's own answer records (kind ``qa``) are left out unless ``include_answers``
    is true; set it only for a question about earlier answers.
    """
    if doc_id is not None:
        doc_id = resolve_doc(store, doc_id)
    agent_labels: dict[tuple[str, str], str] = {}
    for node in store.nodes(doc_id, "agent"):
        agent_labels[(node["doc_id"], node["node_id"])] = node["label"] or node["node_id"]

    folder = collections(store)
    skip = set() if include_answers or doc_id else _answers(store)
    found = []
    for node in store.nodes(doc_id, "activity"):
        if node["doc_id"] in skip:
            continue
        attrs = json.loads(node["attrs"])
        if "yp:verdict" not in attrs:
            continue
        if collection and collection.lower() not in folder[node["doc_id"]].lower():
            continue
        by = [
            agent_labels.get((node["doc_id"], r["object"]), r["object"])
            for r in store.relations_of(node["doc_id"], node["node_id"])
            if r["rel_type"] == "wasAssociatedWith"
        ]
        label = node["label"] or ""
        if (
            agent
            and not any(agent.lower() in str(b).lower() for b in by)
            and (agent.lower() not in label.lower())
        ):
            continue
        inputs = attrs.get("yp:rule_inputs")
        if isinstance(inputs, str):
            with contextlib.suppress(json.JSONDecodeError):
                inputs = json.loads(inputs)
        found.append(
            {
                "doc_id": node["doc_id"],
                "collection": folder[node["doc_id"]],
                "decision": node["node_id"],
                "label": label,
                "agent": by[0] if len(by) == 1 else by,
                "verdict": str(attrs["yp:verdict"]),
                "rule": attrs.get("yp:rule"),
                "rule_inputs": inputs,
                "justification": attrs.get("yp:justification"),
            }
        )
    if not found:
        if collection and not any(collection.lower() in f.lower() for f in folder.values()):
            raise _no_collection(store, collection, folder.values())
        if agent:
            # A wrong guess at the agent's name ("CA1" for NL_Summarizer) read as "there are
            # no decisions here", and the question was declined: say what exists.
            try:
                exist = list_decisions(store, doc_id, include_answers=include_answers)
            except ToolError:
                exist = None
            if exist:
                summary = [
                    {k: d[k] for k in ("agent", "verdict", "rule")} for d in exist["decisions"]
                ]
                raise ToolError(
                    f"no decision by an agent matching {agent!r}; the decisions here are"
                    f" {summary[:10]} (call list_decisions without agent)"
                )
        raise ToolError(
            "no recorded decision matches; decisions exist only in yProv4Agents records"
            " (list_documents shows each document's kind)"
        )
    counts = dict(Counter(d["verdict"] for d in found).most_common())
    out = [
        d
        for d in found
        if (not verdict or d["verdict"].upper() == verdict.upper())
        and (not exclude_verdict or d["verdict"].upper() != exclude_verdict.upper())
    ]
    if not out:
        raise ToolError(f"no decision has that verdict; the verdicts recorded are {counts}")
    return _clip(_capped(out, "decisions") | {"verdict_counts": counts})


#: The tools the agent and the MCP server expose, in the order a model should consider them.
QUERY_TOOLS = (
    list_documents,
    describe_document,
    activity_stats,
    search_nodes,
    count_values,
    get_node,
    trace_lineage,
    metric_summary,
    metric_series,
    list_decisions,
)
