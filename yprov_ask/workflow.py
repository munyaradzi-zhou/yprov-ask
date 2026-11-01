"""Workflow runs read as tasks: what the aggregation tools see in a document's activities.

Workflow producers (yProv4WFs, its openEO extension, Ophidia, analytics4yprov) write one
activity per task, often thousands of them, with machine names such as
``getitem-mean_chunk-a9cc…_0_1_3``. People ask about steps ("which step took longest?"),
not about task ids, and about totals ("how much carbon?"), not about single values. This
module turns activities into ``Task`` records and summarises them; ``tools.py`` exposes the
results.

Conventions, stated once because answers depend on them:

* **Step**: a task's ``yprov4wfs:nice_name`` if it has one; else its ``yprov4wfs:group``
  without the trailing hash; else its label or id without a trailing hash, uuid or ``_(n)``
  counter. ``mean_chunk-mean_agg-aggregate-37e4…`` is the step ``mean_chunk-mean_agg-aggregate``.
* **Wall clock**: from the earliest task start to the latest task end. **Summed time**: the
  durations of the tasks added up. With parallel workers the second is larger, and the two
  answer different questions.
* **Failed**: a status that is not one of ``OK_STATUSES`` (case-insensitive).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .store import ProvStore

#: Statuses that mean a task ended well; any other recorded status counts as failed.
OK_STATUSES = {"success", "ok", "completed", "oph_status_completed", "done", "finished"}

#: Attributes that are numbers but not measurements (positions, indices, levels, ids and
#: ranks: yProv4ML's run_id and global_rank were listed as "measures").
NOT_MEASURES = re.compile(r"cell|line|index|level|^prov:|_id$|rank", re.IGNORECASE)

START_KEYS = ("prov:startTime", "prov:startedAtTime")
END_KEYS = ("prov:endTime", "prov:endedAtTime")


@dataclass
class Task:
    node_id: str
    attrs: dict[str, Any]
    step: str
    start: datetime | None
    end: datetime | None
    status: str | None
    module: str | None
    cell: str | None
    cell_line: str | None
    worker: str | None

    @property
    def seconds(self) -> float | None:
        if self.start and self.end:
            return (self.end - self.start).total_seconds()
        return None

    @property
    def failed(self) -> bool:
        return self.status is not None and self.status.lower() not in OK_STATUSES


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or value.strip() in ("", "None"):
        return None
    try:
        moment = datetime.fromisoformat(value.strip().replace(" ", "T").replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment.replace(tzinfo=None)


def number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def step_name(node_id: str, attrs: dict[str, Any]) -> str:
    nice = attrs.get("yprov4wfs:nice_name")
    if nice not in (None, "", "None"):
        return str(nice)
    group = attrs.get("yprov4wfs:group")
    if group:
        return re.sub(r"-[0-9a-f]{32}.*$", "", str(group))
    label = str(attrs.get("prov:label") or node_id).split(":")[-1]
    return re.sub(r"\d*-[0-9a-f-]{36}$|_\(\d+\)$", "", label) or label


def attr(t: Task, key: str) -> Any:
    """An attribute of a task; ``metadata.domain`` reads a key of a nested object."""
    if key in t.attrs:
        return t.attrs[key]
    head, _, rest = key.partition(".")
    inner = t.attrs.get(head)
    return inner.get(rest) if rest and isinstance(inner, dict) else None


def _flat(attrs: dict[str, Any]) -> dict[str, Any]:
    """Attributes with one level of nested objects spelled ``outer.inner``."""
    out = {}
    for key, value in attrs.items():
        if isinstance(value, dict):
            out |= {f"{key}.{k}": v for k, v in value.items()}
        else:
            out[key] = value
    return out


#: Attributes that differ per task by nature, so never shared facts about the run.
PER_TASK = re.compile(r"time|label|name|cell|line|^prov:|summary|temporal|_id$", re.IGNORECASE)


def shared_attributes(items: list[Task], top: int = 12) -> dict[str, Any]:
    """Values every task that has the attribute agrees on, held by at least half the tasks:
    facts about the whole run (a domain, a data source, an engine, a variable) that would
    otherwise sit unseen on each of thousands of tasks."""
    values: dict[str, set[str]] = {}
    held: Counter[str] = Counter()
    for t in items:
        for key, value in _flat(t.attrs).items():
            if isinstance(value, (dict, list)) or value in (None, "", "None"):
                continue
            if PER_TASK.search(key):
                continue
            held[key] += 1
            values.setdefault(key, set()).add(str(value))
    shared = {
        key: next(iter(v))[:160]
        for key, v in values.items()
        if len(v) == 1 and held[key] * 2 >= len(items) and number(next(iter(v))) is None
    }
    return dict(list(shared.items())[:top])


def _status(attrs: dict[str, Any]) -> str | None:
    for key, value in attrs.items():
        if key == "status" or key.endswith(":status"):
            return str(value)
    return None


def tasks(store: ProvStore, doc_id: str) -> list[Task]:
    out = []
    for node in store.nodes(doc_id, "activity"):
        attrs = json.loads(node["attrs"])
        start = next((parse_time(attrs[k]) for k in START_KEYS if k in attrs), None)
        end = next((parse_time(attrs[k]) for k in END_KEYS if k in attrs), None)
        cell = attrs.get("yprov4wfs:jupyter_cell_index")
        out.append(
            Task(
                node_id=node["node_id"],
                attrs=attrs,
                step=step_name(node["node_id"], attrs),
                start=start,
                end=end,
                status=_status(attrs),
                module=str(attrs["yprov4wfs:module"]) if "yprov4wfs:module" in attrs else None,
                cell=str(cell) if cell not in (None, "None") else None,
                cell_line=attrs.get("yprov4wfs:jupyter_cell_line"),
                worker=str(attrs["yprov4wfs:processed_on"])
                if "yprov4wfs:processed_on" in attrs
                else None,
            )
        )
    return out


def measures(items: list[Task]) -> list[str]:
    """Attributes that hold a number on at least one task and measure something."""
    keys: Counter[str] = Counter()
    for t in items:
        for key, value in t.attrs.items():
            if not NOT_MEASURES.search(key) and number(value) is not None:
                keys[key] += 1
    return [k for k, _ in keys.most_common()]


def _r(x: float) -> float:
    """Six decimal places, or six significant figures for very small values: measures run
    from 1e-7 kg of carbon to 1e4 MB of RAM, and fixed decimals alone erase the small ones
    (seen: 1.9551e-06 became 2e-06)."""
    return round(x, 6) if abs(x) >= 1e-3 or x == 0 else float(f"{x:.6g}")


def timing(items: list[Task]) -> dict[str, Any]:
    timed = [t for t in items if t.seconds is not None]
    if not timed:
        return {"timed_tasks": 0}
    start = min(t.start for t in timed)  # type: ignore[type-var]
    end = max(t.end for t in timed)  # type: ignore[type-var]
    slowest = max(timed, key=lambda t: t.seconds or 0.0)
    fastest = min(timed, key=lambda t: t.seconds or 0.0)
    # Dask gives fused tasks the same start and end, so several tasks often share the
    # longest duration; name every step among them rather than one picked by file order.
    tied = [t for t in timed if t.seconds == slowest.seconds]
    quick = [t for t in timed if t.seconds == fastest.seconds]
    return {
        "timed_tasks": len(timed),
        "first_start": start.isoformat(" "),
        "last_end": end.isoformat(" "),
        "wall_clock_seconds": _r((end - start).total_seconds()),
        "summed_task_seconds": _r(sum(t.seconds or 0.0 for t in timed)),
        "slowest_task": {
            "step": slowest.step,
            "node_id": slowest.node_id,
            "seconds": _r(slowest.seconds or 0.0),
            "tasks_with_this_duration": len(tied),
            "steps_with_this_duration": sorted({t.step for t in tied}),
        },
        # "Which activity was the shortest?" is about one task. Without this the agent gave
        # the step with the least summed time.
        "fastest_task": {
            "step": fastest.step,
            "node_id": fastest.node_id,
            "seconds": _r(fastest.seconds or 0.0),
            "tasks_with_this_duration": len(quick),
            "steps_with_this_duration": sorted({t.step for t in quick}),
        },
    }


def stats(values: list[tuple[float, Task]]) -> dict[str, Any]:
    if not values:
        return {"reported_by": 0}
    top_value, top = max(values, key=lambda x: x[0])
    total = sum(v for v, _ in values)
    return {
        "reported_by": len(values),
        "sum": _r(total),
        "mean": _r(total / len(values)),
        "min": _r(min(v for v, _ in values)),
        "max": _r(top_value),
        "max_step": top.step,
        "max_node": top.node_id,
    }


def summary(items: list[Task], top: int = 12) -> dict[str, Any]:
    """The overview of a workflow run that describe_document returns."""
    statuses = Counter(t.status for t in items if t.status is not None)
    failed = [t for t in items if t.failed]
    by_step: dict[str, list[Task]] = {}
    for t in items:
        by_step.setdefault(t.step, []).append(t)
    steps = sorted(by_step.items(), key=lambda kv: -len(kv[1]))
    out: dict[str, Any] = {
        "tasks": len(items),
        **timing(items),
        "status_counts": dict(statuses.most_common()),
        # With no status recorded, failure is unknown, not zero.
        "failed_tasks": len(failed) if statuses else None,
        "steps_total": len(steps),
        # Sorted by number of tasks; steps_by_time answers "where did the time go".
        "steps": [
            {
                "step": name,
                "tasks": len(group),
                "summed_seconds": _r(sum(t.seconds or 0.0 for t in group)),
            }
            for name, group in steps[:top]
        ],
    }
    by_time = sorted(
        ((name, sum(t.seconds or 0.0 for t in group)) for name, group in by_step.items()),
        key=lambda kv: -kv[1],
    )
    out["steps_by_time"] = [{"step": n, "summed_seconds": _r(s)} for n, s in by_time[:3]]
    if failed:
        out["failed"] = [
            {"step": t.step, "node_id": t.node_id, "status": t.status} for t in failed[:top]
        ]
    groups = Counter(
        re.sub(r"-[0-9a-f]{32}.*$", "", str(t.attrs["yprov4wfs:group"]))
        for t in items
        if t.attrs.get("yprov4wfs:group")
    )
    if groups and set(groups) != set(by_step):
        # Dask's task groups carry what a step name drops, e.g. open_dataset-tas (the
        # variable read) for tasks whose step is "getter".
        out["task_groups"] = dict(groups.most_common(top))
    # xarray names the tasks that read a variable open_dataset-<variable>-<hash>, in the
    # task's id, label or group.
    opened = sorted(
        {
            m.group(1)
            for t in items
            for name in (t.node_id, t.attrs.get("prov:label"), t.attrs.get("yprov4wfs:group"))
            if name and (m := re.match(r"open_dataset-([^-]+)-[0-9a-f]{32}", str(name)))
        }
    )
    if opened:
        out["variables_opened"] = opened
    untimed = [t for t in items if t.seconds is None]
    if 0 < len(untimed) <= 3 and len(untimed) < len(items):
        # A workflow-level activity (openEO's "Workflow 1"): its attributes describe the
        # whole run, e.g. the engine that executed it.
        out["workflow_activities"] = [
            {"node_id": t.node_id, "attributes": t.attrs} for t in untimed
        ]
    shared = shared_attributes(items)
    if shared:
        out["shared_attributes"] = shared
    modules = Counter(t.module for t in items if t.module)
    if modules:
        out["modules"] = dict(modules.most_common(5))
    workers = sorted({t.worker for t in items if t.worker})
    if workers:
        out["workers"] = {"count": len(workers), "names": workers[:8]}
    cells: dict[str, list[Task]] = {}
    for t in items:
        if t.cell is not None:
            cells.setdefault(t.cell, []).append(t)
    if cells:
        out["notebook_cells"] = [
            {"cell": c, "line": str(group[0].cell_line or "")[:80], "tasks": len(group)}
            for c, group in sorted(cells.items(), key=lambda kv: -len(kv[1]))[:5]
        ]
    resources = {}
    for key in measures(items):
        values = [(v, t) for t in items if (v := number(t.attrs.get(key))) is not None]
        resources[key] = stats(values)
    if resources:
        out["measures"] = resources
    out["note"] = (
        "wall_clock_seconds runs from the first task's start to the last task's end;"
        " summed_task_seconds adds every task's duration (larger when tasks ran in"
        " parallel). slowest_task and fastest_task are single tasks (activities); a step's"
        " summed_seconds is all its tasks together, not one task. activity_stats groups and"
        " filters tasks further."
    )
    return out


def _entity_name(node_id: str, label: Any) -> str:
    """An entity as a person would recognise it: its label, unless the label is missing or
    only a Python class name (``<class 'xarray.core.dataarray.DataArray'>``)."""
    text = str(label or "")
    return text if text not in ("", "None") and not text.startswith("<class") else node_id


def data_flow(store: ProvStore, doc_id: str, n: int = 20) -> dict[str, Any]:
    """The run's inputs, read from the graph: entities some activity used and no activity
    generated (input files, datasets), and its final outputs: generated and never used.
    Their names or paths often carry what was read (a variable, a model, a scenario). Left
    out when there are more than ``n``."""
    used: set[str] = set()
    generated: set[str] = set()
    for r in store.relations(doc_id):
        if r["rel_type"] == "used" and r["object"]:
            used.add(r["object"])
        elif r["rel_type"] == "wasGeneratedBy" and r["subject"]:
            generated.add(r["subject"])
    labels = {node["node_id"]: node["label"] for node in store.nodes(doc_id, "entity")}
    out = {}
    for key, ids in (("inputs", used - generated), ("final_outputs", generated - used)):
        names = sorted({_entity_name(i, labels[i]) for i in ids if i in labels})
        # More than n are not the run's data but each task's arguments, as dask records
        # them; a bare count of those read as "the inputs are not named" (seen on dev).
        if 0 < len(names) <= n:
            out[key] = {"count": len(names), "names": names}
    return out


def largest_entities(store: ProvStore, doc_id: str, n: int = 3) -> list[dict[str, Any]]:
    """Entities with the largest recorded size in bytes (yprov4wfs:nbytes and the like)."""
    sized = []
    for node in store.nodes(doc_id, "entity"):
        attrs = json.loads(node["attrs"])
        for key, value in attrs.items():
            if key.lower().endswith("nbytes") and (v := number(value)) is not None:
                sized.append((v, node["node_id"], key))
    sized.sort(reverse=True)
    return [{"node_id": i, "bytes": int(v), "attribute": k} for v, i, k in sized[:n]]


def entities_of_size(store: ProvStore, doc_id: str, size: int) -> int:
    """How many entities record exactly this size in bytes."""
    count = 0
    for node in store.nodes(doc_id, "entity"):
        attrs = json.loads(node["attrs"])
        if any(k.lower().endswith("nbytes") and number(v) == size for k, v in attrs.items()):
            count += 1
    return count
