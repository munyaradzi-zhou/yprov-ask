"""Gold answers for questions about the yProv files in ``data/``, read from the raw JSON.

Deliberately independent of ``yprov_ask``: it does not use the store or the agent's
tools, so a bug in a tool cannot also be in the gold answer. Every function takes a
parsed PROV-JSON document and returns plain values.

Four producers write the files, recognised by their namespaces and attributes:

    yprov4wfs   Python/dask/joblib workflow traces (one activity per task); files ending in
                _jt also record the Jupyter cell that ran each task
    openeo      openEO process graphs traced by yProv4WfS (engine Openeo-LocalProcessing):
                per-process CPU, RAM, execution time and, for some, carbon footprint
    ophidia     Ophidia operator runs (ophidia: namespace); one id may carry several
                attribute sets (a list), which PROV-JSON allows
    analytics   graphs reduced by analytics4yprov: pipelines that aggregate many tasks
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parent.parent / "data"


def load(name: str) -> dict[str, Any]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def nodes(doc: dict[str, Any], kind: str) -> dict[str, dict[str, Any]]:
    """The nodes of one kind; an id with several attribute sets has them merged."""
    out = {}
    for node_id, attrs in (doc.get(kind) or {}).items():
        if isinstance(attrs, list):
            merged: dict[str, Any] = {}
            for part in attrs:
                merged.update(part)
            attrs = merged
        out[node_id] = attrs
    return out


def producer(doc: dict[str, Any]) -> str:
    prefixes = doc.get("prefix") or {}
    acts = nodes(doc, "activity").values()
    if "ophidia" in prefixes:
        return "ophidia"
    if "analytics4yprov" in prefixes:
        return "analytics"
    if any(str(a.get("yprov4wfs:engine", "")).lower().startswith("openeo") for a in acts):
        return "openeo"
    if "yprov4wfs" in prefixes:
        return "yprov4wfs"
    return "prov"


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str) or value in ("", "None"):
        return None
    return datetime.fromisoformat(value.replace(" ", "T"))


def _num(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x


def timed(doc: dict[str, Any]) -> list[tuple[str, dict[str, Any], datetime, datetime]]:
    """Activities with a start and an end time."""
    out = []
    for node_id, a in nodes(doc, "activity").items():
        start, end = _time(a.get("prov:startTime")), _time(a.get("prov:endTime"))
        if start and end:
            out.append((node_id, a, start, end))
    return out


def step_name(node_id: str, a: dict[str, Any]) -> str:
    """The step a task belongs to, as a person would name it."""
    if a.get("yprov4wfs:nice_name") not in (None, "None", ""):
        return str(a["yprov4wfs:nice_name"])
    if a.get("yprov4wfs:group"):
        return re.sub(r"-[0-9a-f]{32}.*$", "", str(a["yprov4wfs:group"]))
    label = a.get("prov:label") or node_id
    return re.sub(r"\d*-[0-9a-f-]{36}$|_\(\d+\)$", "", str(label).split(":")[-1])


# ── facts ─────────────────────────────────────────────────────────────────────────────


def span(doc: dict[str, Any]) -> dict[str, Any]:
    t = timed(doc)
    start = min(x[2] for x in t)
    end = max(x[3] for x in t)
    return {
        "start": start.isoformat(" "),
        "end": end.isoformat(" "),
        "seconds": round((end - start).total_seconds(), 3),
    }


def counts(doc: dict[str, Any]) -> dict[str, int]:
    return {k: len(nodes(doc, k)) for k in ("activity", "entity", "agent")}


def status_counts(doc: dict[str, Any]) -> dict[str, int]:
    key = next(
        (
            k
            for k in ("yprov4wfs:status", "ophidia:status")
            if any(k in a for a in nodes(doc, "activity").values())
        ),
        None,
    )
    if key is None:
        return {}
    return dict(Counter(str(a.get(key)) for a in nodes(doc, "activity").values() if key in a))


def failed(doc: dict[str, Any]) -> list[str]:
    ok = {"success", "ok", "oph_status_completed"}
    out = []
    for node_id, a in nodes(doc, "activity").items():
        status = a.get("yprov4wfs:status", a.get("ophidia:status"))
        if status is not None and str(status).lower() not in ok:
            out.append(step_name(node_id, a))
    return sorted(out)


def steps(doc: dict[str, Any]) -> dict[str, int]:
    """Tasks per step, most first."""
    c = Counter(step_name(i, a) for i, a in nodes(doc, "activity").items())
    return dict(c.most_common())


def slowest(doc: dict[str, Any]) -> dict[str, Any]:
    """The longest task, and every step that has a task of that same duration (ties are
    common: dask gives fused tasks identical start and end times)."""
    t = timed(doc)
    node_id, a, start, end = max(t, key=lambda x: x[3] - x[2])
    longest = end - start
    return {
        "step": step_name(node_id, a),
        "node_id": node_id,
        "seconds": round(longest.total_seconds(), 6),
        "tied_steps": sorted({step_name(i, x) for i, x, s, e in t if e - s == longest}),
    }


def top_ties(counts: dict[str, float]) -> list[str]:
    """Every key that shares the largest value."""
    best = max(counts.values())
    return sorted(k for k, v in counts.items() if v == best)


def time_by_step(doc: dict[str, Any]) -> dict[str, float]:
    total: Counter[str] = Counter()
    for node_id, a, start, end in timed(doc):
        total[step_name(node_id, a)] += (end - start).total_seconds()
    return {k: round(v, 6) for k, v in total.most_common()}


def modules(doc: dict[str, Any]) -> dict[str, int]:
    return dict(
        Counter(
            str(a["yprov4wfs:module"])
            for a in nodes(doc, "activity").values()
            if "yprov4wfs:module" in a
        ).most_common()
    )


def hosts(doc: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(a["yprov4wfs:processed_on"])
            for a in nodes(doc, "activity").values()
            if "yprov4wfs:processed_on" in a
        }
    )


def cells(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Jupyter cell index -> its code line and the number of tasks it ran (_jt files)."""
    out: dict[str, dict[str, Any]] = {}
    for a in nodes(doc, "activity").values():
        if "yprov4wfs:jupyter_cell_index" not in a:
            continue
        cell = out.setdefault(
            str(a["yprov4wfs:jupyter_cell_index"]),
            {"line": a.get("yprov4wfs:jupyter_cell_line"), "tasks": 0},
        )
        cell["tasks"] += 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["tasks"]))


def datasets_opened(doc: dict[str, Any]) -> list[str]:
    """Variables read by open_dataset tasks (yprov4wfs), e.g. 'tas', 'air', 'clt'."""
    found = set()
    for node_id in nodes(doc, "activity"):
        m = re.match(r"open_dataset-([A-Za-z0-9]+)-", node_id)
        if m:
            found.add(m.group(1))
    return sorted(found)


def largest_entity(doc: dict[str, Any]) -> dict[str, Any] | None:
    sized = [(i, _num(e.get("yprov4wfs:nbytes"))) for i, e in nodes(doc, "entity").items()]
    sized = [(i, n) for i, n in sized if n is not None]
    if not sized:
        return None
    i, n = max(sized, key=lambda x: x[1])
    return {"entity": i, "nbytes": int(n)}


def resources(doc: dict[str, Any]) -> dict[str, Any]:
    """openEO: totals and the heaviest process for CPU time, RAM and carbon."""
    acts = nodes(doc, "activity")

    def col(key: str) -> list[tuple[str, float]]:
        return [(i, v) for i, a in acts.items() if (v := _num(a.get(key))) is not None]

    out: dict[str, Any] = {}
    for key, name in (
        ("yprov4wfs:execution_time_sec", "execution_time_sec"),
        ("yprov4wfs:ram_usage_mb", "ram_usage_mb"),
        ("yprov4wfs:cpu_usage", "cpu_usage"),
        ("yprov4wfs:carbon_footprint_kg", "carbon_footprint_kg"),
    ):
        values = col(key)
        if not values:
            continue
        top_id, top = max(values, key=lambda x: x[1])
        out[name] = {
            "n": len(values),
            "sum": sum(v for _, v in values),
            "max": top,
            "max_step": step_name(top_id, acts[top_id]),
            "max_node": top_id,
        }
    return out


def engine(doc: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(a["yprov4wfs:engine"])
            for a in nodes(doc, "activity").values()
            if a.get("yprov4wfs:engine")
        }
    )


def input_files(doc: dict[str, Any]) -> list[str]:
    """openEO: entities named like files (.nc, Sentinel names)."""
    return sorted(
        {
            str(e.get("prov:label"))
            for e in nodes(doc, "entity").values()
            if re.search(r"\.nc$|^SIG0_|^PLIA", str(e.get("prov:label", "")))
        }
    )


def ophidia_runs(doc: dict[str, Any]) -> dict[str, Any]:
    """Ophidia: operators, and the models, scenarios and variables of the input files."""
    ops = Counter(step_name(i, a) for i, a in nodes(doc, "activity").items())
    paths = [i for i in nodes(doc, "entity") if i.startswith("nc:")]
    parts = [re.search(r"CMIP6/\w+/(\w+)/([\w-]+)/(\w+)/\w+/\w+/(\w+)/", p) for p in paths]
    parts = [p.groups() for p in parts if p]
    return {
        "operators": dict(ops.most_common()),
        "institutions": sorted({p[0] for p in parts}),
        "models": sorted({p[1] for p in parts}),
        "scenarios": sorted({p[2] for p in parts}),
        "variables": sorted({p[3] for p in parts}),
        "input_files": len(paths),
    }


def pipelines(doc: dict[str, Any]) -> dict[str, list[str]]:
    """analytics4yprov: each reduced pipeline and the tasks it stands for."""
    return {
        i: list(a.get("analytics4yprov:aggregates") or [])
        for i, a in nodes(doc, "activity").items()
        if a.get("prov:type") == "reduced_pipeline"
    }


def semantic(doc: dict[str, Any]) -> dict[str, Any]:
    """Semantic enrichment (activity 'metadata'): the distinct domain values."""
    meta = [
        a["metadata"]
        for a in nodes(doc, "activity").values()
        if isinstance(a.get("metadata"), dict)
    ]
    return {
        k: sorted({str(m.get(k)) for m in meta})
        for k in ("domain", "subdomain", "variable", "data_source")
    } | {"annotated": len(meta)}


# ── facts added for data_v3 ─────────────────────────────────────────────


def _relations(doc: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    out = []
    for rel in (doc.get(kind) or {}).values():
        out += rel if isinstance(rel, list) else [rel]
    return out


def sources(doc: dict[str, Any]) -> list[str]:
    """Entities some activity used and no activity generated: what the run started from."""
    used = {r.get("prov:entity") for r in _relations(doc, "used")}
    made = {r.get("prov:entity") for r in _relations(doc, "wasGeneratedBy")}
    return sorted(e for e in nodes(doc, "entity") if e in used and e not in made)


def worker_loads(doc: dict[str, Any]) -> dict[str, int]:
    """Tasks per worker, most first."""
    return dict(
        Counter(
            str(a["yprov4wfs:processed_on"])
            for a in nodes(doc, "activity").values()
            if "yprov4wfs:processed_on" in a
        ).most_common()
    )


def measure_peak(doc: dict[str, Any], key: str) -> dict[str, Any]:
    """The largest value of a numeric attribute, and every step with a task at that value."""
    values = [
        (i, a, v) for i, a in nodes(doc, "activity").items() if (v := _num(a.get(key))) is not None
    ]
    top = max(v for _, _, v in values)
    return {"max": top, "steps": sorted({step_name(i, a) for i, a, v in values if v == top})}


def ophidia_periods(doc: dict[str, Any]) -> list[str]:
    """Ophidia: the time ranges the operators subset the input data to."""
    found = set()
    for a in nodes(doc, "activity").values():
        found.update(re.findall(r"subset_filter=([^,]+)", str(a.get("ophidia:arguments", ""))))
    return sorted(found)


def profile(name: str) -> dict[str, Any]:
    """Every fact above that applies to the file."""
    doc = load(name)
    kind = producer(doc)
    out: dict[str, Any] = {
        "file": name,
        "producer": kind,
        "counts": counts(doc),
        "status": status_counts(doc),
        "failed": failed(doc),
        "steps": dict(list(steps(doc).items())[:8]),
    }
    if timed(doc):
        out |= {
            "span": span(doc),
            "slowest": slowest(doc),
            "time_by_step": dict(list(time_by_step(doc).items())[:5]),
        }
    if kind == "yprov4wfs":
        out |= {
            "modules": dict(list(modules(doc).items())[:5]),
            "hosts": hosts(doc),
            "datasets_opened": datasets_opened(doc),
            "largest_entity": largest_entity(doc),
            "cells": {
                k: {"line": str(v["line"])[:60], "tasks": v["tasks"]}
                for k, v in list(cells(doc).items())[:5]
            },
        }
        if any("metadata" in a for a in nodes(doc, "activity").values()):
            out["semantic"] = semantic(doc)
    if kind == "openeo":
        out |= {"engine": engine(doc), "resources": resources(doc), "input_files": input_files(doc)}
    if kind == "ophidia":
        out["ophidia"] = ophidia_runs(doc)
    if kind == "analytics":
        out["pipelines"] = {k: len(v) for k, v in pipelines(doc).items()}
    return out


if __name__ == "__main__":
    for path in sorted(DATA.glob("*.json")):
        print(json.dumps(profile(path.name), default=str))
