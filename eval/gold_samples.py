"""Gold answers for questions about the files in ``samples/``, read from the raw JSON.

Like ``eval/gold_data.py``, and for the same reason, this does not use ``yprov_ask``:
a bug in a tool must not also be in the gold answer. It covers the two producers that
``data/`` does not have:

    yprov4ml      a machine-learning run: the run activity (named after the run, the
                  document's default namespace) holds the environment and the run's start
                  and end; Training, Validation and the metric activities have a start time
                  only; parameters are entities with a value; metric values are NOT in the
                  file but in CSV files beside it (``dcterms:identifier``)
    yprov4agents  one agent's record: its activities, and one decision with a verdict, the
                  rule that produced it, the values the rule read and a justification

Facts the files do not record are part of the gold too: no activity of a yProv4ML run has a
status, and only the run activity has an end time.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def load(rel: str) -> dict[str, Any]:
    return json.loads((SAMPLES / rel).read_text(encoding="utf-8"))


def _clean(value: Any) -> Any:
    """'2026-05-20T10:10:05.005879^^xsd:dateTime' -> '2026-05-20T10:10:05.005879'."""
    return value.split("^^")[0] if isinstance(value, str) else value


def _type(node: dict[str, Any]) -> str:
    t = node.get("prov:type")
    return str(t.get("$")) if isinstance(t, dict) else str(t)


def _time(value: Any) -> datetime | None:
    value = _clean(value)
    if not isinstance(value, str) or not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# ── yProv4ML ──────────────────────────────────────────────────────────────────────────


def ml_run(doc: dict[str, Any]) -> dict[str, Any]:
    """The run activity and what it records about the whole run."""
    name = doc["prefix"]["default"].rstrip("/").split("/")[-1]
    run = doc["activity"][name]
    start, end = _time(run.get("prov:startedAtTime")), _time(run.get("prov:endedAtTime"))
    return {
        "name": name,
        "start": start.isoformat(" "),
        "end": end.isoformat(" "),
        "seconds": round((end - start).total_seconds(), 6),
        "python": str(run["yprov:python_version"]).split(" ")[0],
        "os": run["yprov:operating_system"],
        "runtime": run["yprov:runtime_type"],
    }


def ml_steps(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every activity but the run itself, with whether it has an end time."""
    run = ml_run(doc)["name"]
    out = {}
    for node_id, a in doc["activity"].items():
        if node_id == run:
            continue
        out[node_id] = {
            "start": _clean(a.get("prov:startedAtTime") or a.get("prov:startTime")),
            "end": _clean(a.get("prov:endedAtTime") or a.get("prov:endTime")),
        }
    return out


def has_status(doc: dict[str, Any]) -> bool:
    """Whether any activity records a status (none of the yProv4ML samples does)."""
    return any(k == "status" or k.endswith(":status") for a in doc["activity"].values() for k in a)


def ml_entities(doc: dict[str, Any], prov_type: str) -> list[str]:
    return sorted(i for i, e in doc["entity"].items() if _type(e) == prov_type)


def ml_parameters(doc: dict[str, Any]) -> dict[str, str]:
    """Entities with a value and no type: yProv4ML's parameters."""
    return {
        i: str(e["prov:value"])
        for i, e in doc["entity"].items()
        if "prov:value" in e and _type(e) == "None"
    }


def ml_metric_files(doc: dict[str, Any]) -> dict[str, str]:
    """Each metric and the CSV file its values are in (not part of the PROV file)."""
    return {
        i: str(e.get("dcterms:identifier"))
        for i, e in doc["entity"].items()
        if _type(e) == "provml:Metric"
    }


def ml_script(doc: dict[str, Any]) -> str:
    command = ml_parameters(doc)["execution_command"]
    return re.split(r"[/\\]", command.split()[-1])[-1]


# ── yProv4Agents ──────────────────────────────────────────────────────────────────────


def decision(doc: dict[str, Any]) -> dict[str, Any]:
    """The record's one decision."""
    node_id, a = next((i, a) for i, a in doc["activity"].items() if "yp:verdict" in a)
    inputs = a.get("yp:rule_inputs")
    if isinstance(inputs, str):
        inputs = json.loads(inputs)
    return {
        "node_id": node_id,
        "label": a.get("prov:label"),
        "verdict": a["yp:verdict"],
        "rule": a["yp:rule"],
        "rule_inputs": inputs or {},
        "justification": a.get("yp:justification", ""),
        "attributes": {k: v for k, v in a.items() if k.startswith("yp:")},
    }


def agent_name(doc: dict[str, Any]) -> str:
    return str(next(iter(doc["agent"].values()))["prov:label"])


def stored(doc: dict[str, Any]) -> dict[str, Any] | None:
    """The StorageAgent's receipt: the store's PID and the stored size, if any."""
    pid = next((e["yp:pid"] for e in doc["entity"].values() if "yp:pid" in e), None)
    size = next(
        (a["yp:size_bytes"] for a in doc["activity"].values() if "yp:size_bytes" in a), None
    )
    return {"pid": pid, "size_bytes": size} if pid else None
