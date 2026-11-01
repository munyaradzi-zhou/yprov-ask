"""Every answer the agent gives is itself written down as W3C PROV-JSON.

The record uses the demonstrator's ``yp:`` vocabulary, so the same tools that read a
storage agent's decision read this agent's:

    ReceiveTaskActivity  --generates-->  TaskEntity (the question)
    ToolCallActivity  x n  --used--> TaskEntity, --generates--> ToolResultEntity
    ResponseActivity  (the model's draft, with model, temperature and prompt hash)
    DecisionActivity  yp:rule qa.grounding/v1, yp:rule_inputs, yp:verdict
                      --generates--> ResponseEntity (the answer returned)

When the user asked with a file open in the Explorer, ReceiveTaskActivity used a
SourceEntity: that file's URL and the doc_id it was ingested as.

In a conversation, the question is also a member of a ConversationEntity, and the turn's
ReceiveTaskActivity was informed by the previous turn's decision and used its answer (both
in the previous turn's record, referred to by identifier).

A record can be ingested back into the store, so "what did you answer yesterday, and was
it accepted?" is a question the agent can answer about itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__

NAMESPACES = {
    "yp": "http://hpci-lab.org/yProv4Agents#",
    "prov": "http://www.w3.org/ns/prov#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}

AGENT_LABEL = "ProvenanceQA"


def sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _qn(term: str) -> dict[str, str]:
    return {"$": term, "type": "prov:QUALIFIED_NAME"}


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    ok: bool
    result: str
    started: str
    ended: str


@dataclass(frozen=True)
class ConversationLink:
    """Where a turn sits in a conversation: its id, the turn number, and the previous
    turn's session (its record id), if there was one."""

    id: str
    turn: int
    previous_session: str | None = None


@dataclass(frozen=True)
class Source:
    """The file the user had open when asking: where it was fetched from, and its doc_id."""

    url: str
    doc_id: str


@dataclass
class Trace:
    """What happened while answering one question, collected as the graph runs."""

    session: str
    question: str
    received: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    drafts: list[dict[str, Any]] = field(default_factory=list)
    conversation: ConversationLink | None = None
    source: Source | None = None


def build(
    trace: Trace,
    *,
    answer: str,
    verdict: str,
    rule: str,
    rule_inputs: str,
    justification: str,
    model: str,
    temperature: float,
    prompt_sha: str,
) -> dict[str, Any]:
    sid = trace.session
    agent_id = f"yp:agent_{sid}"
    doc: dict[str, Any] = {
        "prefix": dict(NAMESPACES),
        "agent": {agent_id: {"prov:type": _qn("yp:OperationalAgent"), "prov:label": AGENT_LABEL}},
        "activity": {},
        "entity": {},
        "used": {},
        "wasGeneratedBy": {},
        "wasDerivedFrom": {},
        "wasInformedBy": {},
        "wasAssociatedWith": {},
        "hadMember": {},
    }
    counter = {"n": 0}

    def rel(section: str, prefix: str, **ends: str) -> None:
        counter["n"] += 1
        doc[section][f"_:{prefix}{counter['n']}"] = ends

    def activity(key: str, term: str, label: str, start: str, end: str, **attrs: Any) -> str:
        node = f"yp:{term}_{sid}_{key}"
        doc["activity"][node] = {
            "prov:type": _qn(f"yp:{term}"),
            "prov:label": label,
            "prov:startTime": start,
            "prov:endTime": end,
            **attrs,
        }
        rel("wasAssociatedWith", "a", **{"prov:activity": node, "prov:agent": agent_id})
        return node

    def entity(key: str, term: str, value: str, **attrs: Any) -> str:
        node = f"yp:{term}_{sid}_{key}"
        doc["entity"][node] = {"prov:type": _qn(f"yp:{term}"), "prov:value": value, **attrs}
        return node

    receive = activity(
        "00", "ReceiveTaskActivity", "receive a question", trace.received, trace.received
    )
    task = entity(
        "question", "TaskEntity", trace.question, **{"yp:contentHash": sha256(trace.question)}
    )
    rel("wasGeneratedBy", "g", **{"prov:entity": task, "prov:activity": receive})

    if trace.source is not None:
        opened = entity(
            "source", "SourceEntity", trace.source.url, **{"yp:doc_id": trace.source.doc_id}
        )
        rel("used", "u", **{"prov:activity": receive, "prov:entity": opened})

    link = trace.conversation
    if link is not None:
        doc["activity"][receive]["yp:conversation"] = link.id
        doc["activity"][receive]["yp:turn"] = link.turn
        thread = f"yp:conversation_{link.id}"
        doc["entity"][thread] = {
            "prov:type": _qn("yp:ConversationEntity"),
            "prov:label": "a conversation with the agent",
        }
        rel("hadMember", "m", **{"prov:collection": thread, "prov:entity": task})
        if link.previous_session:
            before = link.previous_session
            rel(
                "wasInformedBy",
                "i",
                **{"prov:informed": receive, "prov:informant": f"yp:DecisionActivity_{before}_91"},
            )
            rel(
                "used",
                "u",
                **{"prov:activity": receive, "prov:entity": f"yp:ResponseEntity_{before}_answer"},
            )

    previous = receive
    results = []
    for i, call in enumerate(trace.tool_calls, start=1):
        node = activity(
            f"t{i:02d}",
            "ToolCallActivity",
            f"query the store: {call.name}",
            call.started,
            call.ended,
            **{
                "yp:tool": call.name,
                "yp:tool_args": json.dumps(call.args, sort_keys=True, separators=(",", ":")),
                "yp:outcome": "ok" if call.ok else "error",
            },
        )
        rel("used", "u", **{"prov:activity": node, "prov:entity": task})
        rel("wasInformedBy", "i", **{"prov:informed": node, "prov:informant": previous})
        result = entity(
            f"r{i:02d}",
            "ToolResultEntity",
            call.result[:2000],
            **{"yp:contentHash": sha256(call.result)},
        )
        rel("wasGeneratedBy", "g", **{"prov:entity": result, "prov:activity": node})
        results.append(result)
        previous = node

    tokens_in = sum(int(d.get("prompt_tokens") or 0) for d in trace.drafts)
    tokens_out = sum(int(d.get("completion_tokens") or 0) for d in trace.drafts)
    last = trace.drafts[-1] if trace.drafts else {}
    respond = activity(
        "90",
        "ResponseActivity",
        "draft the answer with the model",
        trace.drafts[0]["started"] if trace.drafts else trace.received,
        last.get("ended", trace.received),
        **{
            "yp:model": model,
            "yp:temperature": temperature,
            "yp:prompt_sha": prompt_sha,
            "yp:prompt_tokens": tokens_in,
            "yp:completion_tokens": tokens_out,
            "yp:model_calls": len(trace.drafts),
        },
    )
    rel("wasInformedBy", "i", **{"prov:informed": respond, "prov:informant": previous})
    for result in results:
        rel("used", "u", **{"prov:activity": respond, "prov:entity": result})

    now = datetime.now(UTC).isoformat()
    decide = activity(
        "91",
        "DecisionActivity",
        "decide whether the answer is grounded in the records",
        now,
        now,
        **{
            "yp:justification": justification,
            "yp:rule": rule,
            "yp:rule_inputs": rule_inputs,
            "yp:verdict": verdict,
            "yp:grounding_check": "figures-in-tool-results",
        },
    )
    rel("wasInformedBy", "i", **{"prov:informed": decide, "prov:informant": respond})
    response = entity("answer", "ResponseEntity", answer, **{"yp:contentHash": sha256(answer)})
    rel("wasGeneratedBy", "g", **{"prov:entity": response, "prov:activity": decide})
    rel("wasDerivedFrom", "d", **{"prov:generatedEntity": response, "prov:usedEntity": task})
    for result in results:
        rel("wasDerivedFrom", "d", **{"prov:generatedEntity": response, "prov:usedEntity": result})

    doc["entity"][f"yp:bundle_{sid}"] = {
        "prov:type": _qn("prov:Bundle"),
        "prov:label": f"provenance record emitted by {AGENT_LABEL}",
        "yp:emitted_by": "yprov-ask",
        "yp:library_version": __version__,
    }
    return doc


def write(doc: dict[str, Any], directory: Path, session: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"qa_{session}_prov.json"
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path
