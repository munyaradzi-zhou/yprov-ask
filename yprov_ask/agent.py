"""The question-answering agent: a LangGraph loop that may only answer from the tools.

    question
       │
       ▼
    ┌────────┐  tool calls   ┌────────┐
    │ answer │ ────────────► │ tools  │   deterministic queries over the store
    │ (LLM)  │ ◄──────────── │        │
    └────────┘   results     └────────┘
       │ no more tool calls
       ▼
    ┌────────┐  REVISE, revisions left: feedback saying what is wrong
    │ check  │ ─────────────────────────────────────────────► back to answer
    │ (code) │ ◄── the model called decline: OUT_OF_SCOPE, a fixed reply
    └────────┘
       │ ACCEPT, OUT_OF_SCOPE, or out of revisions
       ▼
    record (PROV-JSON) and return

The model chooses which tools to call and words the answer. The check node, which is plain
code, decides whether that answer can be returned as grounded. The final verdict, the rule
and the values it read go into the record, so the decision replays without the model.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict, get_type_hints

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, create_model

from . import grounding, record, vocabulary
from .store import ProvStore
from .tools import QUERY_TOOLS, ToolError

DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"

SYSTEM_PROMPT = """\
You answer questions about W3C PROV provenance records that have been ingested into a store.
The records come from two producers: yProv4ML (machine-learning training runs: parameters,
metrics, models, datasets) and the yProv4Agents demonstrator (agents that record what they
did and why they decided what they decided: verdicts, rules, rule inputs, justifications).

Rules you must follow:
- Use the tools to find every fact you state. Never state a figure, identifier, verdict or
  date that a tool did not return in this conversation.
- Start with list_documents if you do not know which document the question is about. Its
  "collection" is the folder a document came from; a question about a group of records
  ("the fault scenarios") is usually about one collection.
- Prefer one call across documents (filter by collection, verdict, name) over one call per
  document, so nothing is missed.
- Quote figures as the tools gave them; you may round them, but do not convert units or
  compute new figures in your head. If a comparison needs a figure, fetch it.
- Name the document (doc_id) and node each fact comes from, briefly, e.g. (prov_example_GR0_0,
  MSELoss/MSELoss/Validation).
- Answer only from what the tools return. Never add general knowledge, not even as
  background ("historically...", "it is known that...").
- If the question is not about these records (general knowledge, sport, places, people,
  greetings), or the tools return nothing that answers it, call decline with a short reason.
  Do not answer it some other way. If the records hold part of the answer, give that part and
  say what is missing.
- A metric's final value is "final"; its best value is "min" or "max". Use the one the
  question asks for.
- Be concise: a few sentences or a short list.
- The user's message may quote earlier turns of the conversation before the current question.
  They only explain words like "its" or "that run". Answer the current question, never an
  earlier one again, and fetch again any figure you state: only this turn's tool results
  count as evidence."""

PROMPT_SHA = record.sha256(SYSTEM_PROMPT)

#: The prompt when the question is about one file, the one open in the yProv Explorer. The
#: tools are then pinned to that file: they take no doc_id and read nothing else.
FILE_PROMPT = """\
You answer questions about one W3C PROV provenance file: the file the user has open. Your
tools read that file and nothing else. Files like it come from the tools of the yProv
ecosystem: yProv4ML (machine-learning runs), yProv4WFs (workflow tasks, e.g. dask), openEO
process graphs, Ophidia operators, yProv4Agents (agents' decisions), and others.

Rules you must follow:
- Use the tools to find every fact you state. Never state a figure, identifier, verdict or
  date that a tool did not return in this conversation.
- Start with describe_document if you do not yet know what the file holds. For a workflow run
  it gives an overview of the tasks; for counts, durations, failures or totals over many
  tasks, use activity_stats rather than reading tasks one by one.
- Two durations differ: wall clock (first task start to last task end) and summed task time
  (task durations added up, larger when tasks ran in parallel). Say which one you give.
- An activity is one task. The longest is slowest_task, the shortest fastest_task (one task
  each); a step's summed_seconds adds all its tasks. Do not give one as the other.
- If several steps tie (same duration or count), name all of them.
- If a total covers only the tasks that report the value (e.g. carbon on 27 of 88), say so.
- "Most" or "peak" of a measure (memory, CPU, carbon) means its largest single value: the
  max and max_step in the totals, not the group with the largest sum.
- Facts about the whole run (domain, data source, variable, engine) may sit in the
  overview's shared_attributes, variables_opened, task_groups or workflow_activities; look
  there before saying the file does not record them. Those are annotations written on the
  tasks; for which step or module is most common or took most time, use the counts and
  times (steps, steps_by_time, modules, activity_stats), which come from the tasks.
  "recorded_with" is the library that wrote the file, not the engine that ran the work.
- What a run read or produced (input files, datasets) is describe_document's data_flow:
  inputs are entities used and never generated. Their names or paths often carry the
  variable, model or scenario; read them there before saying the file does not record it.
  If you found the fact anywhere in the file, give it and say where it comes from (e.g. the
  input file paths); do not also say it is not recorded.
- describe_document's "attributes" names every attribute in the file, with its values when
  there are few (argument strings split: ophidia:arguments.subset_filter is the period an
  operator read). Look there before saying the file does not record something;
  search_nodes finds the nodes that carry an attribute.
- How many nodes have each value of an attribute (the most common dtype, how many entities
  are numpy arrays, which value occurs most) is count_values: every value with its count,
  most common first. Do not judge "most common" from examples or a sample.
- The size of data ("largest", "biggest result", bytes) is on entities: largest_size, which
  says how many entities have that size; its examples are not a count.
  activity_stats is about tasks and their time, never about data size.
- Whether tasks or operators completed, failed or are "ok" is their status (status_counts,
  failed_tasks). Answer it; it is not a recorded decision.
- An activity that stands for several tasks (a reduced pipeline) lists them in an attribute;
  list_sizes gives how many. Count that list, not the activity.
- Quote figures as the tools gave them; you may round them, but do not convert units or
  compute new figures in your head. If a comparison needs a figure, fetch it.
- A fact the file does not record is "not recorded", never zero: with no status there is no
  failure count, with no end time there is no duration, and metric values kept in separate
  CSV files are not in this file. Say what is missing and what the file does hold.
- Answer the question asked. A short question ("os of X", "scheduler X") wants that one
  fact. Only when asked for a summary ("what does this file record?"), say first what
  happened and its outcome (what ran, on what, how it ended), then a few key facts.
- Name the node a fact comes from when it helps, e.g. (MSELoss/MSELoss/Validation). Never
  name the tools you used; the user sees them separately.
- Answer only from what the tools return. Never add general knowledge, not even as
  background ("historically...", "it is known that...").
- The user may name the open file by its file name; that is this file.
- If the question is about another file, the whole store, or anything this file does not
  record (general knowledge, sport, places, people, greetings), call decline with a short
  reason. Do not answer it some other way. If the file holds part of the answer, give that
  part and say what is missing.
- A metric's final value is "final"; its best value is "min" or "max". Use the one the
  question asks for.
- Be concise: a few sentences or a short list.
- The user's message may quote earlier turns of the conversation before the current question.
  They only explain words like "its" or "that step". Answer the current question, never an
  earlier one again, and fetch again any figure you state: only this turn's tool results
  count as evidence."""

FILE_PROMPT_SHA = record.sha256(FILE_PROMPT)

#: Open files whose pinned tools are kept ready, most recent last.
PINNED_CACHE = 32

#: Rounds of tool calls before the model is made to answer with what it has.
MAX_TOOL_ROUNDS = 8

#: Times an answer with unsupported figures is sent back before it is returned as REVISE.
MAX_REVISIONS = 1

#: What the user is told when the agent declines. Fixed, so a declined question never gets
#: an answer the model made up; the model's reason goes into the record.
DECLINE_REPLY = (
    "The provenance records here don't answer this question. I can only answer questions"
    " about them: runs, parameters, metrics, lineage, and the decisions agents recorded."
)

#: The same, when the question is about the one file the user has open.
DECLINE_REPLY_FILE = (
    "This file doesn't answer that question. Ask me about what it records: the run's steps,"
    " timings, data, parameters, metrics, failures or decisions."
)

#: In one-file mode, a first decline with no lookup is sent back once when the question
#: names the open file. Questions such as "rule of example_0__ca2" were declined after no
#: lookup (samples dev baseline: 19 of 66 answerable questions). Off-topic
#: questions do not name the file, and forcing a lookup on them ("what is dask?") made the
#: agent answer from the file and general knowledge; a second decline always stands.
LOOK_FIRST = (
    "If the question is about this file (its run, steps, data, parameters, metrics, agents,"
    " decisions or rules, even by the file's name), call describe_document first. If it is"
    " about something else (general knowledge, how-to, other topics), call decline again."
)


#: An answer that says the file does not have something.
NOT_RECORDED = re.compile(
    r"\bnot (?:recorded|available|included|provided|specified|listed|mentioned|stated|given)\b"
    r"|\bdo(?:es)?(?:n't| not) (?:record|include|contain|specify|say|list|mention)\b"
    r"|\bno (?:information|record|data|mention) (?:on|about|of)\b"
    r"|\bcannot be determined\b",
    re.IGNORECASE,
)

#: A question word that finds more nodes than this names nothing in particular ("time").
MAX_LEADS = 20


class _DeclineArgs(BaseModel):
    reason: str = Field(description="Why the records cannot answer, in a few words.")


def _decline_tool() -> StructuredTool:
    return StructuredTool(
        name="decline",
        description=(
            "Call this, and nothing else, when the question is not about the provenance records"
            " here (general knowledge, sport, places, people, greetings) or the tools found"
            " nothing that answers it. The user then gets a fixed reply saying the records do"
            " not cover the question."
        ),
        args_schema=_DeclineArgs,
        func=lambda reason: {"declined": reason},
    )


def _tool_for(func: Callable[..., Any], store: ProvStore) -> StructuredTool:
    """A LangChain tool around a query function, with the store argument bound."""
    hints = get_type_hints(func)
    params = list(inspect.signature(func).parameters.values())[1:]  # drop ``store``
    fields: dict[str, Any] = {}
    for p in params:
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[p.name] = (hints[p.name], Field(default=default))
    schema = create_model(f"{func.__name__}_args", **fields)

    def run(**kwargs: Any) -> Any:
        return func(store, **kwargs)

    return StructuredTool(
        name=func.__name__,
        description=inspect.getdoc(func) or func.__name__,
        args_schema=schema,
        func=run,
    )


def _pinned_tool(func: Callable[..., Any], store: ProvStore, doc_id: str) -> StructuredTool:
    """A query function with both the store and the document bound: the model cannot ask
    about any other document, because the argument that names one is gone."""
    hints = get_type_hints(func)
    params = [
        p for p in list(inspect.signature(func).parameters.values())[1:] if p.name != "doc_id"
    ]
    fields: dict[str, Any] = {}
    for p in params:
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[p.name] = (hints[p.name], Field(default=default))
    schema = create_model(f"{func.__name__}_file_args", **fields)

    def run(**kwargs: Any) -> Any:
        return func(store, doc_id=doc_id, **kwargs)

    return StructuredTool(
        name=func.__name__,
        description="(Reads the open file only.) " + (inspect.getdoc(func) or func.__name__),
        args_schema=schema,
        func=run,
    )


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_rounds: int
    revisions: int
    verdict: str
    grounding: dict[str, Any]


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    verdict: str
    ungrounded: list[str]
    figures_checked: int
    tool_calls: list[dict[str, Any]]
    model: str
    seconds: float
    record_path: str | None
    session: str = ""
    declined: str | None = None  # the model's reason, if it declined
    reasons: tuple[str, ...] = ()  # why the verdict is REVISE
    data_lookups: int = 0
    repeat_similarity: float = 0.0
    small_numbers_unchecked: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "verdict": self.verdict,
            "declined": self.declined,
            "grounding": {
                "rule": grounding.GROUNDING_RULE.name,
                "figures_checked": self.figures_checked,
                "ungrounded_figures": self.ungrounded,
                "data_lookups": self.data_lookups,
                "repeat_similarity": self.repeat_similarity,
                "small_numbers_unchecked": self.small_numbers_unchecked,
                "reasons": list(self.reasons),
            },
            "tool_calls": self.tool_calls,
            "model": self.model,
            "seconds": round(self.seconds, 2),
            "record": self.record_path,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat()


def default_model(name: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    """The OpenAI chat model, verifying TLS against the operating system's certificates.

    On networks whose proxy re-signs TLS with a locally trusted root, the bundled
    certificates fail and the client cannot connect at all; truststore fixes that and is
    harmless elsewhere.
    """
    import truststore
    from langchain_openai import ChatOpenAI

    truststore.inject_into_ssl()
    return ChatOpenAI(
        model=name or os.environ.get("YPROV_ASK_MODEL", DEFAULT_MODEL),
        temperature=temperature,
        timeout=60,
        max_retries=2,
    )


class ProvenanceAgent:
    """Answers natural-language questions about the documents in a :class:`ProvStore`."""

    def __init__(
        self,
        store: ProvStore,
        model: BaseChatModel | None = None,
        *,
        model_name: str | None = None,
        temperature: float = 0.0,
        records_dir: str | Path | None = "records",
        tools: Sequence[Callable[..., Any]] = QUERY_TOOLS,
    ) -> None:
        self.store = store
        self.model = model if model is not None else default_model(model_name, temperature)
        self.model_name = str(
            getattr(self.model, "model_name", None)
            or getattr(self.model, "model", None)
            or model_name
            or "unknown"
        )
        self.temperature = temperature
        self.records_dir = Path(records_dir) if records_dir is not None else None
        self.query_functions = tuple(tools)
        self.tools = {t.__name__: _tool_for(t, store) for t in tools}
        self.tools["decline"] = _decline_tool()
        self.bound = self.model.bind_tools(list(self.tools.values()))
        self._pinned: dict[str, tuple[dict[str, StructuredTool], Any]] = {}
        # What the graph uses for the question in hand: the store-wide set, or one file's.
        self._active_tools = self.tools
        self._active_bound = self.bound
        self._prompt = SYSTEM_PROMPT
        # One question at a time: the trace is per question, and the MCP server may
        # receive two at once.
        self._lock = threading.Lock()
        self._trace: record.Trace | None = None
        self._focus: str | None = None
        self._names_file = False
        self._file_words: set[str] = set()
        self._on_step: Callable[[str], None] | None = None
        self._history: Sequence[tuple[str, str]] = ()
        self._declined: str | None = None
        self.graph = self._build()

    # ── graph ─────────────────────────────────────────────────────────────────────────

    def _build(self) -> Any:
        graph = StateGraph(State)
        graph.add_node("answer", self._answer)
        graph.add_node("tools", self._tools)
        graph.add_node("check", self._check)
        graph.add_edge(START, "answer")
        graph.add_conditional_edges("answer", self._after_answer, ["tools", "check"])
        graph.add_conditional_edges("tools", self._after_tools, ["answer", "check"])
        graph.add_conditional_edges("check", self._after_check, ["answer", END])
        return graph.compile()

    def _answer(self, state: State) -> dict[str, Any]:
        messages: list[AnyMessage] = [SystemMessage(self._prompt)]
        if self._focus:
            messages.append(SystemMessage(self._focus))
        messages += state["messages"]
        out_of_rounds = state["tool_rounds"] >= MAX_TOOL_ROUNDS
        if out_of_rounds:
            messages.append(
                HumanMessage("No more tool calls. Answer now from the results you have.")
            )
        started = _now()
        reply = (self.model if out_of_rounds else self._active_bound).invoke(messages)
        usage = getattr(reply, "usage_metadata", None) or {}
        assert self._trace is not None
        self._trace.drafts.append(
            {
                "started": started,
                "ended": _now(),
                "prompt_tokens": usage.get("input_tokens"),
                "completion_tokens": usage.get("output_tokens"),
            }
        )
        return {"messages": [reply]}

    @staticmethod
    def _after_answer(state: State) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "check"

    def _step(self, message: str) -> None:
        if self._on_step is not None:
            self._on_step(message)

    def _tools(self, state: State) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage) and self._trace is not None
        replies = []
        for call in last.tool_calls:
            self._step(f"Looking up: {call['name']}")
            started = _now()
            tool = self._active_tools.get(call["name"])
            ok = True
            try:
                if tool is None:
                    raise ToolError(
                        f"no tool {call['name']!r}; tools are {', '.join(self._active_tools)}"
                    )
                if (
                    call["name"] == "decline"
                    and self._names_file
                    and not any(c.ok and c.name != "decline" for c in self._trace.tool_calls)
                    and not any(c.name == "decline" for c in self._trace.tool_calls)
                ):
                    raise ToolError(LOOK_FIRST)
                result = json.dumps(tool.invoke(call["args"]), default=str)
            except (ToolError, ValueError, TypeError) as exc:
                ok = False
                result = json.dumps({"error": str(exc)})
            if call["name"] == "decline" and ok:
                self._declined = str(call["args"].get("reason", "")).strip() or "no reason given"
            self._trace.tool_calls.append(
                record.ToolCall(call["name"], dict(call["args"]), ok, result, started, _now())
            )
            replies.append(ToolMessage(result, tool_call_id=call["id"], name=call["name"]))
        return {"messages": replies, "tool_rounds": state["tool_rounds"] + 1}

    def _decline_reply(self) -> str:
        return DECLINE_REPLY_FILE if self._focus is not None else DECLINE_REPLY

    def _tools_for(self, doc_id: str) -> tuple[dict[str, StructuredTool], Any]:
        """The tools pinned to one document, and the model bound to them (cached)."""
        if doc_id in self._pinned:
            self._pinned[doc_id] = self._pinned.pop(doc_id)  # most recently used last
            return self._pinned[doc_id]
        pinned = {
            f.__name__: _pinned_tool(f, self.store, doc_id)
            for f in self.query_functions
            if "doc_id" in inspect.signature(f).parameters
        }
        pinned["decline"] = self.tools["decline"]
        entry = (pinned, self.model.bind_tools(list(pinned.values())))
        self._pinned[doc_id] = entry
        while len(self._pinned) > PINNED_CACHE:
            self._pinned.pop(next(iter(self._pinned)))
        return entry

    def _after_tools(self, state: State) -> str:
        # A declined question goes straight to the check: there is nothing left to draft.
        return "check" if self._declined is not None else "answer"

    def _check(self, state: State) -> dict[str, Any]:
        assert self._trace is not None
        declined = self._declined is not None
        self._step("Checking every figure against the records")
        answer = self._decline_reply() if declined else _text(state["messages"][-1])
        sources: list[Any] = [self._trace.question]
        if self._trace.source is not None:
            sources.append({"doc_id": self._trace.source.doc_id, "url": self._trace.source.url})
        lookups = [c for c in self._trace.tool_calls if c.name != "decline"]
        for call in lookups:
            if call.ok:
                sources.append(json.loads(call.result))
        result = grounding.check(
            answer,
            grounding.facts_from(sources),
            labelled=grounding.labelled_facts(sources[1:]),
            tool_calls=len(self._trace.tool_calls),
            data_lookups=sum(c.ok for c in lookups),
            repeat_similarity=(
                0.0
                if declined
                else grounding.repeat_similarity(answer, self._trace.question, self._history)
            ),
            declined=declined,
        )
        reasons = (
            grounding.reasons(result.evidence, result.ungrounded, result.misplaced)
            if result.verdict == "REVISE"
            else []
        )
        update: dict[str, Any] = {
            "verdict": result.verdict,
            "grounding": {
                "evidence": result.evidence,
                "ungrounded": result.ungrounded,
                "figures_checked": result.figures_checked,
                "reasons": reasons,
            },
        }
        if (
            result.verdict == "ACCEPT"
            and state["revisions"] < MAX_REVISIONS
            and self._focus is not None
            and NOT_RECORDED.search(answer)
        ):
            leads = self._search_before_absent()
            if leads:
                update["messages"] = [
                    HumanMessage(
                        "Your answer says something is not recorded. A search of the file for"
                        " the question's words found these entries: "
                        + json.dumps(leads, default=str)
                        + " If they answer the question, answer from them. If they do not,"
                        " keep your answer."
                    )
                ]
                update["revisions"] = state["revisions"] + 1
                self._step("Searching the file before saying it is not recorded")
                return update
        if result.verdict == "REVISE" and state["revisions"] < MAX_REVISIONS:
            problems = []
            if result.misplaced:
                problems.append(
                    "The records do not support how you used these figures: "
                    + "; ".join(grounding.misplaced_reasons(result.misplaced))
                    + ". Check which item each figure belongs to and correct the answer."
                )
            if result.ungrounded:
                problems.append(
                    "These figures in your answer were not returned by any tool: "
                    + ", ".join(result.ungrounded)
                    + ". Fetch them with a tool, quote them as returned, or remove them."
                )
            if not result.evidence["data_lookups"]:
                problems.append(
                    "You answered without looking anything up. Use the tools to find the"
                    " answer in the records, or call decline if they do not cover the question."
                )
            if float(result.evidence["repeat_similarity"] or 0) >= grounding.REPEAT_THRESHOLD:
                problems.append(
                    "Your answer repeats your answer to an earlier question. Answer the"
                    f" current question: {self._trace.question!r}. If the records do not"
                    " cover it, call decline."
                )
            update["messages"] = [HumanMessage(" ".join(problems))]
            update["revisions"] = state["revisions"] + 1
            self._step("The check found a problem; revising the answer")
        return update

    def _search_before_absent(self) -> dict[str, Any]:
        """Search the open file for each word of the question before "not recorded" stands.

        Each search goes into the record as a lookup, so a "not recorded" that stands has
        the empty searches behind it; a word that finds a few nodes is a lead the model is
        shown. Ophidia's period, inside an argument string, was "not recorded".
        """
        assert self._trace is not None
        search = self._active_tools.get("search_nodes")
        if search is None:
            return {}
        leads: dict[str, Any] = {}
        for word in vocabulary.content_words(self._trace.question):
            if word in self._file_words:
                continue
            started = _now()
            found = search.invoke({"query": word})
            self._trace.tool_calls.append(
                record.ToolCall(
                    "search_nodes",
                    {"query": word},
                    True,
                    json.dumps(found, default=str),
                    started,
                    _now(),
                )
            )
            if 0 < found["total"] <= MAX_LEADS and "note" not in found:
                leads[word] = found["matches"][:5]
            if len(self._trace.tool_calls) >= 3 * MAX_TOOL_ROUNDS:
                break
        return leads

    @staticmethod
    def _after_check(state: State) -> str:
        last = state["messages"][-1]
        return "answer" if isinstance(last, HumanMessage) else END

    # ── public ────────────────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        *,
        history: Sequence[tuple[str, str]] = (),
        conversation: record.ConversationLink | None = None,
        source: record.Source | None = None,
        on_step: Callable[[str], None] | None = None,
    ) -> Answer:
        """Answer ``question``. ``history`` is the conversation's earlier (question, answer)
        pairs, oldest first; ``conversation`` links this turn's record to the turn before;
        ``source`` is the document the user has open, which "this file" refers to;
        ``on_step`` is told, in a few words, what the agent is doing as it goes."""
        with self._lock:
            self._on_step = on_step
            try:
                return self._ask(question, history, conversation, source)
            finally:
                self._on_step = None

    def _ask(
        self,
        question: str,
        history: Sequence[tuple[str, str]],
        conversation: record.ConversationLink | None,
        source: record.Source | None,
    ) -> Answer:
        session = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self._trace = record.Trace(
            session=session,
            question=question,
            received=_now(),
            conversation=conversation,
            source=source,
        )
        if source is not None:
            # One file: tools pinned to it, and the prompt that says so.
            self._active_tools, self._active_bound = self._tools_for(source.doc_id)
            self._prompt = FILE_PROMPT
            self._focus = f"The open file is {source.doc_id} (from {source.url})."
            # The doc id is the file name plus a content hash: "example_0__ca2_prov-7af6...".
            stem = re.sub(r"-[0-9a-f]{10}$", "", source.doc_id).removesuffix("_prov").lower()
            self._names_file = bool(stem) and stem in question.lower()
            self._file_words = set(vocabulary.content_words(stem))
        else:
            self._active_tools, self._active_bound = self.tools, self.bound
            self._prompt = SYSTEM_PROMPT
            self._focus = None
            self._names_file = False
            self._file_words = set()
        self._history = history
        self._declined = None
        t0 = time.perf_counter()
        final = self.graph.invoke(
            {
                "messages": [HumanMessage(_with_history(question, history))],
                "tool_rounds": 0,
                "revisions": 0,
                "verdict": "",
                "grounding": {},
            },
            {"recursion_limit": 4 * (MAX_TOOL_ROUNDS + MAX_REVISIONS + 2)},
        )
        seconds = time.perf_counter() - t0
        declined = self._declined
        if declined is not None:
            answer = self._decline_reply()
        else:
            answer = _text(next(m for m in reversed(final["messages"]) if isinstance(m, AIMessage)))
        g = final["grounding"]
        trace = self._trace
        self._trace = None
        self._focus = None
        self._history = ()
        self._declined = None

        record_path = None
        if self.records_dir is not None:
            self._step("Writing the provenance record")
            doc = record.build(
                trace,
                answer=answer,
                verdict=final["verdict"],
                rule=grounding.GROUNDING_RULE.name,
                rule_inputs=json.dumps(g["evidence"], sort_keys=True, separators=(",", ":")),
                justification=_justify(final["verdict"], g, len(trace.tool_calls), declined),
                model=self.model_name,
                temperature=self.temperature,
                prompt_sha=record.sha256(self._prompt),
            )
            record_path = str(record.write(doc, self.records_dir, session))

        return Answer(
            question=question,
            answer=answer,
            verdict=final["verdict"],
            ungrounded=g["ungrounded"],
            figures_checked=g["figures_checked"],
            tool_calls=[{"tool": c.name, "args": c.args, "ok": c.ok} for c in trace.tool_calls],
            model=self.model_name,
            seconds=seconds,
            record_path=record_path,
            session=session,
            declined=declined,
            reasons=tuple(g.get("reasons", ())),
            data_lookups=int(g["evidence"]["data_lookups"]),
            small_numbers_unchecked=int(g["evidence"].get("small_numbers_unchecked", 0)),
            repeat_similarity=float(g["evidence"]["repeat_similarity"]),
        )


def _text(message: AnyMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part) for part in content
    )


def _with_history(question: str, history: Sequence[tuple[str, str]]) -> str:
    """The user's message: earlier turns quoted as context, then the current question.

    Earlier answers are quoted, not given back to the model as its own messages: given as
    its own, a model with nothing to say about a new question was seen to answer an earlier
    one again ("who was Italy's prime minister in 2022?").
    """
    if not history:
        return question
    quoted = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in history)
    return (
        "Earlier in this conversation (context only, to explain words like 'it'; do not"
        " answer these again, and their figures are not evidence):\n\n"
        f"{quoted}\n\nCurrent question: {question}"
    )


def _justify(verdict: str, g: dict[str, Any], tool_calls: int, declined: str | None) -> str:
    checked = g["figures_checked"]
    if verdict == "OUT_OF_SCOPE":
        return (
            f"The agent declined: {declined}. The user was given the fixed reply that the"
            " records do not cover the question. Verdict OUT_OF_SCOPE."
        )
    if verdict == "ACCEPT":
        return (
            f"Every one of the {checked} figure(s) checked in the answer was returned by a"
            f" tool ({tool_calls} call(s)), the answer rests on at least one lookup, and it"
            " does not repeat an earlier answer. This shows no figure was invented; it does"
            " not show that the answer's wording is right. Verdict ACCEPT."
        )
    return "The answer was not accepted: " + "; ".join(g.get("reasons") or []) + ". Verdict REVISE."
