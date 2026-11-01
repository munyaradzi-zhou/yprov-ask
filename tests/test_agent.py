"""The LangGraph loop, driven by a scripted model so the control flow is exact."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from yprov_ask import grounding
from yprov_ask.agent import DECLINE_REPLY, ProvenanceAgent
from yprov_ask.store import ProvStore
from yprov_ask.tools import metric_summary

from .conftest import ScriptedModel, call


def _agent(store, replies, tmp_path: Path) -> tuple[ProvenanceAgent, ScriptedModel]:
    model = ScriptedModel(replies=replies, seen=[])
    return ProvenanceAgent(store, model, records_dir=tmp_path), model


def test_grounded_answer_is_accepted_and_recorded(store, tmp_path):
    loss = metric_summary(store, "MSELoss", "Validation", "prov_example_GR0_0")["metrics"][0]
    agent, model = _agent(
        store,
        [
            call("metric_summary", name="MSELoss", context="Validation", doc_id="example_GR0_0"),
            AIMessage(
                f"The lowest validation loss was {loss['min']:.4f} at epoch "
                f"{loss['min_at_epoch']} (prov_example_GR0_0)."
            ),
        ],
        tmp_path,
    )
    answer = agent.ask("What was the lowest validation loss in example 0?")
    assert answer.verdict == "ACCEPT"
    assert answer.tool_calls == [
        {
            "tool": "metric_summary",
            "ok": True,
            "args": {"name": "MSELoss", "context": "Validation", "doc_id": "example_GR0_0"},
        }
    ]
    # The model saw the tool's real output.
    tool_msgs = [m for m in model.seen[1] if isinstance(m, ToolMessage)]
    assert json.loads(tool_msgs[0].content)["metrics"][0]["min"] == loss["min"]

    doc = json.loads(Path(answer.record_path).read_text())
    (decision,) = [a for a in doc["activity"].values() if "yp:verdict" in a]
    assert decision["yp:rule"] == "qa.grounding/v3"
    assert grounding.replay(decision["yp:rule"], decision["yp:rule_inputs"]) == "ACCEPT"
    # The record is itself ingestible provenance, and the agent's decision reads back.
    s = ProvStore()
    s.ingest_file(Path(answer.record_path))
    from yprov_ask.tools import list_decisions

    (d,) = list_decisions(s, include_answers=True)["decisions"]
    assert d["agent"] == "ProvenanceQA" and d["verdict"] == "ACCEPT"


def test_invented_figure_is_sent_back_once_then_reported(store, tmp_path):
    agent, model = _agent(
        store,
        [
            call("list_documents"),
            AIMessage("There are 35 documents and the best loss was 0.0123."),
            AIMessage("There are 35 documents; the best loss was 0.0456."),
        ],
        tmp_path,
    )
    answer = agent.ask("How many documents, and what was the best loss?")
    feedback = [m for m in model.seen[2] if isinstance(m, HumanMessage)][-1]
    assert "0.0123" in feedback.content
    assert answer.verdict == "REVISE"
    assert answer.ungrounded == ["0.0456"]
    assert answer.answer.endswith("0.0456.")


def test_revision_that_fixes_the_figure_is_accepted(store, tmp_path):
    agent, _ = _agent(
        store,
        [
            call("list_documents"),
            AIMessage("There are 36 documents."),
            AIMessage("There are 35 documents."),
        ],
        tmp_path,
    )
    assert agent.ask("How many documents are there?").verdict == "ACCEPT"


def test_tool_errors_go_back_to_the_model_not_the_user(store, tmp_path):
    agent, model = _agent(
        store,
        [
            call("describe_document", doc_id="nope"),
            AIMessage("No document is called that."),
            call("decline", reason="no document called nope"),
        ],
        tmp_path,
    )
    answer = agent.ask("Describe nope")
    assert answer.tool_calls[0]["ok"] is False
    tool_msg = next(m for m in model.seen[1] if isinstance(m, ToolMessage))
    assert "ingested documents are" in json.loads(tool_msg.content)["error"]
    # An answer resting only on a failed lookup is sent back; the model then declines.
    feedback = [m for m in model.seen[2] if isinstance(m, HumanMessage)][-1]
    assert "without looking anything up" in feedback.content
    assert answer.verdict == "OUT_OF_SCOPE"
    assert answer.declined == "no document called nope"


def test_an_off_topic_question_is_declined_with_a_fixed_reply(store, tmp_path):
    agent, model = _agent(store, [call("decline", reason="about football")], tmp_path)
    answer = agent.ask("Which are the best teams in La Liga?")
    assert answer.verdict == "OUT_OF_SCOPE"
    assert answer.answer == DECLINE_REPLY
    assert answer.declined == "about football"
    assert len(model.seen) == 1  # no second model call after declining
    doc = json.loads(Path(answer.record_path).read_text())
    (decision,) = [a for a in doc["activity"].values() if "yp:verdict" in a]
    assert decision["yp:verdict"] == "OUT_OF_SCOPE"
    assert "about football" in decision["yp:justification"]
    assert grounding.replay(decision["yp:rule"], decision["yp:rule_inputs"]) == "OUT_OF_SCOPE"


def test_an_answer_from_general_knowledge_is_sent_back(store, tmp_path):
    agent, model = _agent(
        store,
        [
            AIMessage("Historically, Real Madrid and Barcelona are the best teams."),
            call("decline", reason="about football"),
        ],
        tmp_path,
    )
    answer = agent.ask("Which are the best teams in La Liga?")
    feedback = [m for m in model.seen[1] if isinstance(m, HumanMessage)][-1]
    assert "without looking anything up" in feedback.content
    assert answer.verdict == "OUT_OF_SCOPE"
    assert answer.answer == DECLINE_REPLY


def test_earlier_turns_are_quoted_and_a_repeat_is_sent_back(store, tmp_path):
    earlier = [("Which decisions were escalated?", "Three were escalated, all by StorageAgent.")]
    agent, model = _agent(
        store,
        [
            call("list_decisions", verdict="ESCALATE"),
            AIMessage("Three were escalated, all by StorageAgent."),
            call("decline", reason="not about the records"),
        ],
        tmp_path,
    )
    answer = agent.ask("Who was Italy's prime minister in 2022?", history=earlier)
    # The earlier turn is quoted inside the one user message, not replayed as the model's own.
    first = model.seen[0]
    assert not any(isinstance(m, AIMessage) for m in first)
    [user] = [m for m in first if isinstance(m, HumanMessage)]
    assert "Q: Which decisions were escalated?" in user.content
    assert user.content.endswith("Current question: Who was Italy's prime minister in 2022?")
    feedback = [m for m in model.seen[2] if isinstance(m, HumanMessage)][-1]
    assert "repeats your answer to an earlier question" in feedback.content
    assert answer.verdict == "OUT_OF_SCOPE"


def test_in_file_mode_a_decline_must_follow_a_look_at_the_file(store, tmp_path):
    """A decline before any lookup is refused with a hint; after describe_document it
    stands, and the reply speaks of the open file."""
    from yprov_ask.agent import DECLINE_REPLY_FILE, LOOK_FIRST
    from yprov_ask.record import Source

    doc_id = next(r["doc_id"] for r in store.documents() if "GR0_0" in r["doc_id"])
    agent, model = _agent(
        store,
        [
            call("decline", reason="no rules here"),
            call("describe_document"),
            call("decline", reason="the file does not cover it"),
        ],
        tmp_path,
    )
    answer = agent.ask(f"which rule decided in {doc_id}?", source=Source("upload:x.json", doc_id))
    first = next(m for m in model.seen[1] if isinstance(m, ToolMessage))
    assert json.loads(first.content)["error"] == LOOK_FIRST
    assert [c["tool"] for c in answer.tool_calls] == ["decline", "describe_document", "decline"]
    assert answer.verdict == "OUT_OF_SCOPE" and answer.answer == DECLINE_REPLY_FILE


def test_in_file_mode_an_off_topic_question_is_declined_at_once(store, tmp_path):
    """A question that does not name the file is declined at once: forcing a lookup made
    the agent answer "what is dask?" from the file and general knowledge."""
    from yprov_ask.agent import DECLINE_REPLY_FILE
    from yprov_ask.record import Source

    doc_id = next(r["doc_id"] for r in store.documents() if "GR0_0" in r["doc_id"])
    agent, _ = _agent(store, [call("decline", reason="general")], tmp_path)
    answer = agent.ask("what is dask?", source=Source("upload:x.json", doc_id))
    assert [c["ok"] for c in answer.tool_calls] == [True]
    assert answer.verdict == "OUT_OF_SCOPE" and answer.answer == DECLINE_REPLY_FILE


def test_not_recorded_is_searched_for_before_it_stands(tmp_path):
    """A "not recorded" answer in one-file mode is checked by searching the file for the
    question's words: a lead is shown to the model; empty searches are kept as proof."""
    from yprov_ask.record import Source

    from .test_workflow_tools import _ophidia

    store, doc_id = _ophidia(tmp_path)
    agent, model = _agent(
        store,
        [
            call("describe_document"),
            AIMessage("The time period is not recorded in this file."),
            AIMessage("It read 2090-01-01 to 2090-01-15."),
        ],
        tmp_path,
    )
    answer = agent.ask(
        "Which time period did it process?", source=Source("upload:fires.json", doc_id)
    )
    feedback = [m for m in model.seen[2] if isinstance(m, HumanMessage)][-1]
    assert "2090-01-01_2090-01-15" in feedback.content
    assert answer.verdict == "ACCEPT" and answer.answer.startswith("It read 2090")
    assert {"tool": "search_nodes", "args": {"query": "period"}, "ok": True} in answer.tool_calls

    agent, model = _agent(
        store, [call("describe_document"), AIMessage("The GPU is not recorded.")], tmp_path
    )
    answer = agent.ask("Which GPU ran it?", source=Source("upload:fires.json", doc_id))
    assert len(model.seen) == 2 and answer.answer == "The GPU is not recorded."
    assert [c["args"] for c in answer.tool_calls[1:]] == [{"query": "gpu"}, {"query": "ran"}]
