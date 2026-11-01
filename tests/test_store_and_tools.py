"""The store and the query tools, on the sample records copied from the demonstrator."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from yprov_ask import tools as T
from yprov_ask.store import ProvStore, plain

from .conftest import SAMPLES


def test_every_sample_is_ingested_and_classified(store):
    docs = T.list_documents(store)
    kinds = {d["kind"] for d in docs["documents"]}
    assert docs["total"] == 2 + 24 + 9
    assert kinds == {"yprov4ml", "yprov4agents"}


def test_typed_literals_are_unwrapped():
    assert plain({"$": 0, "type": "xsd:int"}) == 0
    assert plain("4958085^^xsd:double") == 4958085.0
    assert plain("2026-05-20T10:10:10.055075^^xsd:dateTime") == "2026-05-20T10:10:10.055075"


def test_metric_csv_points_are_loaded(store):
    row = T.metric_summary(store, "MSELoss", "Validation", doc_id="prov_example_GR0_0")
    (metric,) = row["metrics"]
    assert metric["series_available"]
    path = SAMPLES / "yprov4ml/example_0/metrics_GR0/MSELoss_Validation_MSELoss_GR0.csv"
    values = [float(line.split(",")[1]) for line in path.read_text().splitlines()[1:]]
    assert metric["points"] == len(values)
    assert metric["min"] == min(values)
    assert metric["final"] == values[-1]


def test_doc_ids_resolve_by_unique_substring(store):
    assert T.resolve_doc(store, "example_GR0_0") == "prov_example_GR0_0"
    with pytest.raises(T.ToolError, match="several"):
        T.resolve_doc(store, "ca1")
    with pytest.raises(T.ToolError, match="no document"):
        T.resolve_doc(store, "nonexistent")


def test_decisions_carry_rule_inputs_and_verdict(store):
    out = T.list_decisions(store, verdict="ESCALATE")
    by_doc = {d["doc_id"]: d for d in out["decisions"]}
    integrity = by_doc["ca1-integrity__ca1_prov"]
    assert integrity["rule"] == "ca1.persist/v1"
    assert integrity["rule_inputs"]["integrity_ok"] is False
    assert integrity["agent"] == "StorageAgent"


def test_lineage_walks_upstream_and_downstream(store):
    doc = "prov_example_GR0_0"
    up = T.trace_lineage(store, doc, "mnist_model.pt/Training", "upstream", 1)
    assert {
        "from": "mnist_model.pt/Training",
        "relation": "wasDerivedFrom",
        "to": "Original_mnist_model.pt/Training",
        "hop": 1,
    } in up["edges"]
    down = T.trace_lineage(store, doc, "Training", "downstream", 1)
    assert any(e["to"] == "mnist_model.pt/Training" for e in down["edges"])


def test_get_node_suggests_ids_when_missing(store):
    with pytest.raises(T.ToolError, match="similar ids"):
        T.get_node(store, "prov_example_GR0_0", "mnist_model.pt/Trainingx")


def test_reingesting_unchanged_file_is_a_no_op(tmp_path: Path):
    src = SAMPLES / "agents/reference/example_0__ca1_prov.json"
    shutil.copy(src, tmp_path / "a.json")
    s = ProvStore(tmp_path / "db.sqlite")
    assert s.ingest_path(tmp_path / "a.json")[0].status == "added"
    assert s.ingest_path(tmp_path / "a.json")[0].status == "unchanged"
    doc = json.loads((tmp_path / "a.json").read_text())
    doc["agent"]["yp:agent_example_0"]["prov:label"] = "Renamed"
    (tmp_path / "a.json").write_text(json.dumps(doc))
    assert s.ingest_path(tmp_path / "a.json")[0].status == "updated"
    assert T.list_documents(s)["total"] == 1
    assert T.describe_document(s, "a")["agents"][0]["label"] == "Renamed"


def test_non_prov_json_is_skipped(tmp_path: Path):
    (tmp_path / "x.json").write_text('{"hello": 1}')
    assert ProvStore().ingest_path(tmp_path) == []


def test_verdict_counts_show_what_a_filter_left_out(store):
    out = T.list_decisions(store, doc_id="ca1-integrity", exclude_verdict="ACCEPT")
    assert out["verdict_counts"] == {"ESCALATE": 1}
    with pytest.raises(T.ToolError, match="verdicts recorded are"):
        T.list_decisions(store, verdict="REJECT")
    not_accept = T.list_decisions(store, exclude_verdict="ACCEPT")
    assert "ACCEPT" not in {d["verdict"] for d in not_accept["decisions"]}
    assert not_accept["verdict_counts"]["ESCALATE"] == 3


def test_collections_group_documents_by_folder(store):
    faults = T.list_documents(store, collection="faults")
    assert faults["total"] == 9
    assert T.list_documents(store)["collections"]["reference"] == 24
    not_accept = T.list_decisions(store, collection="faults", exclude_verdict="ACCEPT")
    assert not_accept["total"] == 8


def test_answer_records_are_left_out_unless_asked_for(tmp_path):
    from yprov_ask.tools import list_decisions, list_documents, search_nodes

    s = ProvStore()
    s.ingest_path(SAMPLES / "agents" / "faults")
    record = {
        "prefix": {"yp": "http://hpci-lab.org/yProv4Agents#"},
        "agent": {"yp:agent_x": {"prov:label": "ProvenanceQA"}},
        "activity": {"yp:DecisionActivity_x_91": {"yp:verdict": "ESCALATE", "prov:label": "d"}},
        "entity": {"yp:bundle_x": {"yp:emitted_by": "yprov-ask"}},
    }
    path = tmp_path / "records" / "qa_x_prov.json"
    path.parent.mkdir()
    path.write_text(json.dumps(record), encoding="utf-8")
    assert s.ingest_file(path).kind == "qa"

    assert list_documents(s)["total"] == 9
    assert list_documents(s, include_answers=True)["total"] == 10
    assert list_decisions(s, verdict="ESCALATE")["total"] == 3
    assert list_decisions(s, verdict="ESCALATE", include_answers=True)["total"] == 4
    assert search_nodes(s, "DecisionActivity_x")["total"] == 0
    assert search_nodes(s, "DecisionActivity_x", include_answers=True)["total"] == 1
    # Naming the document is asking for it.
    assert list_decisions(s, doc_id="qa_x_prov")["total"] == 1


def test_a_store_written_before_qa_records_had_a_kind_is_reclassified(tmp_path):
    db = tmp_path / "old.sqlite"
    s = ProvStore(db)
    s.ingest_path(SAMPLES / "agents" / "faults")
    with s.db:
        s.db.execute(
            "UPDATE documents SET kind = 'yprov4agents', title = 'ProvenanceQA record (qa_1)'"
            " WHERE doc_id = (SELECT MIN(doc_id) FROM documents)"
        )
    s.close()
    kinds = [r["kind"] for r in ProvStore(db).documents()]
    assert kinds.count("qa") == 1


def test_a_scenario_name_given_as_a_collection_points_to_the_documents(store):
    from yprov_ask.tools import ToolError, list_decisions, list_documents

    for call in (list_documents, list_decisions):
        with pytest.raises(ToolError, match=r"ids contain 'cax-revise'.*cax-revise__cax_prov"):
            call(store, collection="cax-revise")
    with pytest.raises(ToolError) as nothing:
        list_documents(store, collection="nonexistent")
    assert "ids contain" not in str(nothing.value)


def test_repeated_descriptions_are_merged():
    from yprov_ask.store import merge

    same = [{"prov:type": "ophidia:datacube"}, {"prov:type": "ophidia:datacube"}]
    assert merge(same) == {"prov:type": "ophidia:datacube"}
    differ = [{"a": 1, "b": [1, 2]}, {"a": 2, "b": [1, 2]}, {"a": 1, "c": "x"}]
    # A key that disagrees keeps every distinct value; a list-valued attribute stays as it is.
    assert merge(differ) == {"a": [1, 2], "b": [1, 2], "c": "x"}


def _write(tmp_path, name, doc):
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_a_node_or_relation_described_twice_is_ingested(tmp_path):
    doc = {
        "prefix": {"ophidia": "http://ophidia.cmcc.it/"},
        "entity": {"ophidia:cube1": [{"prov:type": "ophidia:datacube"}, {"prov:label": "c"}]},
        "activity": {"ophidia:Import_(1)": {"ophidia:status": "OPH_STATUS_COMPLETED"}},
        "used": {
            "_:u1": [
                {"prov:activity": "ophidia:Import_(1)", "prov:entity": "ophidia:cube1"},
                {"prov:activity": "ophidia:Import_(1)", "prov:entity": "ophidia:cube2"},
            ]
        },
    }
    s = ProvStore()
    result = s.ingest_file(_write(tmp_path, "fires.json", doc))
    assert result.kind == "ophidia"
    node = json.loads(s.node(result.doc_id, "ophidia:cube1")[0]["attrs"])
    assert node == {"prov:type": "ophidia:datacube", "prov:label": "c"}
    assert {r["object"] for r in s.relations(result.doc_id)} == {"ophidia:cube1", "ophidia:cube2"}
    assert s.document(result.doc_id)["n_relations"] == 2


def test_producers_are_recognised_from_their_vocabulary(tmp_path):
    from yprov_ask.store import KINDS, classify

    wfs = {"prefix": {"yprov4wfs": "http://example.org"}, "activity": {"t": {}}}
    openeo = {
        "prefix": {"yprov4wfs": "http://example.org"},
        "activity": {"wfs1": {"yprov4wfs:engine": "Openeo-LocalProcessing"}},
    }
    analytics = {"prefix": {"yprov4wfs": "x", "analytics4yprov": "x"}, "activity": {"p": {}}}
    ophidia = {"prefix": {"ophidia": "x"}, "activity": {"o": {}}}
    other = {"prefix": {"ex": "x"}, "entity": {"e": {}}}
    kinds = [classify(d) for d in (wfs, openeo, analytics, ophidia, other)]
    assert kinds == ["yprov4wfs", "openeo", "analytics4yprov", "ophidia", "prov"]
    assert set(kinds) <= set(KINDS)

    # A store written before these kinds existed is corrected when the file is seen again.
    s = ProvStore(tmp_path / "old.sqlite")
    path = _write(tmp_path, "run.json", wfs)
    doc_id = s.ingest_file(path).doc_id
    with s.db:
        s.db.execute(
            "UPDATE documents SET kind = 'prov', title = 'run' WHERE doc_id = ?", (doc_id,)
        )
    assert s.ingest_file(path).status == "unchanged"
    row = s.document(doc_id)
    assert row["kind"] == "yprov4wfs" and row["title"] == "yProv4WFs workflow run (run)"
    assert T.describe_document(s, doc_id)["recorded_with"].startswith("yProv4WFs")


@pytest.mark.skipif(not (SAMPLES.parent / "data").is_dir(), reason="data/ is not present")
def test_every_file_in_data_is_ingested_with_its_producer():
    from eval import gold_data as G

    expected = {
        "yprov4wfs": "yprov4wfs",
        "openeo": "openeo",
        "ophidia": "ophidia",
        "analytics": "analytics4yprov",
    }
    s = ProvStore()
    for path in sorted(G.DATA.glob("*.json")):
        result = s.ingest_file(path)
        assert result is not None, path.name
        assert result.kind == expected[G.producer(G.load(path.name))], path.name
        counts = G.counts(G.load(path.name))
        row = s.document(result.doc_id)
        assert (row["n_activities"], row["n_entities"]) == (counts["activity"], counts["entity"])


def test_count_values_gives_every_value_with_its_count(store):
    import json

    doc = next(r["doc_id"] for r in store.documents())
    attrs = [json.loads(n["attrs"]) for n in store.nodes(doc, None)]
    key = next(k for a in attrs for k in a if k not in ("prov:label", "prov:type"))
    out = T.count_values(store, doc, key.rsplit(":", 1)[-1])
    counts = [v["nodes"] for v in out["values"]]
    assert counts == sorted(counts, reverse=True)
    assert sum(counts) == out["nodes_with_it"] and out["distinct"] == len(out["values"])
    with pytest.raises(T.ToolError):
        T.count_values(store, doc, "no_such_attribute_anywhere")
