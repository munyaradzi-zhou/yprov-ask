"""The workflow overview and activity_stats: synthetic cases, and agreement with the
independent gold code on every real file in data/ (skipped when data/ is absent)."""

from __future__ import annotations

import json

import pytest

from yprov_ask import tools as T
from yprov_ask.store import ProvStore

from .conftest import SAMPLES

DATA = SAMPLES.parent / "data"


def _run(tmp_path) -> tuple[ProvStore, str]:
    """A small workflow: two workers, a tie for the slowest task, one failure, a measure."""
    act = {
        "a1": ("load", "2025-01-01 10:00:00", "2025-01-01 10:00:04", "success", "w1", "1.5e-07"),
        "a2": ("mean", "2025-01-01 10:00:01", "2025-01-01 10:00:05", "success", "w2", "2.5e-07"),
        "a3": ("mean", "2025-01-01 10:00:05", "2025-01-01 10:00:06", "success", "w1", None),
        "a4": ("save", "2025-01-01 10:00:06", "2025-01-01 10:00:08", "Failed", "w2", "None"),
    }
    doc = {
        "prefix": {"yprov4wfs": "http://example.org"},
        "activity": {
            k: {
                "yprov4wfs:nice_name": step,
                "prov:startTime": start,
                "prov:endTime": end,
                "yprov4wfs:status": status,
                "yprov4wfs:processed_on": worker,
                "yprov4wfs:jupyter_cell_index": "3" if step != "save" else "4",
                **({"yprov4wfs:carbon_footprint_kg": carbon} if carbon is not None else {}),
            }
            for k, (step, start, end, status, worker, carbon) in act.items()
        },
        "entity": {"out": {"yprov4wfs:nbytes": "2048"}},
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    store = ProvStore()
    return store, store.ingest_file(path).doc_id


def test_overview_of_a_workflow_run(tmp_path):
    store, doc_id = _run(tmp_path)
    d = T.describe_document(store, doc_id)
    w = d["workflow"]
    assert d["kind"] == "yprov4wfs" and w["tasks"] == 4
    assert w["wall_clock_seconds"] == 8.0 and w["summed_task_seconds"] == 11.0
    assert w["slowest_task"]["seconds"] == 4.0
    assert w["slowest_task"]["steps_with_this_duration"] == ["load", "mean"]
    assert w["failed_tasks"] == 1 and w["failed"][0]["step"] == "save"
    assert w["workers"]["count"] == 2
    assert w["notebook_cells"][0] == {"cell": "3", "line": "", "tasks": 3}
    carbon = w["measures"]["yprov4wfs:carbon_footprint_kg"]
    # Small values keep their digits; "None" is not a value.
    assert carbon["reported_by"] == 2 and carbon["sum"] == 4e-07 and carbon["max_step"] == "mean"
    assert d["largest_size"] == {
        "bytes": 2048,
        "entities_with_this_size": 1,
        "examples": ["out"],
        "attribute": "yprov4wfs:nbytes",
    }


def test_activity_stats_groups_filters_and_sorts(tmp_path):
    store, doc_id = _run(tmp_path)
    by_time = T.activity_stats(store, doc_id, sort_by="time")
    assert [g["group"] for g in by_time["groups"]] == ["mean", "load", "save"]
    assert by_time["groups"][0]["summed_seconds"] == 5.0
    assert by_time["totals"]["tasks"] == 4 and by_time["totals"]["failed"] == 1

    failed = T.activity_stats(store, doc_id, status="failed", group_by="none")
    assert failed["totals"]["tasks"] == 1 and failed["groups"][0]["group"] == "all tasks"

    carbon = T.activity_stats(
        store,
        doc_id,
        group_by="worker",
        measure="yprov4wfs:carbon_footprint_kg",
        sort_by="measure",
    )
    assert carbon["groups"][0]["group"] == "w2"
    assert carbon["totals"]["yprov4wfs:carbon_footprint_kg"]["reported_by"] == 2

    cells = T.activity_stats(store, doc_id, group_by="cell")
    assert {g["group"]: g["tasks"] for g in cells["groups"]} == {"3": 3, "4": 1}
    assert T.activity_stats(store, doc_id, step="mea")["totals"]["tasks"] == 2


def test_activity_stats_says_what_exists(tmp_path):
    store, doc_id = _run(tmp_path)
    with pytest.raises(T.ToolError, match="numeric attributes are: yprov4wfs:carbon"):
        T.activity_stats(store, doc_id, measure="ram")
    with pytest.raises(T.ToolError, match="no task matches"):
        T.activity_stats(store, doc_id, step="nothing")
    with pytest.raises(T.ToolError, match="no task has the attribute"):
        T.activity_stats(store, doc_id, group_by="colour")
    with pytest.raises(T.ToolError, match="sort_by"):
        T.activity_stats(store, doc_id, sort_by="size")
    with pytest.raises(T.ToolError, match="no task records a module; group_by='step'"):
        T.activity_stats(store, doc_id, group_by="module")


@pytest.mark.skipif(not DATA.is_dir(), reason="data/ is not present")
def test_the_tools_agree_with_the_independent_gold_on_every_file():
    from eval import gold_data as G

    store = ProvStore()
    for path in sorted(G.DATA.glob("*.json")):
        doc = G.load(path.name)
        doc_id = store.ingest_file(path).doc_id
        if not G.timed(doc):
            continue
        d = T.describe_document(store, doc_id)
        w = d["workflow"]
        span, slow = G.span(doc), G.slowest(doc)
        assert w["tasks"] == G.counts(doc)["activity"], path.name
        assert w["wall_clock_seconds"] == pytest.approx(span["seconds"], abs=1e-3)
        assert w["summed_task_seconds"] == pytest.approx(
            sum(G.time_by_step(doc).values()), rel=1e-6
        )
        assert w["slowest_task"]["seconds"] == pytest.approx(slow["seconds"], rel=1e-6)
        assert w["slowest_task"]["steps_with_this_duration"] == slow["tied_steps"], path.name
        # The fastest single task and its ties, from the raw times.
        durations = {(e - st).total_seconds() for _, _, st, e in G.timed(doc)}
        quickest = min(durations)
        assert w["fastest_task"]["seconds"] == pytest.approx(quickest, abs=1e-6), path.name
        assert w["fastest_task"]["steps_with_this_duration"] == sorted(
            {
                G.step_name(i, a)
                for i, a, st, e in G.timed(doc)
                if (e - st).total_seconds() == quickest
            }
        ), path.name
        assert w["status_counts"] == G.status_counts(doc) and w["failed_tasks"] == len(
            G.failed(doc)
        )
        assert w["steps"][0]["step"] in G.top_ties(G.steps(doc)), path.name
        by_time = T.activity_stats(store, doc_id, sort_by="time")["groups"][0]
        assert by_time["group"] in G.top_ties(G.time_by_step(doc)), path.name
        if G.hosts(doc):
            assert w["workers"]["count"] == len(G.hosts(doc))
        if G.cells(doc):
            idx, cell = next(iter(G.cells(doc).items()))
            assert (w["notebook_cells"][0]["cell"], w["notebook_cells"][0]["tasks"]) == (
                idx,
                cell["tasks"],
            )
        if (big := G.largest_entity(doc)) is not None:
            assert d["largest_size"]["bytes"] == big["nbytes"]
            sizes = [G._num(e.get("yprov4wfs:nbytes")) for e in G.nodes(doc, "entity").values()]
            shared = sizes.count(big["nbytes"])
            assert d["largest_size"]["entities_with_this_size"] == shared, path.name
        if G.producer(doc) == "openeo":
            res, m = G.resources(doc), w["measures"]
            carbon = m["yprov4wfs:carbon_footprint_kg"]
            assert carbon["sum"] == pytest.approx(res["carbon_footprint_kg"]["sum"], rel=1e-5)
            assert carbon["reported_by"] == res["carbon_footprint_kg"]["n"]
            ram = m["yprov4wfs:ram_usage_mb"]
            assert (ram["max"], ram["max_step"]) == (
                pytest.approx(res["ram_usage_mb"]["max"]),
                res["ram_usage_mb"]["max_step"],
            )


def test_run_level_facts_are_surfaced(tmp_path):
    """Facts on every task (a domain, a variable in the task group) and on a workflow-level
    activity without times (an engine) reach the overview; the peak of a measure sorts by
    its largest value, not its sum."""
    activity = {
        "wfs1": {"prov:startTime": "None", "yprov4wfs:engine": "Local", "yprov4wfs:status": "Ok"}
    }
    for i, (step, ram) in enumerate([("a", 5.0), ("a", 5.0), ("b", 9.0)]):
        activity[f"t{i}"] = {
            "prov:startTime": f"2025-01-01 10:00:0{i}",
            "prov:endTime": f"2025-01-01 10:00:0{i + 1}",
            "yprov4wfs:nice_name": step,
            "yprov4wfs:group": "open_dataset-tas-" + "0" * 32,
            "yprov4wfs:ram_usage_mb": ram,
            "yprov4wfs:status": "success",
            "metadata": {"domain": "Climate", "semantic_summary": f"task {i}"},
        }
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"activity": activity}), encoding="utf-8")
    store = ProvStore()
    doc_id = store.ingest_file(path).doc_id
    w = T.describe_document(store, doc_id)["workflow"]
    assert w["shared_attributes"]["metadata.domain"] == "Climate"
    assert "metadata.semantic_summary" not in w["shared_attributes"]
    assert w["task_groups"] == {"open_dataset-tas": 3}
    assert w["variables_opened"] == ["tas"]
    assert w["workflow_activities"][0]["attributes"]["yprov4wfs:engine"] == "Local"
    peak = T.activity_stats(store, doc_id, measure="yprov4wfs:ram_usage_mb", sort_by="measure_max")
    assert peak["groups"][0]["group"] == "b"
    summed = T.activity_stats(store, doc_id, measure="yprov4wfs:ram_usage_mb", sort_by="measure")
    assert summed["groups"][0]["group"] == "a"
    assert (summed["peak"]["value"], summed["peak"]["step"]) == (9.0, "b")
    assert T.activity_stats(store, doc_id, group_by="metadata.domain")["groups"][0]["tasks"] == 3


def test_steps_by_time(tmp_path):
    store, doc_id = _run(tmp_path)
    w = T.describe_document(store, doc_id)["workflow"]
    assert [s["step"] for s in w["steps_by_time"]] == ["mean", "load", "save"]
    assert w["steps_by_time"][0]["summed_seconds"] == 5.0


def test_inputs_outputs_and_pipeline_sizes(tmp_path):
    """data_flow: entities used and never generated are the inputs, generated and never used
    the outputs; a reduced pipeline says how many tasks it aggregates."""
    doc = {
        "activity": {
            "load": {
                "prov:startTime": "2025-01-01 10:00:00",
                "prov:endTime": "2025-01-01 10:00:01",
            },
            "pipe": {
                "prov:type": "reduced_pipeline",
                "analytics4yprov:aggregates": ["a", "b", "c"],
            },
        },
        "entity": {
            "in.nc": {"prov:label": "SIG0_x.nc"},
            "mid": {"prov:label": "<class 'xarray.core.dataarray.DataArray'>"},
        },
        "used": {"u1": {"prov:activity": "load", "prov:entity": "in.nc"}},
        "wasGeneratedBy": {"g1": {"prov:entity": "mid", "prov:activity": "load"}},
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    store = ProvStore()
    d = T.describe_document(store, store.ingest_file(path).doc_id)
    assert d["data_flow"] == {
        "inputs": {"count": 1, "names": ["SIG0_x.nc"]},
        "final_outputs": {"count": 1, "names": ["mid"]},  # a class name is not a name
    }
    pipe = next(a for a in d["activities"] if a["node_id"] == "pipe")
    assert pipe["list_sizes"] == {"analytics4yprov:aggregates": 3}
    stats = T.activity_stats(store, d["doc_id"], step="pipe")
    assert stats["groups"][0]["tasks"] == 1
    assert stats["groups"][0]["list_sizes"] == {"analytics4yprov:aggregates": 3}


def test_a_machine_learning_run_is_not_read_as_a_workflow(store):
    """yProv4ML: one timed activity (the run itself) and no status. No workflow overview,
    no "slowest step", no "0 failed": the run's duration, and what is not recorded."""
    doc_id = next(r["doc_id"] for r in store.documents() if "GR0_0" in r["doc_id"])
    d = T.describe_document(store, doc_id)
    assert "workflow" not in d
    assert d["run"]["name"] == "example_GR0_0" and d["run"]["seconds"] == pytest.approx(
        7.81, abs=0.01
    )
    assert d["timing_coverage"].startswith("0 of 8 steps")
    assert "not recorded" in d["status_coverage"]
    # The samples carry the metric CSVs: the losses' values are given, not the 17 counters.
    assert set(d["metric_values"]) == {"MSELoss/MSELoss/Training", "MSELoss/MSELoss/Validation"}
    stats = T.activity_stats(store, doc_id, sort_by="max_time")
    assert "not recorded" in stats["status_coverage"] and stats["timing_coverage"]
    # The run is not one of its steps: no group is it, and none has a duration.
    assert stats["run"]["name"] == "example_GR0_0" and stats["groups_total"] == 8
    assert not any("slowest" in g for g in stats["groups"])
    assert "run" not in T.activity_stats(store, doc_id, step="Training")
    failed = T.activity_stats(store, doc_id, status="failed")
    assert failed["matching_tasks"] is None and "not recorded" in failed["status_coverage"]
    assert "yprov:run_id" not in (T.activity_stats(store, doc_id).get("totals") or {})


def test_an_uploaded_ml_file_says_where_its_metric_values_are(tmp_path):
    src = SAMPLES / "yprov4ml" / "example_0" / "prov_example_GR0_0.json"
    alone = tmp_path / src.name
    alone.write_bytes(src.read_bytes())
    s = ProvStore()
    d = T.describe_document(s, s.ingest_file(alone).doc_id)
    assert "metric_values" not in d
    assert "separate CSV files" in d["metric_values_note"]


def test_agent_records_give_their_decision_and_errors_say_what_exists(store):
    doc_id = next(
        r["doc_id"] for r in store.documents() if r["doc_id"].startswith("example_0__cax_prov")
    )
    d = T.describe_document(store, doc_id)
    (dec,) = d["decision_details"]
    assert (dec["agent"], dec["verdict"], dec["rule"]) == (
        "NL_Summarizer",
        "ACCEPT",
        "cax.summary/v1",
    )
    with pytest.raises(T.ToolError, match=r"the decisions here are.*NL_Summarizer"):
        T.list_decisions(store, doc_id=doc_id, agent="CA1")
    assert "overview" in T.search_nodes(store, "example_0__cax", doc_id=doc_id)["note"]
    some = T.search_nodes(store, "summary checker", doc_id=doc_id)
    assert some["matches"] and "some of them" in some["note"]


def test_small_records_are_read_whole_and_identifiers_surface(store):
    """An agent record's storage PID sits on an entity: the overview gives it, the file's
    name (never its path on the server), and no "uploads" collection."""
    doc_id = next(r["doc_id"] for r in store.documents() if r["doc_id"].startswith("ca1-timeout"))
    d = T.describe_document(store, doc_id)
    assert d["file"] == "ca1-timeout__ca1_prov.json"
    assert {"attribute": "yp:pid", "value": "21.T11999/yprovstore-abfe4946dd38"}.items() <= (
        d["identifiers"][0].items()
    )
    assert len(d["entity_details"]) == 4
    assert "collection" not in d["decision_details"][0]


def test_the_fastest_task_comes_first_with_its_ties(tmp_path):
    store, doc_id = _run(tmp_path)
    quick = T.activity_stats(store, doc_id, sort_by="min_time")
    assert list(quick)[:2] == ["group_by", "fastest"]
    assert quick["fastest"]["seconds"] == 1.0 and quick["groups"][0]["group"] == "mean"


def _ophidia(tmp_path) -> tuple[ProvStore, str]:
    """Two operators whose period sits inside an argument string, as Fires_0 writes it."""
    args = "imp_dim=time,container=fires,subset_filter=2090-01-01_2090-01-15,cwd=/"
    doc = {
        "prefix": {"ophidia": "http://example.org"},
        "activity": {
            f"ophidia:Import_variable_({i})": {
                "prov:startTime": "2025-07-14T15:46:33",
                "prov:endTime": "2025-07-14T15:46:39",
                "ophidia:status": "OPH_STATUS_COMPLETED",
                "ophidia:arguments": args,
                "analytics4yprov:level": i % 2,
            }
            for i in (1, 2)
        },
    }
    path = tmp_path / "fires.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    store = ProvStore()
    return store, store.ingest_file(path).doc_id


def test_argument_strings_are_split_and_every_attribute_is_catalogued(tmp_path):
    store, doc_id = _ophidia(tmp_path)
    fields = T.fields({"a": "x=1,y=2", "b": "not=an argument list"})
    assert fields["a.x"] == "1" and fields["a.y"] == "2" and "b.not" not in fields
    attributes = T.describe_document(store, doc_id)["attributes"]
    assert attributes["ophidia:arguments.subset_filter"] == {
        "on": "activity",
        "nodes": 2,
        "values": {"2090-01-01_2090-01-15": 2},
    }
    assert attributes["analytics4yprov:level"]["values"] == {"1": 1, "0": 1}


def test_search_reaches_the_files_words_and_says_how(tmp_path):
    store, doc_id = _ophidia(tmp_path)
    found = T.search_nodes(store, "date range", doc_id=doc_id)
    assert found["total"] == 2
    hit = found["matches"][0]
    assert hit["matched"] == "date, as subset_filter; range, as subset_filter"
    assert hit["fields"] == {"ophidia:arguments.subset_filter": "2090-01-01_2090-01-15"}
    # A word the file writes itself needs no explanation.
    assert "matched" not in T.search_nodes(store, "container", doc_id=doc_id)["matches"][0]
