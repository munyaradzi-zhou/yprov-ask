"""The independent gold computations and the question bank over data/, offline.

Skipped when data/ is absent (it is not part of the repository)."""

from __future__ import annotations

import json

import pytest

from eval import gold_data as G
from eval import questions_data as Q

pytestmark = pytest.mark.skipif(not G.DATA.is_dir(), reason="data/ is not present")


def test_producers_are_recognised():
    assert G.producer(G.load("climatology_anomaly_jt.json")) == "yprov4wfs"
    assert G.producer(G.load("yprov4openEO.json")) == "openeo"
    assert G.producer(G.load("Fires_0.json")) == "ophidia"
    assert G.producer(G.load("climatology_detail_red.json")) == "analytics"


def test_repeated_ids_are_merged():
    fires = G.load("Fires_0.json")
    assert isinstance(next(iter(fires["entity"].values())), list)
    assert all(isinstance(v, dict) for v in G.nodes(fires, "entity").values())


def test_facts_checked_by_hand():
    # Each value was also obtained by separate code over the raw JSON (the
    # profiling scripts of that session, and one-line checks of the span and the workers).
    openeo = G.load("yprov4openEO.json")
    assert G.status_counts(openeo) == {"Ok": 81, "Failed": 7}
    assert sorted(G.failed(openeo)) == ["merge_cubes"] * 4 + ["multiply"] + ["reduce_dimension"] * 2
    assert G.resources(openeo)["carbon_footprint_kg"]["n"] == 27
    ml = G.load("machine_learning_jt.json")
    assert G.counts(ml)["activity"] == 1164
    assert G.cells(ml)["12"]["tasks"] == 940
    clim = G.load("climatology_anomaly_jt.json")
    assert G.span(clim)["seconds"] == pytest.approx(23.996, abs=0.001)
    assert len(G.hosts(clim)) == 8
    fires = G.ophidia_runs(G.load("Fires_0.json"))
    assert fires["models"] == ["CMCC-ESM2"] and fires["scenarios"] == ["ssp126"]


def test_the_question_bank_is_deterministic_and_complete():
    first, second = Q.build(), Q.build()
    assert json.dumps(first, default=str) == json.dumps(second, default=str)
    ids = [i["id"] for i in first]
    assert len(ids) == len(set(ids))
    for item in first:
        assert len(item["phrasings"]) == 3 and item["gold"] and item["check"]["type"]
        assert item["split"] in ("dev", "used", "test")
        assert item["expect"] in ("ACCEPT", "ABSENT", "OUT_OF_SCOPE")
    # The bank file on disk is the one this code builds: nobody edited the frozen set by hand.
    on_disk = json.loads(Q.OUT.read_text(encoding="utf-8"))["items"]
    assert json.dumps(on_disk, default=str) == json.dumps(first, default=str)


def test_data_v3_keeps_the_v2_items_and_adds_a_new_test_split():
    """The v2 items are unchanged but for their split ("test" became "used") and two more
    "no failure" phrases; every item of the new test split is new."""
    v2_text = (Q.OUT.parent / "data_v2.json").read_text(encoding="utf-8")
    v2 = {i["id"]: i for i in json.loads(v2_text)["items"]}
    v3 = {i["id"]: i for i in Q.build()}
    for item_id, old in v2.items():
        new = v3[item_id]
        assert new["split"] == ("dev" if old["split"] == "dev" else "used")
        assert new["phrasings"] == old["phrasings"] and new["gold"] == old["gold"]
        if new["check"] != old["check"]:
            assert set(old["check"]["terms"]) < set(new["check"]["terms"]), item_id
    assert all(v3[i]["split"] == "test" for i in set(v3) - set(v2))
    assert not any(v3[i]["split"] == "test" for i in v2)


def test_the_samples_bank_is_deterministic_and_accepts_its_own_gold():
    from eval import questions_samples as S
    from eval.run_data import check_for

    first = S.build()
    assert json.dumps(first, default=str) == json.dumps(S.build(), default=str)
    on_disk = json.loads(S.OUT.read_text(encoding="utf-8"))
    assert on_disk["data_dir"] == "samples"
    assert json.dumps(on_disk["items"], default=str) == json.dumps(first, default=str)
    for item in first:
        assert len(item["phrasings"]) == 3 and item["split"] in ("dev", "test")
        if item["expect"] == "ACCEPT":
            assert check_for(item["check"])(item["gold"], "ACCEPT"), item["id"]
    # The two questions seen in the session that prompted the bank are tuning questions.
    assert {i["id"]: i["split"] for i in first if i["id"] in S.SEEN} == dict.fromkeys(S.SEEN, "dev")
