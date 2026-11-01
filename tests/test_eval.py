"""The evaluation harness's own logic: case construction, checks and statistics, offline."""

from __future__ import annotations

from eval.run_eval import absent, dialogues, out_of_scope, summarise, wilson
from yprov_ask.agent import DECLINE_REPLY


def test_wilson_interval_matches_known_values():
    assert wilson(0, 0) is None
    low, high = wilson(15, 16)
    assert (low, high) == (0.717, 0.989)
    assert wilson(10, 10)[1] == 1.0 and wilson(0, 10)[0] == 0.0


def test_cases_are_built_from_the_store(store):
    built = dialogues(store)
    ids = [d.id for d in built]
    assert len(ids) == len(set(ids))
    categories = {d.category for d in built}
    assert categories == {"single", "out-of-scope", "dialogue"}
    reported = next(d for d in built if d.id == "dialogue/reported-session")
    assert [t.id for t in reported.turns] == [
        "documents-loaded",
        "escalated-why",
        "pizza",
        "la-liga",
        "messi",
        "italy-pm",
    ]
    assert all(t.expect == "OUT_OF_SCOPE" for t in reported.turns[2:])


def test_checks_score_verdicts_and_text():
    assert out_of_scope()(DECLINE_REPLY, "OUT_OF_SCOPE")
    assert not out_of_scope()("Real Madrid.", "ACCEPT")
    gone = absent("test accuracy")
    assert gone("Test accuracy is not recorded.", "ACCEPT")
    assert gone("No test accuracy is recorded; the validation loss was 0.031.", "ACCEPT")
    assert not gone("The test accuracy was 0.98.", "ACCEPT")
    assert not gone("It does not say, but test accuracy reached 98%.", "ACCEPT")
    # Digits inside an identifier are not a value (a mis-score in an evaluation run).
    assert gone("The test accuracy for example_GR0_0 is not provided; MSELoss was 0.031.", "ACCEPT")


def test_summary_separates_accepted_but_wrong():
    def row(expect, verdict, correct, turn=1):
        return {
            "expect": expect,
            "verdict": verdict,
            "correct": correct,
            "turn": turn,
            "repeat_similarity": 0.0,
            "seconds": 1.0,
            "data_lookups": 1,
            "prompt_tokens": 1000,
            "completion_tokens": 100,
        }

    s = summarise(
        [
            row("ACCEPT", "ACCEPT", True),
            row("ACCEPT", "ACCEPT", False, turn=2),
            row("OUT_OF_SCOPE", "OUT_OF_SCOPE", True),
            row("OUT_OF_SCOPE", "ACCEPT", False),
        ]
    )
    assert s["in_scope_correct"]["k"] == 1 and s["follow_up_correct"]["n"] == 1
    assert s["out_of_scope_declined"]["rate"] == 0.5
    assert s["accepted_but_wrong"] == {"k": 2, "n": 3, "rate": 0.667, "ci95": wilson(2, 3)}
    assert s["verdicts_by_expected"] == {
        "ACCEPT": {"ACCEPT": 2},
        "OUT_OF_SCOPE": {"OUT_OF_SCOPE": 1, "ACCEPT": 1},
    }


def test_terms_match_whole_words_and_the_bank_accepts_its_own_gold():
    import json

    from eval.run_data import BANK, check_for, has_all

    # "tas" is inside "dataset": a substring match scored "not recorded" as correct.
    assert not has_all("tas")("the dataset variable is not recorded", "ACCEPT")
    assert has_all("tas")("it read open_dataset-tas", "ACCEPT")
    for item in json.loads(BANK.read_text(encoding="utf-8"))["items"]:
        if item["check"]["type"] not in ("verdict", "absent"):
            assert check_for(item["check"])(item["gold"], "ACCEPT"), item["id"]


def test_scorer_v2_rules():
    from eval.run_data import check_for

    near = check_for({"type": "figure_and_any", "value": 0.785, "rel": 0.02, "terms": ["getter"]})
    assert near("The slowest task was getter: 785 ms.", "ACCEPT")  # another unit
    assert not near("getter ran." + " x" * 200 + " Another task took 0.785 s.", "ACCEPT")
    size = check_for({"type": "figure", "value": 86787072, "rel": 0.01})
    assert size("The largest object is 86.8 MB.", "ACCEPT")
    terms = check_for({"type": "all_terms", "terms": ["tas"]})
    assert not terms(
        "The variable is not explicitly named; the step is open_dataset-tas.", "ACCEPT"
    )
    total = check_for(
        {"type": "figure_and_terms", "value": 7, "rel": 0, "terms": ["merge_cubes", "multiply"]}
    )
    assert total(
        "1. merge_cubes" + " x" * 200 + "\n2. multiply\n\n7 failed tasks in all.", "ACCEPT"
    )
    # v5: a remark after the answer about another fact is not a hedge.
    count = check_for({"type": "figure", "value": 19, "rel": 0})
    assert count("It logged 19 metrics. Their values are not included in this file.", "ACCEPT")
    assert not count("The count is not recorded; 19 metric entities exist.", "ACCEPT")
    # v3: a partial fact is not a hedge.
    coverage = check_for({"type": "figure", "value": 27, "rel": 0})
    assert coverage("The carbon footprint is not recorded for every process: 27 of 88.", "ACCEPT")
    assert not coverage("The carbon footprint is not recorded; 27 tasks ran.", "ACCEPT")


def test_not_recorded_is_right_only_when_the_answer_says_so():
    from eval.run_data import check_for

    nr = check_for({"type": "not_recorded"})
    assert not nr("The step that took the longest is example_GR0_0, 7.81 seconds.", "ACCEPT")
    assert not nr("No, there were 0 failed tasks.", "ACCEPT")
    assert nr("Step durations are not recorded; only a start time. The run took 7.81 s.", "ACCEPT")
    assert nr("No activity records a status, so failures cannot be determined.", "ACCEPT")
    assert nr("The loss values are in a separate CSV file.", "ACCEPT")
    assert nr("The records do not cover whether any tasks failed.", "ACCEPT")
    assert nr("anything", "OUT_OF_SCOPE")


def test_number_words_count_as_figures():
    from eval.run_data import check_for

    two = check_for({"type": "figure", "value": 2, "rel": 0})
    assert two("Two distinct executors processed the tasks.", "ACCEPT")
    assert not two("Three executors.", "ACCEPT")


def test_audit_sampling_and_agreement():
    from eval.audit import agreement, distinct, kappa, sample

    rows = [
        {
            "id": f"q{i}",
            "phrasing": 0,
            "answer": f"a{i}",
            "expect": "ACCEPT",
            "correct": i % 10 != 0,
        }
        for i in range(200)
    ]
    rows.append(dict(rows[1]))  # the same answer again: scored once
    rows.append({**rows[2], "id": "off", "expect": "OUT_OF_SCOPE"})
    kept = distinct(rows)
    assert len(kept) == 200
    drawn = sample(kept)
    assert len(drawn["rows"]) == 100
    assert sum(1 for _, auto in drawn["rows"] if not auto) == 20  # every failed row
    assert drawn["population"] == {"auto_correct": 180, "auto_wrong": 20}

    assert kappa([(True, True), (False, False)]) == 1.0
    scored = [{"auto": True, "human": True, "extra": False}] * 9 + [
        {"auto": True, "human": False, "extra": True},
        {"auto": False, "human": False, "extra": False},
    ]
    a = agreement(scored, {"auto_correct": 90, "auto_wrong": 10})
    assert a["scorer_passed_but_person_says_wrong"]["k"] == 1
    assert a["agreement_weighted_to_run"] == 0.91
    assert a["answers_with_an_extra_wrong_claim"] == 1


def test_evaluations_refuse_models_other_than_gpt_4o_mini():
    import pytest

    from eval.run_eval import allowed_model

    allowed_model(None)
    allowed_model("gpt-4o-mini-2024-07-18")
    with pytest.raises(SystemExit):
        allowed_model("gpt-4o-2024-08-06")
