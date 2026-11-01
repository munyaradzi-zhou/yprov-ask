from __future__ import annotations

from yprov_ask import grounding as G


def test_rounded_figures_are_supported():
    facts = G.facts_from([{"min": 0.09629829227924347, "bytes": 21634}])
    result = G.check("The minimum was 0.0963 and the file 21,634 bytes.", facts, tool_calls=2)
    assert result.verdict == "ACCEPT"
    assert result.figures_checked == 2


def test_a_wrongly_rounded_figure_is_not():
    facts = G.facts_from([0.09629829227924347])
    result = G.check("The minimum was 0.097.", facts, tool_calls=1)
    assert result.verdict == "REVISE"
    assert result.ungrounded == ["0.097"]


def test_small_integers_and_identifiers_are_not_claims():
    result = G.check(
        "Run example_GR0_0 has 3 phases; see MSELoss_Training_GR0 and Python 3.13.11.",
        [],
        tool_calls=1,
    )
    assert result.figures_checked == 0
    assert result.verdict == "ACCEPT"


def test_numbers_inside_tool_strings_count_as_facts():
    facts = G.facts_from(["Record carries 9 activities / 39 entities, 21634 bytes."])
    assert G.check("It was 21634 bytes.", facts, tool_calls=1).verdict == "ACCEPT"


def test_figures_without_any_tool_call_are_not_accepted():
    result = G.check("The loss was 0.25.", G.facts_from(["0.25"]), tool_calls=0)
    assert result.verdict == "REVISE"


def test_exponent_form_is_compared_at_its_precision():
    facts = G.facts_from([0.000123456])
    assert G.check("about 1.23e-4", facts, tool_calls=1).verdict == "ACCEPT"
    assert G.check("about 1.30e-4", facts, tool_calls=1).verdict == "REVISE"


def test_the_recorded_verdict_replays():
    result = G.check("The loss was 0.5.", [0.25], tool_calls=1)
    inputs = {k: result.evidence[k] for k in sorted(result.evidence)}
    assert G.replay("qa.grounding/v2", inputs) == result.verdict == "REVISE"
    # A record written under v1 still replays with v1.
    v1 = {"figures_checked": 0, "ungrounded_figures": 0, "tool_calls": 0}
    assert G.replay("qa.grounding/v1", v1) == "ACCEPT"


def test_timestamp_parts_written_in_prose_are_supported():
    facts = G.facts_from([{"start": "2026-05-20T10:10:05.005879"}])
    result = G.check("It started on May 20, 2026 at 10:10:05.005879.", facts, tool_calls=1)
    assert result.verdict == "ACCEPT", result.ungrounded


def test_v2_accepts_nothing_that_rests_on_no_lookup():
    # v1 accepted this: no figures, so nothing to check.
    result = G.check("Real Madrid and Barcelona are the best teams.", [], tool_calls=0)
    assert result.verdict == "REVISE"
    assert G.reasons(result.evidence, result.ungrounded) == ["nothing was looked up in the records"]
    # Failed lookups are not lookups.
    failed = G.check("No such document.", [], tool_calls=2, data_lookups=0)
    assert failed.verdict == "REVISE"


def test_a_declined_question_is_out_of_scope():
    result = G.check("The records don't cover this.", [], tool_calls=1, declined=True)
    assert result.verdict == "OUT_OF_SCOPE"
    assert G.replay("qa.grounding/v2", result.evidence) == "OUT_OF_SCOPE"


def test_an_answer_repeated_for_a_different_question_is_sent_back():
    earlier = [
        ("Which decisions were escalated?", "Three decisions in ca1-timeout were escalated.")
    ]
    same = G.repeat_similarity(
        "Three decisions in ca1-timeout were escalated.", "Who was Italy's prime minister?", earlier
    )
    assert same == 1.0
    result = G.check("x", [], tool_calls=1, repeat_similarity=same)
    assert result.verdict == "REVISE"
    assert "repeats an earlier answer" in G.reasons(result.evidence, [])[0]
    # The same question asked again may get the same answer.
    again = G.repeat_similarity(
        "Three decisions in ca1-timeout were escalated.", "Which decisions were escalated?", earlier
    )
    assert again == 0.0
    # A related follow-up shares words but is not a repeat.
    follow_up = G.repeat_similarity(
        "ca1-timeout escalated because no consumer acknowledged within 3.0 s.",
        "Why was it escalated?",
        earlier,
    )
    assert follow_up < G.REPEAT_THRESHOLD
    # The same sentence about another run carries new information: not a repeat, even when
    # the wording is almost identical (seen in the evaluation).
    runs = [
        (
            "Which optimizer did example_GR0_2 use, and with which learning rate and batch size?",
            "The run example_GR0_2 used the Adam optimizer with learning rate 0.001, batch size"
            " 64 and 10 epochs, as recorded in its parameters (prov_example_GR0_2).",
        )
    ]
    other_run = (
        "The run example_GR0_0 used the Adam optimizer with learning rate 0.001, batch size"
        " 64 and 10 epochs, as recorded in its parameters (prov_example_GR0_0)."
    )
    assert G.similarity(other_run, runs[0][1]) >= G.REPEAT_THRESHOLD
    assert G.repeat_similarity(other_run, "And example_GR0_0?", runs) == 0.0


def test_ordered_list_numbers_are_not_figures():
    answer = "".join(f"{i}. **doc_{i}**: 21634 bytes\n" for i in range(1, 13))
    result = G.check(answer, G.facts_from(["21634"]), tool_calls=1)
    assert result.verdict == "ACCEPT", result.ungrounded
    assert result.figures_checked == 12  # the byte counts, not the item numbers
    # A number at the start of a line that is not a list marker is still a figure.
    assert G.check("12 runs failed.", [], tool_calls=1).ungrounded == ["12"]


def test_a_clock_time_with_whole_seconds_is_supported():
    """The tool returns 16:31:55.628597; an answer writes 16:31:55. The 55 is the same fact,
    not an invented figure (a correct answer was sent back for it)."""
    from yprov_ask.grounding import check, facts_from

    facts = facts_from([{"first_start": "2025-10-03 16:31:55.628597"}])
    result = check("It started on October 3, 2025, at 16:31:55.", facts, tool_calls=1)
    assert result.ungrounded == [] and result.verdict == "ACCEPT"


def test_a_real_figure_in_the_wrong_role_is_sent_back():
    """v3: the busiest worker's "4 tasks" beside a worker with 5 (frozen v3 test); the
    whole run's time given as the slowest step's. Right answers of the same shape pass."""
    stats = {
        "groups": [
            {"group": "tcp://127.0.0.1:35979/2", "tasks": 5, "summed_seconds": 0.31},
            {"group": "tcp://127.0.0.1:35753/1", "tasks": 4, "summed_seconds": 0.255},
        ]
    }
    labelled = G.labelled_facts([stats])
    wrong = "The busiest worker is `tcp://127.0.0.1:35753/1`, which handled 4 tasks."
    flagged = G.misplaced(wrong, labelled)
    assert flagged == [
        {
            "figure": "4",
            "said_as": "the largest",
            "returned_as": ["groups.tasks"],
            "but": "5 is also returned as groups.tasks",
        }
    ]
    right = "The busiest worker is `tcp://127.0.0.1:35979/2`, which handled 5 tasks."
    assert G.misplaced(right, labelled) == []
    assert G.misplaced("The next most used worker ran 4 tasks.", labelled) == []

    overview = {"run": {"seconds": 7.809871}, "workflow": {"summed_task_seconds": 35.0}}
    labelled = G.labelled_facts([overview])
    assert G.misplaced("The slowest step took 7.81 seconds.", labelled)[0]["said_as"] == (
        "one step or task"
    )
    assert G.misplaced("The run took 7.81 s, from its first task to its last task.", labelled) == []
    # Each figure is read by the words before it, not by its neighbour's.
    assert G.misplaced("Summed 35.0 s, with a wall clock time of 7.81 s.", labelled) == []

    result = G.check(wrong, G.facts_from([stats]), 1, labelled=G.labelled_facts([stats]))
    assert result.verdict == "REVISE" and result.evidence["misplaced_figures"] == 1
    assert G.replay("qa.grounding/v3", result.evidence) == "REVISE"
    assert G.replay("qa.grounding/v2", result.evidence) == "ACCEPT"
    assert "4 is given as the largest" in G.reasons(result.evidence, [], result.misplaced)[0]


def test_numbers_written_as_words_are_figures():
    from yprov_ask.grounding import check, spelled_as_digits

    assert spelled_as_digits("thirty-six tasks and two hundred and five rows") == (
        "36 tasks and 205 rows"
    )
    assert spelled_as_digits("the most common one") == "the most common one"
    assert check("It ran thirty-six tasks.", [36.0], 1).verdict == "ACCEPT"
    wrong = check("It ran forty tasks.", [36.0], 1)
    assert wrong.verdict == "REVISE" and wrong.ungrounded == ["40"]


def test_a_superlative_count_written_as_a_word_is_checked():
    from yprov_ask.grounding import check, labelled_facts

    returned = [{"values": [{"value": "TaskRef", "nodes": 30}, {"value": "list", "nodes": 3}]}]
    labelled = labelled_facts(returned)
    facts = [f for f, _ in labelled]
    said = "The most common dtype is list, which appears in three entities."
    assert check(said, facts, 1, labelled=labelled).verdict == "REVISE"
    right = "The most common dtype is TaskRef, which appears in thirty entities."
    assert check(right, facts, 1, labelled=labelled).verdict == "ACCEPT"


def test_a_small_count_of_something_is_checked():
    from yprov_ask.grounding import check, facts_from

    facts = facts_from([{"tasks": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}])
    assert check("It ran 3 move_mean tasks.", facts, 1).verdict == "ACCEPT"
    wrong = check("It ran 4 move_mean tasks.", facts, 1)
    assert wrong.verdict == "REVISE" and wrong.ungrounded == ["4"]
    assert check("It ran four tasks.", facts, 1).verdict == "REVISE"


def test_small_numbers_that_count_nothing_stay_prose_and_are_counted():
    from yprov_ask.grounding import check

    result = check("In 2 words: step 3 was slow.", [100.0], 1)
    assert result.verdict == "ACCEPT"
    assert result.evidence["small_numbers_unchecked"] == 2
