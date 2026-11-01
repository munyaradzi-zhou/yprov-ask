"""A person checks the automatic scorer: a sample of answers to score by hand, then agreement.

The automatic scorer (``eval/run_data.py``) is an instrument, and its errors go into every
reported rate. A person scores a sample of answers, without seeing the automatic score;
this module draws the sample and measures how often the two disagree.

    python -m eval.audit make eval/results/<run>/results.json      # writes audit/ beside it
    python -m eval.audit score eval/results/<run>/audit/sheet.csv  # after filling it in

Sampling: one row per distinct (question, answer), so repeats of the same answer are not
scored twice. Rows the scorer failed are rare and are where it errs most often, so all of
them are taken (up to half the sample) and the rest are drawn at random from the rows it
passed. The two groups are weighted by their size in the run when the overall agreement is
estimated. Out-of-scope rows are left out: their check is the verdict, not a judgement.

The person fills two columns per row:
    correct   Y (the answer is right), N (wrong), P (partly right: counted as wrong)
    extra     Y if the answer also states something wrong or not in the file, else N
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from .run_eval import wilson

SIZE = 100
SEED = 20261024
COLUMNS = ["n", "file", "question", "gold_answer", "agent_answer", "correct", "extra", "notes"]


def distinct(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen, out = set(), []
    for r in rows:
        key = (r["id"], r["phrasing"], r["answer"].strip())
        if r["expect"] != "OUT_OF_SCOPE" and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def sample(rows: list[dict[str, Any]], size: int = SIZE, seed: int = SEED) -> dict[str, Any]:
    rng = random.Random(seed)
    failed = [r for r in rows if not r["correct"]]
    passed = [r for r in rows if r["correct"]]
    take_failed = failed if len(failed) <= size // 2 else rng.sample(failed, size // 2)
    take_passed = rng.sample(passed, min(len(passed), size - len(take_failed)))
    chosen = [(r, False) for r in take_failed] + [(r, True) for r in take_passed]
    rng.shuffle(chosen)
    return {
        "rows": chosen,
        "population": {"auto_correct": len(passed), "auto_wrong": len(failed)},
    }


def make(results: Path) -> None:
    report = json.loads(results.read_text(encoding="utf-8"))
    drawn = sample(distinct(report["rows"]))
    out = results.parent / "audit"
    out.mkdir(exist_ok=True)
    key = []
    # utf-8-sig so that Excel shows the text correctly.
    with (out / "sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for n, (r, auto) in enumerate(drawn["rows"], start=1):
            writer.writerow([n, r["file"], r["question"], r["gold"], r["answer"], "", "", ""])
            key.append(
                {"n": n, "id": r["id"], "phrasing": r["phrasing"], "repeat": r["repeat"],
                 "auto_correct": auto}
            )  # fmt: skip
    (out / "key.json").write_text(
        json.dumps(
            {
                "results": results.name,
                "run": results.parent.name,
                "seed": SEED,
                "population": drawn["population"],
                "rows": key,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{len(key)} rows -> {out / 'sheet.csv'} (automatic scores in key.json: keep it closed)")


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa for two raters' yes/no judgements."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return None if math.isclose(expected, 1.0) else round((observed - expected) / (1 - expected), 3)


def agreement(rows: list[dict[str, Any]], population: dict[str, int]) -> dict[str, Any]:
    """rows: {"auto": bool, "human": bool, "extra": bool}."""
    passed = [r for r in rows if r["auto"]]
    failed = [r for r in rows if not r["auto"]]
    too_generous = sum(1 for r in passed if not r["human"])
    too_strict = sum(1 for r in failed if r["human"])
    # Overall agreement, each group weighted by its share of the run.
    total = population["auto_correct"] + population["auto_wrong"]
    weighted = sum(
        population[k] / total * (sum(1 for r in g if r["auto"] == r["human"]) / len(g))
        for k, g in (("auto_correct", passed), ("auto_wrong", failed))
        if g
    )
    return {
        "scored": len(rows),
        "scorer_passed_but_person_says_wrong": {
            "k": too_generous,
            "n": len(passed),
            "ci95": wilson(too_generous, len(passed)),
        },
        "scorer_failed_but_person_says_right": {
            "k": too_strict,
            "n": len(failed),
            "ci95": wilson(too_strict, len(failed)),
        },
        "agreement_weighted_to_run": round(weighted, 3),
        "cohens_kappa_in_sample": kappa([(r["auto"], r["human"]) for r in rows]),
        "answers_with_an_extra_wrong_claim": sum(1 for r in rows if r["extra"]),
    }


def score(sheet: Path) -> None:
    key = json.loads((sheet.parent / "key.json").read_text(encoding="utf-8"))
    auto = {k["n"]: k for k in key["rows"]}
    rows, blank, disagreements = [], [], []
    with sheet.open(encoding="utf-8-sig", newline="") as f:
        for line in csv.DictReader(f):
            mark = line["correct"].strip().upper()[:1]
            if mark not in ("Y", "N", "P"):
                blank.append(line["n"])
                continue
            k = auto[int(line["n"])]
            row = {
                "auto": k["auto_correct"],
                "human": mark == "Y",
                "extra": line["extra"].strip().upper().startswith("Y"),
            }
            rows.append(row)
            if row["auto"] != row["human"]:
                disagreements.append({"n": k["n"], "id": k["id"], "auto": row["auto"],
                                      "person": mark, "notes": line["notes"]})  # fmt: skip
    if blank:
        print(f"not scored yet: rows {', '.join(blank)}")
    result = agreement(rows, key["population"]) | {"disagreements": disagreements}
    (sheet.parent / "agreement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["make", "score"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    (make if args.action == "make" else score)(args.path)


if __name__ == "__main__":
    main()
