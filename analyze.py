"""Validate raw choices and ledger changes, then report matched contrasts."""

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path

from run import COSTS, DEDUCTION, OUTCOMES, PAYOFFS, RESERVE, digest


def read(path):
    return json.loads(path.read_text())


def signed_difference(first, second):
    transitions = Counter(zip(first, second))
    return {
        "first_only": transitions[True, False],
        "second_only": transitions[False, True],
        "both": transitions[True, True],
        "neither": transitions[False, False],
        "difference_pp": 100 * (sum(first) - sum(second)) / len(first),
    }


def require(test, context):
    if not test:
        raise ValueError(context)


def validate_play(row):
    raw = row["history"][-1]["content"].strip()
    require(row["choice"] == (raw if raw in {"1", "2"} else None), row["id"])


def verified_execution(branch, a, b_by_id):
    context = (a["id"], branch["outcome"], branch["cost"])
    outcome, cost = branch["outcome"], branch["cost"]
    button_b, cause = OUTCOMES[outcome]
    require(branch["history"][:-2] == a["history"], context)
    require(cause in branch["history"][-2]["content"], context)
    require(branch["button_b"] == button_b, context)
    partner = branch["b_trial"]
    require(partner is None if outcome == "system_2" else b_by_id[partner]["choice"] == button_b, context)
    pay_a, pay_b = PAYOFFS["1", button_b]
    before = branch["before"]
    require(before == {"a_round": pay_a, "b_round": pay_b, "a_reserve": RESERVE, "b_reserve": RESERVE}, context)
    raw = branch["history"][-1]["content"].strip()
    require(set(branch["mapping"]) == {"X", "Y"} and set(branch["mapping"].values()) == {"keep", "reduce"}, context)
    action = branch["mapping"].get(raw)
    require(action == branch["action"], context)
    ledger = branch["after"]
    require(ledger["a_round"] == pay_a and ledger["b_reserve"] == RESERVE, context)
    expected_debit = (cost, DEDUCTION) if action == "reduce" else (0, 0)
    require((RESERVE - ledger["a_reserve"], pay_b - ledger["b_round"]) == expected_debit, context)
    return {"deducted": ledger["b_round"] < pay_b, "invalid": action is None}


def summarize(folder):
    manifest = read(folder / "manifest.json")
    require(manifest["runner_sha256"] == digest(Path(__file__).with_name("run.py")), "Check runner hash")
    plays = read(folder / "play.json")
    play_counts = {}
    for role in ("A", "B"):
        trials = plays[role]
        require(len({row["id"] for row in trials}) == len(trials) == manifest["play_trials_per_role"], role)
        for row in trials:
            validate_play(row)
        counts = Counter(row["choice"] for row in trials)
        play_counts[role] = {"1": counts["1"], "2": counts["2"], "invalid": counts[None]}
    b_by_id = {row["id"]: row for row in plays["B"]}
    eligible = [row for row in plays["A"] if row["choice"] == "1"]
    complete = read(folder / "complete.json")
    require(complete == {"eligible": len(eligible), "branches": len(eligible) * 6}, "Completion mismatch")
    require({path.stem for path in folder.glob("A-*.json")} == {a["id"] for a in eligible}, "Missing or extra trials")
    cells = {key: [] for key in product(COSTS, OUTCOMES)}
    exposures = Counter()
    for a in eligible:
        trial = read(folder / f"{a['id']}.json")
        require(trial["a_trial"] == a["id"], a["id"])
        branches = trial["settlements"]
        require(len(branches) == 6 and {(x["cost"], x["outcome"]) for x in branches} == cells.keys(), a["id"])
        require(len({(tuple(x["mapping"].items()), tuple(x["label_order"])) for x in branches}) == 1, a["id"])
        for outcome in OUTCOMES:
            require(len({x["b_trial"] for x in branches if x["outcome"] == outcome}) == 1, a["id"])
        for branch in branches:
            key = (branch["cost"], branch["outcome"])
            result = verified_execution(branch, a, b_by_id)
            cells[key].append(result)
            reduce_label = next(label for label, action in branch["mapping"].items() if action == "reduce")
            shown_first = reduce_label == branch["label_order"][0]
            exposures[(key, reduce_label, shown_first, result["deducted"])] += 1

    step = {key: [x["deducted"] for x in records] for key, records in cells.items()}
    return {
        "manifest": manifest,
        "stage_1_counts": play_counts,
        "matched_trajectories": len(eligible),
        "settlement_branches": complete["branches"],
        "conditions": {
            str(cost): {
                outcome: {
                    "deducted": sum(step[cost, outcome]),
                    "assigned": len(cells[cost, outcome]),
                    "invalid": sum(x["invalid"] for x in cells[cost, outcome]),
                    "percent": 100 * sum(step[cost, outcome]) / len(cells[cost, outcome]),
                }
                for outcome in OUTCOMES
            }
            for cost in COSTS
        },
        "matched_contrasts": {
            str(cost): {
                f"voluntary_2_minus_{control}": signed_difference(step[cost, "voluntary_2"], step[cost, control])
                for control in ("mutual_1", "system_2")
            }
            for cost in COSTS
        },
        "cost_0_minus_cost_1000": {
            outcome: signed_difference(step[0, outcome], step[1000, outcome]) for outcome in OUTCOMES
        },
        "choice_display_audit": [
            {"cost": key[0], "outcome": key[1], "reduce_label": label, "reduce_shown_first": first,
             "deducted": deducted, "n": count}
            for (key, label, first, deducted), count in sorted(exposures.items())
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.results), ensure_ascii=False, indent=2))
