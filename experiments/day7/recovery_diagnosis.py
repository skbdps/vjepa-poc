"""Posthoc Day7 recovery accounting from saved CSV/JSON evidence only.

No renderer, feature cache, model inference, threshold change, or model selection
is used. Repeated training seeds are reported explicitly, never as new videos.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics

SEEDS = (1701, 1702, 1703)
ARMS = ("retrieval_only", "predictive")
CONDITIONS = ("long_occlusion", "crossing", "scale_camera")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boolean(value):
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Invalid saved boolean: {value!r}")


def rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def diagnose(run):
    run = Path(run)
    inputs = ("analysis/independent_analysis.json", "test/results.json",
              "test/prediction_rows.csv", "test/future_rows.csv",
              "diagnostics/mechanism_reliance_v1/reliance_results.json")
    hashes = {name: sha256(run / name) for name in inputs}
    audit = json.loads((run / inputs[0]).read_text())
    primary = json.loads((run / inputs[1]).read_text())
    interventions = json.loads((run / inputs[4]).read_text())
    if audit.get("status") != "passed" or primary.get("test_selection") != "none":
        raise ValueError("A completed audited, unselected primary test is required")
    for name in inputs[1:4]:
        if audit["source_files"].get(name) != hashes[name]:
            raise ValueError(f"Independent audit is stale for {name}")
    for key, name in (("primary_results_sha256", inputs[1]),
                      ("primary_prediction_rows_sha256", inputs[2])):
        if interventions.get(key) != hashes[name]:
            raise ValueError("Mechanism diagnostics refer to different primary evidence")
    rows, futures = read_csv(run / inputs[2]), read_csv(run / inputs[3])
    key = lambda r: (r["method"], r["scene"], r["target"], int(r["tubelet"]))
    indexed = {key(row): row for row in rows}
    if len(indexed) != len(rows) or len(rows) != 26784:
        raise ValueError("Primary rows must contain every unique saved observation")
    base = [r for r in rows if r["method"] == "predictive_seed1701"]
    base_recovery = [r for r in base if boolean(r["recovery"])]
    event_key = lambda r: (r["scene"], r["target"], int(r["tubelet"]))
    expected_events = {event_key(r) for r in base_recovery}
    if len(base) != 18 * 4 * 31 or len(expected_events) != 18:
        raise ValueError("Unexpected fixed Day7 observation/recovery cohort")
    method_records, condition_records, future_records, latency_records = {}, {}, {}, {}
    for arm in ARMS:
        method_records[arm], future_records[arm], latency_records[arm] = {}, {}, {}
        for seed in SEEDS:
            method = f"{arm}_seed{seed}"
            subset = [r for r in rows if r["method"] == method]
            recovery = [r for r in subset if boolean(r["recovery"])]
            if {event_key(r) for r in recovery} != expected_events:
                raise ValueError("Paired training seeds have different recovery events")
            partitions = Counter("correct" if boolean(r["hit"]) else
                                 "not_emitted" if not boolean(r["present"]) else
                                 "emitted_wrong_car" if boolean(r["wrong_car"]) else
                                 "other_emitted_error" for r in recovery)
            correct = partitions.get("correct", 0)
            method_records[arm][str(seed)] = {
                "correct_recovery": rate(correct, len(recovery)),
                "failure_partitions": {k: partitions.get(k, 0) for k in
                    ("not_emitted", "emitted_wrong_car", "other_emitted_error")},
                "non_emission_among_failures": rate(partitions.get("not_emitted", 0), len(recovery)-correct)}
            future_records[arm][str(seed)] = {}
            ff = [r for r in futures if r["method"] == method]
            if len(ff) != primary["methods"][method]["future"]["targets"]:
                raise ValueError("Future CSV count differs from primary report")
            for is_recovery in (False, True):
                selected = [r for r in ff if boolean(indexed[
                    (method, r["scene"], r["target"], int(r["target_step"]))]["recovery"]) == is_recovery]
                label = "first_visible_after_absence" if is_recovery else "other_visible_future_targets"
                future_records[arm][str(seed)][label] = {
                    "scored_targets": len(selected),
                    "forecast_mean_cosine": statistics.mean(float(r["forecast_cosine"]) for r in selected),
                    "anchor_copy_mean_cosine": statistics.mean(float(r["anchor_copy_cosine"]) for r in selected),
                    "forecast_own_future_part_hit": rate(sum(boolean(r["forecast_own_future_part_hit"]) for r in selected), len(selected)),
                    "anchor_own_future_part_hit": rate(sum(boolean(r["anchor_own_future_part_hit"]) for r in selected), len(selected))}
            latency_records[arm][str(seed)] = {}
            for condition in CONDITIONS:
                events = [r for r in recovery if r["condition"] == condition]
                histogram = Counter()
                for event in events:
                    first = int(event["tubelet"])
                    delay = "no_later_correct_hit_before_clip_end_or_next_absence"
                    for t in range(first, 32):
                        current = indexed[(method, event["scene"], event["target"], t)]
                        if current["state"] == "absent":
                            break
                        if boolean(current["hit"]):
                            delay = str(t-first)
                            break
                    histogram[delay] += 1
                latency_records[arm][str(seed)][condition] = {
                    "recovery_events": len(events), "first_correct_hit_delay_tubelets": dict(sorted(histogram.items())),
                    "available_tubelets_including_recovery": dict(sorted(Counter(
                        str(32-int(r["tubelet"])) for r in events).items()))}
        condition_records[arm] = {}
        for condition in CONDITIONS:
            records = [audit["methods"][f"{arm}_seed{s}"]["by_condition"][condition] for s in SEEDS]
            condition_records[arm][condition] = {}
            for metric in ("localization_accuracy_given_visible", "false_presence_given_absent",
                           "wrong_car_given_visible", "recovery_accuracy", "presence_recall_given_visible"):
                values = [r[metric] for r in records]
                condition_records[arm][condition][metric] = {
                    "per_seed": {str(s): v for s, v in zip(SEEDS, values)},
                    "mean_rate_across_seeds": statistics.mean(v["rate"] for v in values) if values[0]["rate"] is not None else None}
    repeated = {}
    for arm, seeds in method_records.items():
        correct = sum(r["correct_recovery"]["numerator"] for r in seeds.values())
        total = sum(r["correct_recovery"]["denominator"] for r in seeds.values())
        failures = {k: sum(r["failure_partitions"][k] for r in seeds.values()) for k in
                    ("not_emitted", "emitted_wrong_car", "other_emitted_error")}
        repeated[arm] = {"seed_event_evaluations": total, "distinct_recovery_events": len(expected_events),
                         "correct_recovery": rate(correct, total), "failure_partitions": failures,
                         "non_emission_among_failures": rate(failures["not_emitted"], total-correct)}
    reliance = {}
    for method, record in interventions["methods"].items():
        reliance[method] = {}
        for name, changed in record["interventions"].items():
            baseline = record["normal"]
            reliance[method][name] = {
                "balanced_utility_change": changed["overall"]["balanced_localization_and_absence"]-baseline["balanced_localization_and_absence"],
                "false_presence_change": changed["overall"]["false_presence_given_absent"]["rate"]-baseline["false_presence_given_absent"]["rate"],
                "emitted_prediction_change": rate(changed["sensitivity"]["emitted_prediction_changes"], changed["sensitivity"]["scorable_part_pairs"])}
    return {"format": "day7_posthoc_recovery_diagnosis_v1", "source_sha256": sha256(__file__),
        "input_sha256": hashes, "posthoc": True, "model_selection_performed": False,
        "model_inference_performed": False, "scene_generation_performed": False,
        "independent_clips": 18, "training_seeds": list(SEEDS),
        "distinct_recovery_events": len(expected_events),
        "clips_with_recovery_events": len({r["scene"] for r in base_recovery}),
        "distinct_recovery_events_by_condition": dict(Counter(r["condition"] for r in base_recovery)),
        "primary_contrast_unchanged": audit["primary_contrast"],
        "recovery_per_seed": method_records, "repeated_seed_event_accounting": repeated,
        "condition_metrics": condition_records, "future_visibility_conditional_diagnostics": future_records,
        "post_recovery_first_hit": latency_records, "fixed_checkpoint_reliance": reliance,
        "interpretation_limits": [
            "This is descriptive analysis after viewing the completed test; it is not a new held-out success criterion.",
            "There are18 distinct recovery events in12 clips, repeated across3 training seeds;54 evaluations are not54 independent events.",
            "Future cosine and own-part ranking condition on a visible future target and compare GT-pooled part representations, not dense spatial localization or absence decisions.",
            "No later correct hit is right-censored by clip end or subsequent absence. Long-occlusion recovery begins at step28, leaving only4 tubelets including that step.",
            "Non-emission identifies the immediate output failure; it does not establish a unique cause or prove the hidden conditional patch would be correct.",
            "Owner-context removal and forecast replacement shift the trained input/state distribution; reliance is not proof of architectural superiority or the original failure's cause.",
            "Any redesign now treats Day7 as development evidence and needs new held-out clips. The source experiment and its primary decision remain unchanged."],
        "next_hypothesis": "Decouple part-identity confidence from visibility and explicitly train rare reappearance transitions while retaining predictive JEPA state. Test on a new holdout with absence and overall-utility guards; no SAM substitution or claimed unique root cause."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = diagnose(args.run)
    path = args.out or args.run / "analysis/recovery_diagnosis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(path), "distinct_events": result["distinct_recovery_events"],
                      "repeated_seed_event_accounting": result["repeated_seed_event_accounting"]}, indent=2))
