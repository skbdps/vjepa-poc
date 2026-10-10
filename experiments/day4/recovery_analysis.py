"""Supplementary recovery latency analysis of saved Day 4 prediction CSVs.

This post-hoc outcome was proposed after V-JEPA test results, before SAM2
held-out scores were viewed. It does not replace the frozen immediate-recovery
metric and does not change inference, calibration, or primary scoring.

Each event is an existing CSV recovery/is_recovery label. Horizons are measured
from the first frame of that event's two-frame tubelet, at offsets 0, 2, 4, 8.
Only the saved `hit` on a visible target is a recovery. Ambiguous tubelets are
observed but unscorable, cannot be hits, and are reported explicitly. A new
fully absent tubelet ends that episode; later reappearance belongs to a new
event. Clip end or missing rows also censor unfinished follow-up. An observed
hit before censoring remains a known success, separately from complete horizons.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


HORIZONS = (0, 2, 4, 8)
TUBELET_FRAMES = 2
STATES = {"visible", "absent", "ambiguous"}


def _bool(value):
    if value is True or value == "True" or value == "true" or value == "1":
        return True
    if value is False or value == "False" or value == "false" or value == "0":
        return False
    raise ValueError(f"Expected an explicit boolean, got {value!r}")


def normalize_rows(rows):
    """Validate the information needed for temporal, per-part analysis."""
    result, seen = [], set()
    for raw in rows:
        missing = {"scene", "target", "method", "first_frame", "state", "hit"} - raw.keys()
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        flag_names = [key for key in ("recovery", "is_recovery") if key in raw]
        if not flag_names:
            raise ValueError("Need existing recovery or is_recovery labels; events are not inferred")
        flags = [_bool(raw[key]) for key in flag_names]
        if len(set(flags)) != 1:
            raise ValueError("Conflicting recovery and is_recovery labels")
        frame = int(raw["first_frame"])
        if frame < 0 or frame % TUBELET_FRAMES:
            raise ValueError("Expected nonnegative first_frame on the two-frame tubelet grid")
        if "tubelet" in raw and int(raw["tubelet"]) * TUBELET_FRAMES != frame:
            raise ValueError("tubelet and first_frame disagree")
        row = {"scene": str(raw["scene"]), "target": str(raw["target"]),
               "method": str(raw["method"]), "condition": str(raw.get("condition", "unknown")),
               "first_frame": frame, "state": raw["state"], "hit": _bool(raw["hit"]),
               "recovery": flags[0]}
        if row["state"] not in STATES:
            raise ValueError(f"Unexpected target state: {row['state']!r}")
        if (row["hit"] or row["recovery"]) and row["state"] != "visible":
            raise ValueError("Hits and recovery events must be on visible target tubelets")
        if row["hit"] and "present" in raw and not _bool(raw["present"]):
            raise ValueError("A hit cannot be an absent prediction")
        key = (row["method"], row["scene"], row["target"], frame)
        if key in seen:
            raise ValueError(f"Duplicate trajectory row: {key}")
        seen.add(key)
        result.append(row)
    if not result:
        raise ValueError("No prediction rows")
    return result


def _horizon(event, trajectory, horizon):
    start, last = event["first_frame"], max(trajectory)
    states, ambiguous, first_hit, censor = [], [], None, None
    for offset in range(0, horizon + 1, TUBELET_FRAMES):
        frame = start + offset
        if frame not in trajectory:
            censor = {"reason": "clip_end" if frame > last else "missing_row", "offset_frames": offset}
            break
        row = trajectory[frame]
        states.append({"offset_frames": offset, "state": row["state"], "hit": row["hit"]})
        if offset and row["state"] == "absent":
            censor = {"reason": "new_absence", "offset_frames": offset}
            break
        if offset and row["recovery"]:
            # An unexpected second event without an intervening recorded absence
            # indicates incompatible/incomplete labels, not a recoverable gap.
            raise ValueError("Another recovery event appears before recorded absence or a row gap")
        if row["state"] == "ambiguous":
            ambiguous.append(offset)
        if row["state"] == "visible" and row["hit"] and first_hit is None:
            first_hit = offset
    fully_observed = censor is None
    status = ("recovered" if first_hit is not None else
              "not_recovered" if fully_observed else
              "interrupted_by_absence" if censor["reason"] == "new_absence" else
              "censored_unresolved")
    return {"horizon_frames": horizon, "fully_observed": fully_observed,
            "status": status, "first_hit_offset_frames": first_hit,
            "ambiguous_offsets_frames": ambiguous, "observed_states": states,
            "censor": censor}


def _summary(events, horizon):
    checks = [event["horizons"][str(horizon)] for event in events]
    complete = [check for check in checks if check["fully_observed"]]
    recovered = sum(check["status"] == "recovered" for check in checks)
    complete_recovered = sum(check["status"] == "recovered" for check in complete)
    unresolved = sum(check["status"] == "censored_unresolved" for check in checks)
    return {"events": len(checks), "fully_observed_horizons": len(complete),
            "censored_horizons": len(checks) - len(complete),
            "observed_recovered_within_horizon": recovered,
            "recovered_within_fully_observed_horizons": {
                "numerator": complete_recovered, "denominator": len(complete),
                "rate": complete_recovered / len(complete) if complete else None},
            "recovered_before_censoring": sum(
                check["status"] == "recovered" and not check["fully_observed"] for check in checks),
            "fully_observed_without_recovery": len(complete) - complete_recovered,
            "interrupted_by_absence_without_recovery": sum(
                check["status"] == "interrupted_by_absence" for check in checks),
            "unresolved_censored_events": unresolved,
            "observed_success_fraction_lower_bound": recovered / len(checks) if checks else None,
            "success_fraction_upper_bound_allowing_censored_recovery":
                (recovered + unresolved) / len(checks) if checks else None,
            "events_with_ambiguous_tubelets": sum(bool(check["ambiguous_offsets_frames"]) for check in checks),
            "fully_observed_without_ambiguity": sum(not check["ambiguous_offsets_frames"] for check in complete),
            "censor_reasons": dict(sorted(Counter(
                check["censor"]["reason"] for check in checks if check["censor"]).items()))}


def analyze(rows):
    rows = normalize_rows(rows)
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["method"], row["scene"], row["target"])][row["first_frame"]] = row
    events = []
    for key, trajectory in sorted(grouped.items()):
        if len({row["condition"] for row in trajectory.values()}) != 1:
            raise ValueError(f"Condition changes within a trajectory: {key}")
        for frame, row in sorted(trajectory.items()):
            if row["recovery"]:
                events.append({"method": key[0], "scene": key[1], "target": key[2],
                    "condition": row["condition"], "event_first_frame": frame,
                    "horizons": {str(h): _horizon(row, trajectory, h) for h in HORIZONS}})
    methods = {}
    for method in sorted({row["method"] for row in rows}):
        subset = [event for event in events if event["method"] == method]
        conditions = sorted({row["condition"] for row in rows if row["method"] == method})
        methods[method] = {"overall": {str(h): _summary(subset, h) for h in HORIZONS},
            "by_condition": {condition: {str(h): _summary(
                [event for event in subset if event["condition"] == condition], h) for h in HORIZONS}
                for condition in conditions}}
        # This is the frozen immediate metric's event population and hit rule.
        expected = [row for row in rows if row["method"] == method and row["recovery"]]
        immediate = methods[method]["overall"]["0"]["recovered_within_fully_observed_horizons"]
        assert immediate["denominator"] == len(expected)
        assert immediate["numerator"] == sum(row["hit"] for row in expected)
    return {"analysis": "supplementary post-hoc recovery latency", "version": 1,
        "status": "Proposed after V-JEPA test, before SAM2 held-out scores were viewed; not a new primary outcome.",
        "definition": {
            "event": "Existing recovery/is_recovery label only: first unambiguous visible tubelet after absence.",
            "horizons_source_frames": list(HORIZONS), "tubelet_frames": TUBELET_FRAMES,
            "latency": "Difference between tubelet first-frame indices; two-frame resolution, not exact per-frame recovery time, wall-clock latency, or causal detection delay. V-JEPA predictions are offline within encoder windows.",
            "hit": "Saved correct localization hit on a visible target; later hits after renewed absence belong to another episode.",
            "ambiguity": "Observed but unscorable tubelets; never hits; retained and reported, not silently excluded.",
            "censoring": "New full absence, clip end, or missing row truncates follow-up. Success before censoring stays known. New absence is a known episode interruption, not unknown recovery; only clip end/missing rows leave an unrecovered outcome unresolved.",
            "fully_observed": "Every expected tubelet through the horizon is present before renewed absence; ambiguous states may occur.",
            "rates": "Complete-horizon rates use their explicit denominator. Overall bounds leave unresolved censored outcomes unknown; no imputation.",
            "limits": "Descriptive analysis of saved predictions, with repeated parts/events within clips; no frame-independent inference or model rerun."},
        "input_rows": len(rows), "event_count_across_methods": len(events),
        "methods": methods, "events": events}


def verify_immediate(report, primary):
    """Require exact agreement with the saved primary recovery counts."""
    verified = {}
    for method, value in report["methods"].items():
        current = value["overall"]["0"]["recovered_within_fully_observed_horizons"]
        expected = primary["methods"][method]["overall"]["recovery_accuracy"]
        if any(current[key] != expected[key] for key in ("numerator", "denominator")):
            raise ValueError(f"Immediate recovery differs from primary report: {method}")
        verified[method] = {key: current[key] for key in ("numerator", "denominator")}
    return verified


def self_test():
    def row(scene, frame, state, recovery=False, hit=False):
        return dict(scene=scene, target="part", method="fixture", condition="fixture",
                    first_frame=frame, state=state, recovery=recovery, hit=hit)
    rows = [row("delayed", 2, "absent"), row("delayed", 4, "visible", True),
            row("delayed", 6, "ambiguous"), row("delayed", 8, "visible", hit=True),
            row("delayed", 10, "visible", hit=True), row("delayed", 12, "visible", hit=True),
            row("reabsence", 2, "absent"), row("reabsence", 4, "visible", True),
            row("reabsence", 6, "absent"), row("reabsence", 8, "visible", True, True),
            row("reabsence", 10, "visible", hit=True),
            row("gap", 2, "absent"), row("gap", 4, "visible", True),
            row("gap", 8, "visible", hit=True)]
    result = analyze(rows)
    events = {(e["scene"], e["event_first_frame"]): e["horizons"] for e in result["events"]}
    delayed = events[("delayed", 4)]
    assert delayed["2"]["status"] == "not_recovered"
    assert delayed["4"]["first_hit_offset_frames"] == 4
    assert delayed["4"]["ambiguous_offsets_frames"] == [2]
    assert delayed["8"]["fully_observed"]
    assert events[("reabsence", 4)]["4"]["censor"]["reason"] == "new_absence"
    assert events[("reabsence", 4)]["4"]["status"] == "interrupted_by_absence"
    assert events[("reabsence", 8)]["4"]["censor"]["reason"] == "clip_end"
    assert events[("reabsence", 8)]["4"]["status"] == "recovered"
    assert events[("gap", 4)]["4"]["censor"]["reason"] == "missing_row"
    assert events[("gap", 4)]["4"]["first_hit_offset_frames"] is None
    immediate = result["methods"]["fixture"]["overall"]["0"]["recovered_within_fully_observed_horizons"]
    assert immediate == {"numerator": 1, "denominator": 4, "rate": .25}
    verify_immediate(result, {"methods": {"fixture": {"overall": {"recovery_accuracy": immediate}}}})
    for mutation in (rows + [rows[0]], [row("invalid", 2, "absent", hit=True)]):
        try:
            analyze(mutation)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid rows accepted")
    alias = [{"is_recovery": r["recovery"], **{k: v for k, v in r.items() if k != "recovery"}} for r in rows]
    assert analyze(alias)["methods"] == result["methods"]
    return {"status": "passed", "delayed_recovery_with_ambiguity": True,
            "new_absence_starts_new_episode": True, "clip_end_censoring": True,
            "known_success_before_censoring": True, "missing_row_censoring": True,
            "duplicate_and_invalid_state_rejected": True, "immediate_count_preserved": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--primary-report", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    if args.predictions:
        if args.out is None:
            parser.error("--out is required with --predictions")
        with args.predictions.open(newline="") as stream:
            report = analyze(list(csv.DictReader(stream)))
        report["source_predictions"] = str(args.predictions)
        if args.primary_report:
            report["primary_immediate_count_verification"] = verify_immediate(
                report, json.loads(args.primary_report.read_text()))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        for method, value in report["methods"].items():
            print(method, json.dumps({h: {key: v[key] for key in (
                "events", "fully_observed_horizons", "observed_recovered_within_horizon",
                "unresolved_censored_events", "interrupted_by_absence_without_recovery",
                "events_with_ambiguous_tubelets", "censor_reasons")}
                for h, v in value["overall"].items()}))
    elif not args.self_test:
        parser.error("Choose --self-test or --predictions")
