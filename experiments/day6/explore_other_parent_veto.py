"""POST-HOC development exploration on already-viewed Day5 caches only.

One fixed alternative: raw child M_i AND NOT predicted other parent P_other;
effective edit E_i AND NOT P_other, where E_i preserves all original child
overlaps. No thresholds, dilation, parameter search, inference or new seeds.
Day5's former test set is development evidence for this new policy. This script
cannot establish a held-out improvement; a new frozen cohort would be required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
DAY5 = HERE.parent / "day5"
sys.path.insert(0, str(DAY5))
import audit_results as audit
import parent_benchmark as renderer

BASELINE = "raw_part"
CANDIDATE = "other_parent_veto"
REPRESENTATIONS = ("raw_mask", "effective_edit")
REGIONS = ("own_parent_only", "both_parents", "other_parent_only", "neither_parent")
POLICY = {"name": CANDIDATE,
          "raw_mask": "M_i AND NOT P_other(i)",
          "effective_edit": "E_i AND NOT P_other(i)",
          "E_i": "M_i AND count(all original raw child masks at pixel)==1",
          "thresholds": "none", "dilation": 0, "parameter_search": "none",
          "new_model_inference": False, "new_seeds": False,
          "source_cohort": "Completed Day5 12-clip former test, now viewed development data",
          "annotation_cost": "Four first-frame child masks plus two first-frame whole-parent masks",
          "quality_guards": "At most .01 absolute loss in raw AND effective visible mean IoU and pooled pixel recall"}


def add_counts(target, source):
    for category, values in source.items():
        if category not in target:
            target[category] = {region: 0 for region in REGIONS}
        for region, count in values.items():
            target[category][region] += count


def classify_parent_support(scene, owners, children, parents):
    """Where does correct baseline paint disappear under the Day5 own-parent gate?"""
    effective = children & (np.count_nonzero(children, axis=1) == 1)[:, None]
    result = {category: {region: 0 for region in REGIONS}
              for category in ("correct_effective_pixels", "wrong_car_effective_pixels",
                               "other_false_positive_effective_pixels", "correct_visible_sliver_pixels")}
    eligible = audit.eligible_pairs(scene.masks)
    for frame in range(1, 64):
        for j, pid in enumerate((1, 2, 3, 4)):
            owner = 1 + j // 2
            own, other = parents[frame, j // 2], parents[frame, 1 - j // 2]
            regions = {"own_parent_only": own & ~other,
                       "both_parents": own & other,
                       "other_parent_only": ~own & other,
                       "neither_parent": ~own & ~other}
            truth = scene.masks[frame] == pid
            mask = effective[frame, j]
            categories = {"correct_effective_pixels": mask & truth,
                          "wrong_car_effective_pixels": mask & (owners[frame] == 3 - owner),
                          "other_false_positive_effective_pixels": mask & ~truth & (owners[frame] != 3 - owner),
                          "correct_visible_sliver_pixels": mask & truth if not eligible[pid][frame // 2] else np.zeros_like(mask)}
            for category, pixels in categories.items():
                require_total = 0
                for region, region_mask in regions.items():
                    count = int(np.count_nonzero(pixels & region_mask))
                    result[category][region] += count
                    require_total += count
                audit.require(require_total == int(np.count_nonzero(pixels)), "Parent support partition failed")
    return result


def criterion(arms):
    guards = {}
    for representation in REPRESENTATIONS:
        for metric in ("mean_iou_given_visible", "visible_micro_pixel_recall"):
            raw = arms[BASELINE][representation]["overall"][metric]["rate"]
            candidate = arms[CANDIDATE][representation]["overall"][metric]["rate"]
            loss = raw - candidate
            guards[f"{representation}.{metric}"] = {"baseline_minus_candidate": loss,
                "maximum_allowed_loss": .01, "passed": loss <= .01 + 1e-12}
    baseline = arms[BASELINE]["effective_edit"]["overall"]
    candidate = arms[CANDIDATE]["effective_edit"]["overall"]
    removed_wrong = baseline["wrong_car_pixels"] - candidate["wrong_car_pixels"]
    removed_correct = (baseline["visible_micro_pixel_recall"]["numerator"] -
                       candidate["visible_micro_pixel_recall"]["numerator"])
    return {"status": "POST_HOC_DEVELOPMENT_ONLY",
            "wrong_car_effective_pixels_strictly_lower": removed_wrong > 0,
            "wrong_car_pixels_removed": removed_wrong,
            "relative_wrong_car_reduction": audit.ratio(removed_wrong, baseline["wrong_car_pixels"]),
            "correct_visible_effective_pixels_removed": removed_correct,
            "false_positive_effective_pixels_removed": baseline["false_positive_pixels"] - candidate["false_positive_pixels"],
            "four_quality_guards": guards,
            "passes_point_estimate_development_gate": removed_wrong > 0 and all(g["passed"] for g in guards.values()),
            "held_out_evidence": False}


def run(run_root, output):
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    original_path = run_root / "test" / "results.json"
    original = json.loads(original_path.read_text())
    verified_path = run_root / "test" / "independent_audit.json"
    verified = json.loads(verified_path.read_text())
    audit.require(verified["status"] == "passed" and verified["audited_result_sha256"] == audit.sha(original_path),
                  "A passed independent audit of these completed Day5 results is required")
    freeze = json.loads((DAY5 / "frozen_config.json").read_text())
    audit.equal(original["frozen_config"], freeze, "Day5 unchanged freeze")
    audit.equal({name: audit.sha(HERE.parent / name) for name in freeze["source_hashes"]},
                freeze["source_hashes"], "Day5 source hashes")
    expected = {f"test_{condition}_{seed}": (condition, seed)
                for condition, seeds in audit.EXPECTED_TEST.items() for seed in seeds}
    records = {c["scene"]: c for c in original["clips"]}
    audit.require(len(records) == len(original["clips"]) == 12 and records.keys() == expected.keys(),
                  "Only the complete already-viewed Day5 cohort is allowed")
    # Verify the entire cached cohort before regeneration. Never render a new seed.
    manifests = {}
    for name in expected:
        folder = run_root / "clips" / name
        path = folder / "clip_manifest.json"
        audit.require(audit.sha(path) == records[name]["clip_manifest_sha256"], f"Changed Day5 manifest: {name}")
        manifests[name] = json.loads(path.read_text())
        for role in ("children", "parents"):
            audit.require(audit.sha(folder / role / "sam2_masks.npz") == manifests[name]["cache_sha256"][role],
                          f"Changed Day5 prediction cache: {name}/{role}")
    rows, support, per_clip_support = [], {}, {}
    child_seconds = parent_seconds = 0.0
    for name, (condition, seed) in expected.items():
        scene, owners = renderer.generate_scene(seed, condition, name)
        audit.require(audit.array_sha(scene.frames) == manifests[name]["identity"]["rgb_sha256"], "Changed source RGB")
        audit.require(audit.array_sha(scene.masks) == records[name]["part_truth_sha256"], "Changed part truth")
        audit.require(audit.array_sha(owners) == records[name]["owner_truth_sha256"], "Changed owner truth")
        arrays = {}
        for role, ids in (("children", [1, 2, 3, 4]), ("parents", [101, 102])):
            with np.load(run_root / "clips" / name / role / "sam2_masks.npz", allow_pickle=False) as cache:
                audit.require(cache["ids"].tolist() == ids, "Changed mask ID order")
                arrays[role] = cache["masks"]
            audit.require(arrays[role].dtype == bool and arrays[role].shape == (64, len(ids), 384, 384),
                          "Unexpected cached mask shape/type")
        child, parent = arrays["children"], arrays["parents"]
        other_parent = parent[:, [1, 1, 0, 0]]
        effective = child & (np.count_nonzero(child, axis=1) == 1)[:, None]
        candidate_effective = effective & ~other_parent
        audit.require(not np.any(candidate_effective & ~effective), "Veto released a protected edit pixel")
        variants = {BASELINE: {"raw_mask": child, "effective_edit": effective},
                    CANDIDATE: {"raw_mask": child & ~other_parent, "effective_edit": candidate_effective}}
        for arm, representations in variants.items():
            for representation, masks in representations.items():
                rows.extend(audit.independently_score(scene, owners, masks, arm, representation))
        per_clip_support[name] = classify_parent_support(scene, owners, child, parent)
        add_counts(support, per_clip_support[name])
        child_seconds += manifests[name]["timing"]["children"]["seconds"]
        parent_seconds += manifests[name]["timing"]["parents"]["seconds"]
        print(f"POSTHOC_DAY5_CLIP_DONE {name}", flush=True)
    arms = {arm: {representation: audit.aggregate([r for r in rows if r["arm"] == arm and r["representation"] == representation])
                  for representation in REPRESENTATIONS} for arm in (BASELINE, CANDIDATE)}
    # Baseline consistency is a prerequisite, not an additional candidate search.
    audit.equal(arms[BASELINE], original["arms"][BASELINE], "Unchanged Day5 baseline")
    correct = support["correct_effective_pixels"]
    wrong = support["wrong_car_effective_pixels"]
    mechanism = {"parent_support_partition": support,
        "own_parent_constraint_correct_pixels_lost": correct["other_parent_only"] + correct["neither_parent"],
        "own_parent_constraint_loss_due_to_neither_parent_support": correct["neither_parent"],
        "own_parent_constraint_loss_due_to_other_parent_only_support": correct["other_parent_only"],
        "other_parent_veto_correct_pixels_lost": correct["other_parent_only"] + correct["both_parents"],
        "other_parent_veto_wrong_car_pixels_removed": wrong["other_parent_only"] + wrong["both_parents"],
        "interpretation": "Support categories describe prediction geometry, not proven internal model causes or semantic identity recovery."}
    candidate_assessment = criterion(arms)
    audit.require(mechanism["own_parent_constraint_correct_pixels_lost"] ==
                  original["arms"][BASELINE]["effective_edit"]["overall"]["visible_micro_pixel_recall"]["numerator"] -
                  original["arms"]["parent_intersection"]["effective_edit"]["overall"]["visible_micro_pixel_recall"]["numerator"],
                  "Own-parent failure diagnosis does not reproduce frozen Day5 loss")
    audit.require(mechanism["other_parent_veto_correct_pixels_lost"] == candidate_assessment["correct_visible_effective_pixels_removed"],
                  "Alternative loss does not match support diagnosis")
    result = {"status": "POST_HOC_EXPLORATORY_DEVELOPMENT", "policy": POLICY,
        "cohort": {"clips": 12, "scored_part_frames_per_arm_representation": 3024,
                   "seeds": audit.EXPECTED_TEST, "source_result_sha256": audit.sha(original_path),
                   "source_independent_audit_sha256": audit.sha(verified_path),
                   "Day5_results_unchanged": True, "new_scenes_or_predictions_generated": False},
        "provenance": {"script_sha256": audit.sha(__file__), "audit_helpers_sha256": audit.sha(DAY5 / "audit_results.py"),
                       "Day5_frozen_source_digest": freeze["source_digest"]},
        "arms": arms, "candidate_assessment": candidate_assessment,
        "own_parent_constraint_diagnosis": mechanism, "parent_support_by_clip": per_clip_support,
        "inference_cost_from_Day5_manifests": {"child_seconds": child_seconds, "parent_seconds": parent_seconds,
            "total_seconds": child_seconds + parent_seconds, "parent_overhead_fraction": parent_seconds / child_seconds,
            "new_inference_seconds_for_this_exploration": 0},
        "limitations": ["The alternative was proposed after viewing Day5 results; all current comparisons are exploratory development.",
                        "No held-out or real-video improvement is established.",
                        "Both candidate masks are subsets of the baseline, so wrong-car paint cannot increase by construction.",
                        "A reduction matters only alongside lost correct detail; stable tags do not guarantee identity recovery."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "status": result["status"], "assessment": candidate_assessment}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=HERE / "day5_development_other_parent_veto.json")
    args = parser.parse_args()
    run(args.run_root, args.out)
