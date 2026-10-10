# Day7 progress — completed 9 October 2026

The learned JEPA experiment is complete. A custom 871,810-parameter recurrent
part head now tracks four prompted parts over frozen V-JEPA 2.1 features, with
shared sibling context and a forecast consumed by later retrieval. This is
supervised head training using synthetic part labels. SAM is not used.

## Completed work

- Created fresh 36/9/18 training/development/test splits with deterministic
  spatial flips and coherent car-identity permutations. Test inference receives
  frame-zero part prompts and RGB-derived features only.
- Extracted frozen features, fitted centering on training clips, and trained
  retrieval-only, predictive and coordinate-only arms for 30 epochs each with
  paired seeds 1701, 1702 and 1703: nine models and 270 recorded epoch rows.
- Selected each checkpoint on development data and committed the complete
  checkpoint freeze before opening the 18 test clips. No training adjustment or
  OOM fallback was needed.
- Evaluated all three fixed baselines and all nine learned models. Preserved
  216 prediction archives, 26,784 primary rows, 8,700 future diagnostic rows,
  learning curves, selected-epoch records, source hashes and logs.
- Independently recounted saved predictions and verified the primary reports.
  The primary bootstrap uses 18 paired clips, with the same draw shared across
  the three training seeds. It does not treat repeated models as extra videos.
- Ran descriptive forecast-replacement and sibling-context interventions on all
  six visual checkpoints. Normal replay matched saved results exactly; the
  interventions did not replace or select the primary results.

## What the result supports

The predictive head reached **92.87% visible localization**, compared with
**46.21%** for frozen full-feature cosine matching. The matched retrieval-only
head reached **92.63%** and the coordinate-only control reached **37.73%**.
Learned visual retrieval is therefore a useful component in this rendered world.

The additional future-latent objective remains unproven as a tracking improvement.
Predictive minus retrieval-only balanced utility was **−0.008 percentage points**,
with a paired clip interval **[−3.118, +4.131] pp**. The future target itself was
learned: predictive forecast cosine reached 0.8661, versus 0.7348 for copying the
initial anchor. Better forecast representations did not translate into a clear
primary task gain.

Recovery after full disappearance is the clearest remaining weakness. Predictive
first-visible recovery was **1/18, 9/18 and 1/18** across seeds; every predictive
failure at those recovery events was a non-emission. The working research
question is how to recover visibility without emitting incorrect identities or
hallucinating absent parts. No next-stage training or new test claim is implied
by this diagnosis.

## Provenance and evidence

| Checkpoint | Git revision |
|---|---|
| Experiment source/model/data/extraction | [`1d9f12a`](https://github.com/skbdps/vjepa-poc/commit/1d9f12a5f51b061e33a4417d2789c203b5d1dc86) |
| Analysis/export helpers | [`9af4259`](https://github.com/skbdps/vjepa-poc/commit/9af4259467caa0bdf7980ef83d856fae1f70f61f) |
| Development checkpoint freeze, verified before test extraction | [`b603bd7`](https://github.com/skbdps/vjepa-poc/commit/b603bd76ee97673beff91a79f08292d19276d85c) |

Read [RESULTS.md](RESULTS.md) for the complete interpretation and limits,
[the independent audit](run_2026-10-09/analysis/INDEPENDENT_ANALYSIS.md) for exact
seed effects, and [the fixed comparison video](run_2026-10-09/analysis/test_crossing_10200_fixed_comparison.mp4)
for crossing clip 10200 with seed 1701. See
[README_EXPORT.md](run_2026-10-09/README_EXPORT.md) before attempting replay:
the compact Git prediction files omit weights and internal arrays, while the
separate full archive preserves the original run. Feature caches must be
retained or regenerated for inference.

This is a completed synthetic patch-tracking experiment, not a demonstrated
real-video/face consistency system. Existing Day3–Day6 evidence remains intact.

The [reproducible post-hoc recovery diagnosis](run_2026-10-09/analysis/recovery_diagnosis.json) records exact event counts, input hashes and the limits of the visibility-gate interpretation; [recovery_diagnosis.py](recovery_diagnosis.py) rebuilds it from saved evidence.
