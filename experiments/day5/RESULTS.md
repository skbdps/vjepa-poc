# Day 5 results: parent containment fails the fixed accuracy guardrails

**Keep parent containment off by default.** Across twelve new synthetic clips, intersecting each part with its tracked car reduced wrong-car effective paint from **839 to 70 pixels**, but removed **122,251 correctly covered visible pixels**. Effective visible mean IoU fell **93.03% → 90.50%** and pooled recall **93.07% → 91.30%**. All four predeclared accuracy guardrails failed. The four-part editor is a useful control artifact; this containment rule is not an overall tracking improvement under the frozen criterion.

Three smoke clips and all twelve held-out clips completed on a Tesla T4. An independent CPU audit regenerated the labels, rescored the saved masks, and matched the per-frame rows, aggregates, sliver subgroup, all four guards and paired bootstrap intervals. No policy, threshold or model changed after the freeze; unsuccessful smoke results did not alter the test. Browser interaction and exported-recipe replay verification also passed; their distinct scope and saved evidence are documented in [EDITOR_VALIDATION.md](EDITOR_VALIDATION.md).

## Evidence and fixed setup

- [Frozen protocol](PROTOCOL.md) and [configuration](frozen_config.json). Experiment commit: `de419ad916e57e04271c14fd830a8a52022b764e`, committed before Day 5 GPU inference.
- Source digest: `fea39b4e91e3e02177503b1f3c0444c54f62966e8cf4a9844ae76833cbfc6314`.
- [Measured test JSON](run_2026-10-09/test/results.json), [runner-generated report](run_2026-10-09/test/RESULTS.md), and [independent audit](run_2026-10-09/independent_audit.md).
- [Original GPU run archive](run_2026-10-09/gpu_run.zip) includes all child/parent caches, per-frame CSVs, source and manifests; model weights and regenerable frames are excluded.
- [Smoke results](run_2026-10-09/smoke/results.json) and [fixed crossing diagnostic](run_2026-10-09/test/test_crossing_6200_comparison.mp4). The preview seed was chosen before viewing its results.

SAM2.1 Hiera Tiny uses the same official revision, checkpoint, FP32 inference, disabled optional postprocessing and quality-100 JPEG input conversion as Day 4. Four child parts and two whole cars are tracked in separate inference states. Both arms reuse identical child predictions; only the parent intersection differs. Each tracker receives first-frame annotations and RGB, never later labels.

## What was scored

The primary unit is a part-frame after frame zero: **12 × 63 × 4 = 3,024**, partitioned into **2,670 visible** and **354 absent**. Any nonempty true part counts as visible, including thin slivers. Visible mean IoU averages all visible part-frames; pooled recall divides correctly covered pixels by **6,929,781 visible ground-truth pixels**. Empty predictions on a visible target score zero.

Raw masks are the tracker output before editing protection. Effective edits exclude the union of every other **original raw child mask**, including disabled children. Parent containment then intersects that already-protected region with its own predicted car. It cannot release overlaps or add paint. All four parts are treated as active for evaluation. Wrong-car pixels intersect the full visible other car, including its untagged body.

| Representation | Arm | Visible mean IoU | Pooled visible recall | Wrong-car pixels | Total predicted/permitted pixels |
|---|---|---:|---:|---:|---:|
| Raw mask | Raw parts | 95.49% | 96.26% | 222,521 | 7,051,505 |
| Raw mask | Parent intersection | 92.99% | 94.50% | 196 | 6,705,910 |
| Effective edit | Raw parts | 93.03% | 93.07% | 839 | 6,608,141 |
| Effective edit | Parent intersection | 90.50% | 91.30% | 70 | 6,484,286 |

Before adding any parent constraint, the existing overlap veto already reduces the raw masks’ 222,521 wrong-car pixels to 839 permitted pixels. Thus the parent arm’s raw-mask reduction of 222,521 → 196 substantially overstates the **incremental editor benefit** if presented without overlap protection. This is an arithmetic diagnostic of the existing policy on these same caches, not a new independently tested overlap-veto method.

The primary reduction is **769 pixels, or 91.66% of an already small baseline**. Its predicted-pixel fraction changes from **839/6,608,141 = 0.01270%** to **70/6,484,286 = 0.001080%**. Wrong-car part-frame events change from **103/3,024 = 3.41%** to **19/3,024 = 0.63%**. The changing-denominator fraction does not replace the primary absolute count.

Containment removes 123,855 effective pixels in total: **122,251 true-positive pixels and 1,604 false-positive pixels**, of which only 769 are on the other car. Therefore **98.70% of removed effective pixels were correctly covered**. It also leaves 156,943 same-car wrong-region pixels. Containment cannot verify that a retained mask is the correct door/window within the correct car.

## Every accuracy guardrail failed

The allowed loss was at most **1.00 percentage point** for each metric, fixed before inference. Improvement in wrong-car paint alone was insufficient.

| Guardrail | Observed loss | Allowed loss | Outcome |
|---|---:|---:|---|
| Raw mask: visible mean IoU | 2.498 pp | 1.000 pp | Fail |
| Raw mask: pooled visible recall | 1.767 pp | 1.000 pp | Fail |
| Effective edit: visible mean IoU | 2.534 pp | 1.000 pp | Fail |
| Effective edit: pooled visible recall | 1.764 pp | 1.000 pp | Fail |

## Regressions by condition

Each condition has four clips and 1,008 scored part-frames. All incremental wrong-car paint reduction occurs in crossing clips, where visible accuracy also deteriorates substantially.

| Condition | Visible / absent | Effective IoU: raw → parent | Effective recall: raw → parent | Wrong-car effective pixels |
|---|---:|---:|---:|---:|
| long_occlusion | 675 / 333 | 95.968% → 95.948% | 97.097% → 97.081% | 0 → 0 |
| crossing | 987 / 21 | 85.934% → 79.100% | 83.376% → 78.534% | 839 → 70 |
| scale_camera | 1008 / 0 | 98.007% → 98.000% | 99.509% → 99.497% | 0 → 0 |

Crossing accounts for 121,640 of the 122,251 correct effective pixels lost. All four crossing clips lose visible mean IoU. The largest regression, `test_crossing_6202`, falls **72.79% → 61.82% IoU** and **66.92% → 59.16% recall**, while wrong-car paint falls 267 → 32 pixels. Even the fixed demonstration clip, `test_crossing_6200`, falls **99.13% → 92.66% IoU** and **99.81% → 95.06% recall**. It is not a best-result selection.

## Thin visible parts, absence and parent failures

Fifty-six visible part-frames have no eligible Day 4 ground-truth patch in their two-frame pair. This includes thin or moving parts, not only physically small objects. Their effective mean IoU falls **70.60% → 46.83%**, pooled recall **82.37% → 57.73%**, and nonempty output coverage **55/56 → 49/56**. Raw-mask subgroup IoU falls **70.43% → 46.83%** and recall **83.02% → 57.73%**. Parent clipping particularly damages this subgroup.

Across the 354 absent child part-frames, false mask presence falls only **39/354 (11.02%) → 36/354 (10.17%)**. Effective absent paint falls **47,412 → 47,285 pixels**; raw absent-mask pixels fall **50,678 → 47,285**. All three removed false-presence events come from crossing; long-occlusion false presence remains **36/333**, with 47,293 → 47,285 effective absent pixels. The rule does not solve hidden-part hallucination.

Whole-parent evaluation has **1,512 car-frames: 1,399 visible and 113 absent**. Parent visible mean IoU is **94.86%** and pooled recall **97.86%**, with no predicted masks on the 113 fully absent car-frames. However, crossing parent IoU is only **86.33%** and recall **94.21%**; its masks include **995,312 wrong-owner pixels**, all in crossing. Parent IoU in `test_crossing_6202` is **72.23%**. These post-run diagnostics support the interpretation that imperfect parent tracking removes correct children during interactions. A high aggregate parent score does not guarantee safe containment, and no identity-recovery capability is demonstrated.

## Uncertainty and the separate patch score

The frozen analysis resamples paired whole clips 2,000 times with seed 9514, recomputing pooled ratios on each draw. Intervals are descriptive for these twelve clips in this renderer family; they are not evidence of generalization. Differences below are parent minus raw.

| Representation / metric | Difference | Paired clip 95% interval |
|---|---:|---:|
| raw mask / wrong-car pixels | -222,325 pixels | [-530,793, -98] pixels |
| raw mask / visible mean IoU | -2.498 pp | [-4.744, -0.573] pp |
| raw mask / pooled visible recall | -1.767 pp | [-3.464, -0.198] pp |
| effective edit / wrong-car pixels | -769 pixels | [-1,570, -68] pixels |
| effective edit / visible mean IoU | -2.534 pp | [-4.815, -0.585] pp |
| effective edit / pooled visible recall | -1.764 pp | [-3.460, -0.196] pp |

The accuracy-loss intervals include losses both below and above the 1 pp guardrail; the **point estimates** fail the predeclared milestone. The intervals consistently favor lower visible accuracy after containment, even though wrong-car pixels also decrease.

The secondary Day 4 patch readout keeps threshold **0.9404296875**, without gating edits. Its denominator is **1,283 visible tubelets, 169 absent and 36 ambiguous excluded**. Correct visible localization declines **1,231/1,283 (95.95%) → 1,204/1,283 (93.84%)**. Wrong-car locations decrease 31 → 0, absent false presence stays 7/169, and immediate recovery decreases 5/12 → 4/12. Those two-frame location measures are not dense edit accuracy.

## Cost, artifact status and interpretation

Each clip adds **two whole-car masks at frame zero** to the four child annotations, plus a separate two-parent SAM2 pass. Measured child propagation averages **82.06 seconds/clip** and parent propagation **46.74 seconds/clip** on the T4. Across twelve clips: **984.73 + 560.83 = 1,545.56 seconds**, a **56.95% inference-time increase** over child tracking alone. These are model-pass timings, excluding setup, downloads, scoring and export.

The [four-part hierarchy editor](run_2026-10-09/hierarchy/interactive_hierarchy_editor.html) and [registry](run_2026-10-09/hierarchy/scene_registry.json) preserve independent car/part controls and replayable recipes. Containment is **off by default**. Native replay and browser share the same controls schema and pixel arithmetic; actual source pixels differ because the browser embeds compressed video. Protection guarantees concern predicted masks before lossy encoding. Browser QA verified independent controls, playback/scrubbing, actual recipe downloads, reset and pasted-recipe restoration, and atomic rejection of an invalid schema. Both exported containment-OFF and containment-ON recipes replayed natively over 64 frames each with zero protection violations; see [EDITOR_VALIDATION.md](EDITOR_VALIDATION.md). PNG export was not verified.

This is a controlled negative result for unconditional parent intersection, using new seeds in the same 2-D sprite renderer. No faces, 3-D persistence, generative consistency or unseen real-video behavior was tested. The useful outcome is a reusable hierarchy-control artifact and a clear boundary: the current parent gate suppresses some unsafe paint but sacrifices too much correct coverage. A future uncertainty-aware parent or recovery method would require a newly frozen hypothesis and fresh test cases; this run does not validate such a method.

## Bounded post-hoc follow-up

After these results were known, one [other-parent veto](../day6/POSTHOC_DAY5_EXPLORATION.md) was evaluated using the same Day 5 caches as **development data**, with no new GPU inference. It reduced effective wrong-car pixels 839 → 136 but lost 211,396 correct visible pixels; effective IoU fell 93.03% → 89.60% and recall 93.07% → 90.02%. All four accuracy guards failed again. This bounded negative screen did not justify promoting that rule to a fresh GPU validation run. It is not a second held-out result and does not change the original Day 5 conclusion or default.
