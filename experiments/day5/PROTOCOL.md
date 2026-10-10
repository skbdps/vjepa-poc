# Day 5: parent-car constraints for persistent part edits

## Question and status

Can an independently tracked whole-car mask reduce door/window edits on the wrong car, without materially damaging correct visible parts?

This protocol follows the completed Day 4 results. All Day 4 clips and failure cases are prior development evidence. Day 4 results and model code remain unchanged. Day 5 uses new seeds within the same renderer family; this does not establish generalization to new real footage.

## Fixed methods

Use the same official SAM2.1 Hiera Tiny revision, checkpoint, FP32 precision, disabled optional postprocessing, and quality-100 JPEG input conversion as Day 4. Track the four parts and the two whole cars in separate inference states. Each tracker receives only frame-zero masks and RGB input; subsequent true part/owner masks enter scoring only.

Both arms reuse exactly the same predicted child masks. Let M_i be a raw child mask and P_i its independently predicted parent-car mask.

- Raw tracker mask: M_i.
- Parent-constrained tracker mask: M_i intersect P_i.
- Raw effective edit region: E_i = M_i minus the union of every other raw child mask.
- Parent-constrained effective edit region: E_i intersect P_i.

Keep original raw overlap protection in both arms. This makes every parent-constrained effective edit region a subset of the baseline. Do not recompute overlap from the smaller masks, which could otherwise release protected pixels.

No dilation, additional confidence threshold, learned gate, appearance adaptation, later correction prompt, or parameter search is included. Parent masks add two whole-car frame-zero annotations and extra model computation. Any gain belongs to this combined pipeline, not to an improved SAM2 model or a V-JEPA effect.

## New scenes and freeze

All clips have 64 frames at 384 × 384, two cars, and four tagged parts.

| Split | Long occlusion | Crossing | Scale/camera |
| --- | --- | --- | --- |
| Smoke, execution/implementation checks | 5100 | 5200 | 5300 |
| Held-out comparison | 6100–6103 | 6200–6203 | 6300–6303 |

Commit the implementation, source digests, and fixed policy before executing the held-out stage. The smoke run checks execution and scoring, not parameter selection. If a correctness fix changes the implementation, save the reason and a new freeze before the test. Do not tune or rerun a revised policy on these test clips after seeing their results.

## Outcomes and denominators

The primary outcome is the **absolute number of effective edit pixels on the other car's visible ground-truth owner mask**, summed across the twelve clips, excluding frame zero. Report the rate among all effective edit pixels too, but do not substitute that changing-denominator rate for the primary count. All four parts are treated as active.

There are 12 × 63 × 4 = **3,024 scored part-frames**. Derive and report visible/absent counts from exact renderer labels. Include thin visible parts, not just parts large enough to fill a patch.

Report separately for raw tracker masks and effective edit regions: mean IoU over visible part-frames; pooled visible-pixel recall and precision; absent predicted-mask presence and pixel counts; total predicted/edited pixels; wrong-car pixels; per-condition and per-clip outcomes. Include whole-parent mask IoU and wrong-owner pixels to identify failure sources. Report visible slivers separately using the existing Day 4 eligibility definition.

The common Day 4 patch readout with its original fixed threshold 0.9404296875 is secondary. Preserve its different units, exclusions, and denominators. It is not dense edit accuracy.

## Fixed success criterion

A bounded point-estimate milestone passes only if:

1. Parent-constrained effective edits produce strictly fewer absolute wrong-car pixels.
2. Mean visible IoU loses at most 0.01 absolute versus baseline, for both raw masks and effective edit regions.
3. Pooled visible-pixel recall loses at most 0.01 absolute versus baseline, for both raw masks and effective edit regions.

Report effect size, relative wrong-car reduction, and descriptive paired whole-clip bootstrap intervals. A point-estimate pass does not imply statistical certainty or practical significance. Any failed guard means no overall improvement under this declared criterion. Retain unsuccessful clips and regressions as well as successes.

## Interpretation

Intersection can suppress an edit on the wrong car, but cannot reconstruct a missing door, recover a lost identity, or guarantee a correct parent track. A parent identity failure can affect both of its children. No face, 3-D, automatic part-discovery, or generative-video claim is supported by this experiment.
