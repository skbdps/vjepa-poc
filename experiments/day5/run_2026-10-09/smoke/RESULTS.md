# Day5 parent-intersection experiment

Stage: **smoke**. Fixed masks, first-frame prompts only; no tuning.

Parent masks cost a second SAM2 pass and two additional whole-car annotations.

## Primary effective edit region

Original overlapping child masks remain protected in both arms. Parent clipping cannot release protected paint.

| Arm | Wrong-car pixels | Wrong-car / predicted pixels | Wrong-car part-frames | Visible mean IoU | Visible pooled recall | Absent false presence |
|---|---:|---:|---:|---:|---:|---:|
| raw_part | 221 | 0.013% (221/1737837) | 1.984% (15/756) | 96.756% (647.3/669) | 98.078% (1.706e+06/1739397) | 14.943% (13/87) |
| parent_intersection | 6 | 0.000% (6/1671641) | 0.529% (4/756) | 93.001% (622.2/669) | 94.297% (1.64e+06/1739397) | 12.644% (11/87) |

## Raw mask accuracy (before overlap protection)

| Arm | Visible mean IoU | Visible pooled recall | Predicted pixels | Absent predicted pixels |
|---|---:|---:|---:|---:|
| raw_part | 97.906% (655/669) | 99.443% (1.73e+06/1739397) | 1,785,361 | 17,412 |
| parent_intersection | 94.022% (629/669) | 95.627% (1.663e+06/1739397) | 1,695,337 | 13,999 |

Fixed success criterion passed: **False**.
Relative reduction in wrong-car effective pixels: 97.285% (215/221).
Requires strictly lower absolute wrong-car effective pixels, with no more than 1 percentage point loss in raw AND effective visible mean IoU and pooled recall. A zero-paint solution cannot pass the recall guard.

## Paired clip bootstrap: parent minus raw

Intervals are descriptive; paired whole clips are resampled and pooled ratios recomputed.

| Representation | Metric | Difference | 95% interval |
|---|---|---:|---|
| raw_mask | wrong_car_pixels | -23433.0000 pixels | [-70299.0000, +0.0000] pixels |
| raw_mask | mean_iou_given_visible | -3.8837 pp | [-10.4945, -0.0132] pp |
| raw_mask | visible_micro_pixel_recall | -3.8163 pp | [-10.5637, -0.0107] pp |
| raw_mask | wrong_car_pixel_fraction_of_predictions | -1.3109 pp | [-3.6084, +0.0000] pp |
| raw_mask | wrong_car_part_frame_rate | -2.2487 pp | [-6.7460, +0.0000] pp |
| raw_mask | false_presence_given_absent | -2.2989 pp | [-40.0000, +0.0000] pp |
| effective_edit | wrong_car_pixels | -215.0000 pixels | [-645.0000, +0.0000] pixels |
| effective_edit | mean_iou_given_visible | -3.7542 pp | [-10.1438, -0.0132] pp |
| effective_edit | visible_micro_pixel_recall | -3.7813 pp | [-10.4663, -0.0107] pp |
| effective_edit | wrong_car_pixel_fraction_of_predictions | -0.0124 pp | [-0.0357, +0.0000] pp |
| effective_edit | wrong_car_part_frame_rate | -1.4550 pp | [-4.3651, +0.0000] pp |
| effective_edit | false_presence_given_absent | -2.2989 pp | [-40.0000, +0.0000] pp |

## Parent accuracy and visible slivers

Whole-parent visible mean IoU: 92.433% (325.4/352); pooled recall: 97.389% (5.496e+06/5643379); wrong-owner pixels: 485,405.

The subgroup below contains visible source part-frames whose two-frame pair has no eligible Day4 ground-truth patch. It includes moving/sliver parts, not only small physical parts.

| Arm | Representation | Visible subgroup frames | Mean IoU | Pooled recall |
|---|---|---:|---:|---:|
| raw_part | raw_mask | 17 | 70.125% (11.92/17) | 77.937% (7669/9840) |
| raw_part | effective_edit | 17 | 59.757% (10.16/17) | 71.575% (7043/9840) |
| parent_intersection | raw_mask | 17 | 49.210% (8.366/17) | 59.482% (5853/9840) |
| parent_intersection | effective_edit | 17 | 48.857% (8.306/17) | 59.319% (5837/9840) |

Scored units per arm/representation: 756 = 669 visible + 87 absent part-frames. Frame0 excluded.

Wrong-car pixels intersect the full visible OTHER car, including its untagged body. False-positive fractions use predicted-pixel denominators; event rates use all scored part-frames. Dense visibility means any nonempty true part; empty visible predictions score zero. Raw and effective denominators are reported separately.

Secondary patch localization reuses the Day4 readout and fixed 0.9404296875 threshold, without gating edits. It has a different tubelet denominator and excludes ambiguous slivers.

This is a new seed set in the same simple 2-D renderer. It tests containment at extra annotation/inference cost, not learned parent identity, face consistency, generative edits, or real-video generalization.
