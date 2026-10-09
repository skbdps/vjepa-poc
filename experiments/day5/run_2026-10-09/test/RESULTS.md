# Day5 parent-intersection experiment

Stage: **test**. Fixed masks, first-frame prompts only; no tuning.

Parent masks cost a second SAM2 pass and two additional whole-car annotations.

## Primary effective edit region

Original overlapping child masks remain protected in both arms. Parent clipping cannot release protected paint.

| Arm | Wrong-car pixels | Wrong-car / predicted pixels | Wrong-car part-frames | Visible mean IoU | Visible pooled recall | Absent false presence |
|---|---:|---:|---:|---:|---:|---:|
| raw_part | 839 | 0.013% (839/6608141) | 3.406% (103/3024) | 93.029% (2484/2670) | 93.068% (6.449e+06/6929781) | 11.017% (39/354) |
| parent_intersection | 70 | 0.001% (70/6484286) | 0.628% (19/3024) | 90.495% (2416/2670) | 91.304% (6.327e+06/6929781) | 10.169% (36/354) |

## Raw mask accuracy (before overlap protection)

| Arm | Visible mean IoU | Visible pooled recall | Predicted pixels | Absent predicted pixels |
|---|---:|---:|---:|---:|
| raw_part | 95.488% (2550/2670) | 96.262% (6.671e+06/6929781) | 7,051,505 | 50,678 |
| parent_intersection | 92.990% (2483/2670) | 94.496% (6.548e+06/6929781) | 6,705,910 | 47,285 |

Fixed success criterion passed: **False**.
Relative reduction in wrong-car effective pixels: 91.657% (769/839).
Requires strictly lower absolute wrong-car effective pixels, with no more than 1 percentage point loss in raw AND effective visible mean IoU and pooled recall. A zero-paint solution cannot pass the recall guard.

## Paired clip bootstrap: parent minus raw

Intervals are descriptive; paired whole clips are resampled and pooled ratios recomputed.

| Representation | Metric | Difference | 95% interval |
|---|---|---:|---|
| raw_mask | wrong_car_pixels | -222325.0000 pixels | [-530793.4500, -98.0000] pixels |
| raw_mask | mean_iou_given_visible | -2.4976 pp | [-4.7436, -0.5732] pp |
| raw_mask | visible_micro_pixel_recall | -1.7668 pp | [-3.4644, -0.1981] pp |
| raw_mask | wrong_car_pixel_fraction_of_predictions | -3.1527 pp | [-7.4319, -0.0016] pp |
| raw_mask | wrong_car_part_frame_rate | -3.2407 pp | [-6.8130, -0.4299] pp |
| raw_mask | false_presence_given_absent | -0.8475 pp | [-4.6971, +0.0000] pp |
| effective_edit | wrong_car_pixels | -769.0000 pixels | [-1570.0000, -68.0000] pixels |
| effective_edit | mean_iou_given_visible | -2.5337 pp | [-4.8147, -0.5851] pp |
| effective_edit | visible_micro_pixel_recall | -1.7641 pp | [-3.4597, -0.1961] pp |
| effective_edit | wrong_car_pixel_fraction_of_predictions | -0.0116 pp | [-0.0251, -0.0011] pp |
| effective_edit | wrong_car_part_frame_rate | -2.7778 pp | [-5.7870, -0.3968] pp |
| effective_edit | false_presence_given_absent | -0.8475 pp | [-4.6971, +0.0000] pp |

## Parent accuracy and visible slivers

Whole-parent visible mean IoU: 94.855% (1327/1399); pooled recall: 97.862% (2.199e+07/22466599); wrong-owner pixels: 995,312.

The subgroup below contains visible source part-frames whose two-frame pair has no eligible Day4 ground-truth patch. It includes moving/sliver parts, not only small physical parts.

| Arm | Representation | Visible subgroup frames | Mean IoU | Pooled recall |
|---|---|---:|---:|---:|
| raw_part | raw_mask | 56 | 70.435% (39.44/56) | 83.018% (2.298e+04/27682) |
| raw_part | effective_edit | 56 | 70.598% (39.53/56) | 82.375% (2.28e+04/27682) |
| parent_intersection | raw_mask | 56 | 46.834% (26.23/56) | 57.727% (1.598e+04/27682) |
| parent_intersection | effective_edit | 56 | 46.834% (26.23/56) | 57.727% (1.598e+04/27682) |

Scored units per arm/representation: 3024 = 2670 visible + 354 absent part-frames. Frame0 excluded.

Wrong-car pixels intersect the full visible OTHER car, including its untagged body. False-positive fractions use predicted-pixel denominators; event rates use all scored part-frames. Dense visibility means any nonempty true part; empty visible predictions score zero. Raw and effective denominators are reported separately.

Secondary patch localization reuses the Day4 readout and fixed 0.9404296875 threshold, without gating edits. It has a different tubelet denominator and excludes ambiguous slivers.

This is a new seed set in the same simple 2-D renderer. It tests containment at extra annotation/inference cost, not learned parent identity, face consistency, generative edits, or real-video generalization.
