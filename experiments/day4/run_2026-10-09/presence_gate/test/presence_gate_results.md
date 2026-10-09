# Supplementary frozen mask-presence gate

Split: **test**. Threshold: **0.9404296875**, copied unchanged from the original SAM2 calibration freeze.

A part's mask is retained in both frames of a pair only when the existing patch readout is eligible and its score reaches that threshold. Frame 0 is preserved. Scores measure mask coverage; they are not calibrated probabilities. This is a two-frame postprocessing rule, not a causal streaming claim.

| Dense metric | Raw masks | Gated masks |
|---|---:|---:|
| Mean visible IoU | 94.42% | 92.20% |
| Visible pixel precision | 94.22% | 94.32% |
| Visible pixel recall | 95.10% | 94.43% |
| Precision including hidden targets | 93.55% | 93.80% |
| False mask presence while hidden | 11.13% | 5.57% |

Removed 128 nonempty predicted part-frame masks: 98 while the true part was visible, and 30 while hidden.

Small/sliver diagnostic: {"part_frames": 81, "nonempty_predicted_masks_removed": 65, "raw_mean_iou": 0.7390657641977265, "gated_mean_iou": 0.0001134023678433748, "interpretation": "Includes thin/sliver or moving visible parts without a qualifying 16x16 patch across the pair; not only small physical parts."}

This comparison does not replace the primary raw-mask or patch metrics. The exact policy was proposed and frozen during test execution before its scores were viewed. No threshold was adjusted after this comparison.
