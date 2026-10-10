# Supplementary frozen mask-presence gate

Split: **calibration**. Threshold: **0.9404296875**, copied unchanged from the original SAM2 calibration freeze.

A part's mask is retained in both frames of a pair only when the existing patch readout is eligible and its score reaches that threshold. Frame 0 is preserved. Scores measure mask coverage; they are not calibrated probabilities. This is a two-frame postprocessing rule, not a causal streaming claim.

| Dense metric | Raw masks | Gated masks |
|---|---:|---:|
| Mean visible IoU | 94.26% | 92.14% |
| Visible pixel precision | 94.52% | 94.61% |
| Visible pixel recall | 95.44% | 94.84% |
| Precision including hidden targets | 93.76% | 94.16% |
| False mask presence while hidden | 12.22% | 5.56% |

Removed 42 nonempty predicted part-frame masks: 30 while the true part was visible, and 12 while hidden.

Small/sliver diagnostic: {"part_frames": 30, "nonempty_predicted_masks_removed": 22, "raw_mean_iou": 0.6924954426662986, "gated_mean_iou": 0.01362734423282116, "interpretation": "Includes thin/sliver or moving visible parts without a qualifying 16x16 patch across the pair; not only small physical parts."}

This comparison does not replace the primary raw-mask or patch metrics. The exact policy was proposed and frozen during test execution before its scores were viewed. No threshold was adjusted after this comparison.
