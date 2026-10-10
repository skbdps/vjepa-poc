# Initial native-image run: movement works, transferred color readout fails

The first single-image test improves the requested JEPA state and localizes the
moved object, but **fails the frozen appearance-readout gate**. It must not be
reported as successful appearance-preserving image animation.

Four fresh procedural still images (seeds13200–13203), one supplied selected
mask per image, and five nonzero horizontal displacements were evaluated. The
native V-JEPA2.1 image branch, existing Day8 spatial fill/token copy and frozen
Day8 video-trained probe were used in FP32. No other source frames or target
features enter the editor. Positions are commanded, not predicted.

| Arm | Balanced regional latent error/no edit | Source hole | Destination | Centroid error, px |
|---|---:|---:|---:|---:|
| No edit | 1.000000 | 1.000000 | 1.000000 | 48.2792 |
| Copy and repair | 0.162924 | 0.262528 | 0.063320 | 1.7316 |
| Wrong direction | 0.844271 | 0.608490 | 1.080052 | 96.2635 |
| Genuine target | 0.000000 | 0.000000 | 0.000000 | 2.0133 |

These are equal image means after averaging the five nonzero positions within
each image. The balanced error falls83.71%; all four images have both regional
ratios below1. With only four images, these are descriptive results. Wrong
direction also lowers the balanced ratio because it removes some of the old
object, but it misses the requested destination and its destination ratio is
worse than no edit. A balanced ratio alone cannot establish correct motion.

## Why the full gate failed

Genuine targets localize to1.9663px mean error across all24states, but the old
probe's coarse two-color identity accuracy is **0/24**, below the predeclared90%
gate. Unedited source readouts already fail0/4. Direct inspection shows orange
objects read as blue/purple and blue objects read as red. Thus the problem is
present before an edit is made, and independently encoded correct targets
suffer it too.

An independent functional-layer implementation reproduces every readout
bitwise and every score exactly. The edited selected-core readout colors equal
the original source-readout colors at all24states: the operator preserves an
already wrong visualization. This supports a failure of the video-trained
readout to transfer to native-image features; it does not isolate the precise
internal cause. We did not recolor the display, loosen the gate, or retune on
these four images.

Source-hole RGB error also worsens for image13200 despite lower latent error.
Spatial filling estimates an unseen background and is not exact inpainting.
Exact token copying and unchanged unrelated tokens are implementation
guarantees, not proof of detailed face, texture or part identity.

## Evidence

- [Frozen protocol](PROTOCOL.md), [pretest commit](https://github.com/skbdps/vjepa-poc/commit/5b2f68a84badb047411b2d4197a6c5261aeb7176), and [publication receipt](run_2026-10-10/pretest_publication.json).
- [Summary](run_2026-10-10/summary.json) and [all96position rows](run_2026-10-10/per_position.csv).
- [Independent audit](run_2026-10-10/analysis/independent_audit.json):20,123checks passed;96full tensors and96readouts reconstructed exactly. It did not rerun the encoder or independently verify publication chronology.
- [Color diagnosis](run_2026-10-10/analysis/readout_domain_diagnostic.json).
- [Fixed primary preview](run_2026-10-10/analysis/image_13200_preview.gif) and [visual audit](run_2026-10-10/analysis/visual_audit.json). All four images are displayed; eight MP4s decode fully. Visual integrity passing does not make the experiment gate pass.

The follow-up trains the same small diagnostic readout on disjoint native-image
data and reserves new images for validation. The JEPA encoder and latent edit
operator stay unchanged. This initial failed readout transfer remains part of
the record.
