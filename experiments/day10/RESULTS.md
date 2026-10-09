# Day10: controlled JEPA edits from one image

**The original experiment passed its feature-distance and localization checks
but failed its overall qualification because the transferred color readout was
invalid. A separately frozen native-image readout follow-up is in progress.**
The [original failure report](INITIAL_RESULTS.md) and its evidence remain
unchanged; the follow-up does not replace that result.

## What was actually tested

One still image, one supplied object mask, and an explicit horizontal movement
path are enough to construct a sequence of JEPA feature edits. We use the
official V-JEPA 2.1 native image branch, not repeated video frames. The source
is encoded once. Selected feature vectors are copied to requested positions,
and a fixed estimate from neighboring features fills their old location.
There are no temporal background donors when the input is one image.

The editor receives no target image, hidden background, second-object mask, or
other source frame. Correct target images are separately rendered and encoded
for scoring. The display converts edited features to coarse 24×24 patch colors
using a small diagnostic readout. It is not full-resolution video generation.
The movement is prescribed; the system does not discover the object or predict
how it should move.

## Original run: useful latent intervention, failed transferred readout

Four fresh source images, seeds 13200–13203, were each tested at five nonzero
displacements and zero displacement. The diagnostic probe was reused unchanged
from Day8's video experiment. These are equal-image averages over nonzero
positions; no edit has normalized regional error 1.

| Original-run method | Balanced hole/destination error | Hole error | Destination error | Selected centroid error |
| --- | ---: | ---: | ---: | ---: |
| No edit | 1.0000 | 1.0000 | 1.0000 | 48.279 px |
| Copy and repair | 0.1629 | 0.2625 | 0.0633 | 1.732 px |
| Wrong-direction control | 0.8443 | 0.6085 | 1.0801 | 96.264 px |
| Genuine target reference | 0 | 0 | 0 | 2.013 px |

Every image improved its source-hole and destination feature errors. However,
the probe identified the correct coarse object color in **0 of 24 genuine
target states**, including zero displacement. This failed the predeclared
90% reference-color gate. The overall result is therefore **not qualified**,
even though localization and latent-distance arithmetic remain informative.

An independent audit reproduced all 96 evaluated feature states and readouts,
with 20,123 checks and no audit errors. Color failure was already present on the
unedited source images; copied object features and their probe colors remained
unchanged. That identifies a failed video-to-image readout transfer rather than
evidence that copying changed the object's original feature vectors.

- [Original summary and gates](run_2026-10-10/summary.json)
- [Original independent audit](run_2026-10-10/analysis/independent_audit.json)
- [Original fixed preview](run_2026-10-10/analysis/image_13200_preview.gif)

## Separate follow-up: calibrating the same readout architecture on images

The [follow-up protocol](native_readout_PROTOCOL.md) changes only the diagnostic
readout's training domain. The pretrained JEPA encoder, native image interface,
source-only editor, and regional metrics stay unchanged. The initial failure
is preserved, and its four test scenes are excluded from follow-up training,
development, and fresh validation.

The fixed configuration uses 32 training scenes and eight development scenes,
each with a genuine source image and one genuine shifted target image. The
same token-only MLP is trained for exactly 40 epochs, with training-only
normalization and final-epoch selection. Development is a single pass/fail
check, with no checkpoint or hyperparameter tuning. The training protocol and
configuration were published in commit `305ebd66742410904c77ec108291169e99cac28c`
before those scenes were encoded.

Fresh test scenes are seeds 13600–13603. They remain unopened until the fixed
probe passes development and its checkpoint, evidence, and fresh specifications
are published. The follow-up also requires at least 90% **edited** coarse-color
accuracy, alongside all original localization, reference-readout, latent-error,
and exact-preservation checks.

Fresh results and audit evidence will be recorded here after that frozen run.
Any comparisons will be made against controls within each run: the two runs
use different scene seeds, so differences between their headline feature errors
cannot be attributed to retraining the readout.

## Scope and exact guarantees

The same original features are used at every requested position. Zero movement
returns the original features exactly; returning to a previous position returns
the same edit exactly. Tokens outside the source/destination union are unchanged,
and copied destination tokens equal their translated source tokens. The same
background estimate is reused wherever the source becomes uncovered. These are
properties of the algorithm, not learned identity or motion consistency.

The spatial repair is mathematically the previous Day8 naive fill applied to a
single image, not a new inpainting algorithm. A single image generally cannot
reveal the true background hidden behind an object. The test contains simple
synthetic scenes, supplied masks, integer 16-pixel shifts, and no occlusion or
rotation. Coarse colors cannot establish texture, face, or part identity.
Neither run demonstrates photorealistic animation, automatic object selection,
physical motion prediction, a new JEPA architecture, or superiority over video
generators. See [README.md](README.md) for reproducible commands and input limits.
