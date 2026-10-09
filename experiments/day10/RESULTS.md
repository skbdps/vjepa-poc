# Day10: controlled JEPA edits from one image

**A single image, one selection, and a prescribed path produced controlled JEPA
feature edits that passed the separately frozen follow-up's gates on four fresh
synthetic images.** The coarse readout placed the selected object within 1.50
pixels on average over nonzero moves and identified its coarse color correctly
in all 24 states. This is a diagnostic animation, not generated realistic video.

The first attempt failed because its transferred video-trained color readout
was invalid. The [original failure report](INITIAL_RESULTS.md) and its evidence
remain unchanged. We then trained the same small diagnostic architecture on
separate native-image data and tested it on new scenes; the editor did not change.

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

The completed fixed configuration used 32 training scenes and eight development
scenes, each with a genuine source image and one genuine shifted target image.
The same 131,716-parameter token-only MLP was trained for exactly 40 epochs on
64 images, with training-only normalization and final-epoch selection.
Development was a single pass/fail check, with no checkpoint or hyperparameter
tuning. The training protocol and configuration were published in commit
`305ebd66742410904c77ec108291169e99cac28c` before those scenes were encoded.

| Genuine development readout | Result |
| --- | ---: |
| Correct coarse color | 16/16 eligible images; none skipped |
| Mean selected centroid error | 0.816 px |
| Mean selected occupancy IoU | 0.9494 |
| Selected centroids present | 16/16 |
| Predeclared development gate | Passed |

The [development gate](native_readout_run_2026-10-10/training/dev_gate.json)
passed. The [independent readout audit](native_readout_run_2026-10-10/analysis/independent_readout_audit.json)
passed 4,716 checks, including training-only normalization and development
predictions. It did not refit the optimizer or repeat encoder inference.

The resulting fixed checkpoint and fresh specifications were published in
commit `99e4edba1ab7d27242a988024ef653675ba2b1a2` and byte-verified before fresh
test seeds 13600–13603 were opened. That test additionally requires at least
90% **edited** coarse-color accuracy, alongside all original localization,
reference-readout, latent-error, and exact-preservation checks.

## Fresh follow-up result

The completed fresh test passed every predeclared gate, including the additional
edited-color requirement. The table compares methods **within this fresh run**.
These scene seeds differ from the original run, so differences between the two
runs' headline feature errors cannot be attributed to retraining the readout.
Readout training does not change the feature-space edit or its feature errors.

| Fresh-run method | Balanced hole/destination error | Hole error | Destination error | Selected centroid error |
| --- | ---: | ---: | ---: | ---: |
| No edit | 1.0000 | 1.0000 | 1.0000 | 48.300 px |
| Copy and repair | **0.1306** | **0.2302** | **0.0309** | **1.495 px** |
| Wrong-direction control | 0.8325 | 0.5844 | 1.0805 | 94.261 px |
| Genuine target reference | 0 | 0 | 0 | 0.761 px |

The balanced regional feature error was 86.94% below this run's no-edit control.
Both source-hole and destination feature errors improved at all 20 nonzero
positions across all four images. Every selected centroid was present. Genuine
and edited coarse-color identity each scored **24/24**, with no skipped states.
The edited occupancy IoU averaged 0.8853 versus 0.9201 for genuine targets.
Tables average the five nonzero positions within each image, then weight images
equally. The color gates also include the four zero-displacement states; display
revisits are never counted again. No confidence interval is claimed.

Background restoration is still imperfect. Mean hole ghost occupancy fell from
0.5053 to 0.0971, but the genuine-target reference was only 0.00113. Mean hole RGB
error fell from 0.05006 to 0.02424; only **15 of 20 individual states improved**,
and image 13603's mean RGB hole error worsened from 0.01417 to 0.02142. Closer JEPA
features therefore do not guarantee a better reconstructed background in every
case. The previews show remaining fill artifacts rather than hiding them.

The [fresh independent audit](native_readout_run_2026-10-10/fresh_test/analysis/independent_audit.json)
passed **20,151 checks**, reconstructing all 96 evaluated feature states and all
96 readouts exactly, with zero metric difference. The [visual audit](native_readout_run_2026-10-10/fresh_test/analysis/visual_audit.json)
verified all four fixed previews and all eight MP4s by full decoding. All
predetermined scenes are available below; no best-case preview was selected.

- [Fresh summary and gates](native_readout_run_2026-10-10/fresh_test/summary.json)
- [All position-level metrics](native_readout_run_2026-10-10/fresh_test/per_position.csv)
- [Image 13600, fixed primary preview](native_readout_run_2026-10-10/fresh_test/analysis/image_13600_preview.gif)
- [Image 13601](native_readout_run_2026-10-10/fresh_test/analysis/image_13601_preview.gif)
- [Image 13602](native_readout_run_2026-10-10/fresh_test/analysis/image_13602_preview.gif)
- [Image 13603](native_readout_run_2026-10-10/fresh_test/analysis/image_13603_preview.gif)
- [Primary evidence panels](native_readout_run_2026-10-10/fresh_test/analysis/image_13600_evidence.mp4), with genuine target RGB clearly labeled as evaluation only

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
