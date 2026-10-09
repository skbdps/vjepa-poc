# Direct latent intervention: paired-video translation

## Question and scope

Does editing frozen JEPA tokens express a requested object displacement while
preserving the selected object's appearance and the unselected object?
This directly extends Day2's latent relocation. It is not another tracker test.

The simulator produces matched source/target videos, differing only by an exact
horizontal translation of one textured moving ball. A second textured moving
ball and background are unchanged. Selected/distractor lanes are randomized but
separate; there is no occlusion, collision, clipping, depth, face or car claim.
The full trajectory is edited explicitly, not forecast from an initial frame.

24 train, 8 development, 16 sealed test scenes have independent scene seeds.
Train/dev use signed32/64px shifts. Eight test scenes use those magnitudes and
eight use signed48/80px shifts, with directions balanced. Every clip has32frames
at384px, encoded as two independent16frame blocks by the pinned official
V-JEPA2.1 ViT-L/384. Features remain1024dimensional; no PCA-space edit is scored.
Encoder attention is offline within each block. CPU/TPU float32, GPU float16,
or conditionally approved CPU bfloat16 inference is recorded. A complete
production run uses one backend and one inference precision for every split;
stored feature caches are float16 in all cases.

CPU bfloat16 autocast may be used only after a precision check on the existing
TRAIN fixture `train_11000_dx+32`, before any test access. Compare source and
target encodings separately against Colab float32 encodings of the exact same
videos using the same official weights/checksum and original1024dimensions.
For each side, divide precision MSE by the Colab float32 source-to-target edit
MSE. All six ratios must be below0.01: source and target, each over the full grid,
fixed source-hole region, and fixed destination region. Retain raw numerator,
denominator, region counts and hashes; a zero edit denominator cannot qualify
the precision check. If any check fails, production uses float32. The check is
an engineering fidelity gate on one training fixture, not evidence of semantic
editing and not a guarantee of identical numerics on every scene. Do not mix
these calibration caches with the chosen production cache.

## Explicit information budget

Operators receive source JEPA tokens, source selected-object and distractor
masks for all frames, and requested dx. This oracle selection isolates editing
from segmentation/tracking. Destination masks are derived by translating source
masks. Genuine target RGB/features are training supervision or scoring only.
The source renderer does not depend on the edit magnitude or direction.
No SAM model, labels, loss or predictions are used.

## Operators

- No-op, naive full-token transport with locally estimated source-hole fill,
  and background-subtracted residual transport.
- Wrong-direction and wrong-object transport are negative controls.
- Learned residual correction: a small shared content/geometry MLP adds a
  correction to residual transport within the source/destination union plus a
  one-patch halo. It has continuous dx, a sequence-wide source appearance anchor,
  and exact zero-displacement identity. It does not update the backbone.
- Geometry-only correction keeps the same transport base, but the correction
  network cannot see token/anchor content. Thus this is an ablation of correction
  content, not a completely content-blind editor.

The two learned arms use three paired seeds1801/1802/1803 and30epochs; choose each
checkpoint only by development error. Retain all seeds. Fit a separate pointwise
occupancy/RGB-patch-mean readout on genuine training embeddings, choose its epoch
on genuine development examples, then freeze it. This readout receives tokens
only, never masks, dx or intended destination. It is a coarse semantic probe,
not a photorealistic video decoder.

## Measurements and decision

Normalize latent squared error by the no-op source-to-genuine-target error:
R=||edited-target||²/||source-target||². On test, no-op is1 in nondegenerate
regions. Report destination and vacated
source-hole separately, plus their equal-weight mean, halo, unchanged distractor,
and background. Scoring regions are fixed from the known intervention and shared
by every method. Incorrect control edits cannot choose their own scoring region.
Also report errors over a common region containing source and both shift signs.
Full-frame averages alone cannot establish success. Preserve numerator,
denominator and counts, including degenerate regions.

Correction training uses raw, unnormalized, region-balanced MSE. Development
checkpoint selection uses a denominator floor fitted on TRAIN only, separately
for each region: `max(1e-8, 0.01 * median(training no-op region MSE))`. Selection
equally averages source-hole and destination ratios; halo is trained and
reported separately. Record the fitted floors and train/dev clipped-denominator
counts. Test reporting instead divides by the actual scene/region no-op MSE,
with only a1e-12 numerical floor, and reports empty/degenerate region counts.
Thus development selection scores and test ratios use different denominator
policies and must not be presented as an identical-scale generalization curve.
The no-op-is1 statement does not apply when the numerical floor is active.

JEPA features are contextual: genuine target features outside changed pixels may
change. Therefore report both exact outside-token preservation and preservation
of unselected-object semantic readout, without conflating them.

The frozen readout checks coarse occupancy, selected-ball centroid trajectory,
appearance, old-location ghosts and distractor preservation. First measure it on
genuine held-out targets; if it cannot localize genuine target balls to within
one16px patch on average, semantic edit conclusions are inconclusive. Neither
PCA colours nor latent distance alone prove meaningful edits.

Primary exploratory evidence: a method's paired scene-level latent ratio is
below1 with a95% scene-bootstrap interval excluding1, both source-hole and
destination improve on no-op, and semantic centroid error improves on no-op
without degrading distractor output. Appearance is checked separately: genuine
target color-identity accuracy must be at least90%; edited accuracy must remain
within5percentage points of that reference, and selected-object RGB error must
improve on no-op with a paired95% interval excluding zero. Color identity compares
the predicted target-core mean RGB with the true selected and distractor source
core means. Cases require nonempty cores and true-color Euclidean separation
of at least0.15 on the0–1scale; eligible/skipped counts are explicit. This is
coarse color identity, not detailed texture or face identity. The implementation
field `combined_exploratory_pass` requires the latent, positional-improvement,
appearance and distractor checks. It means improvement under these checks; it
does not establish arrival at the requested position. For example, reducing a
large centroid error can pass the relative positional check while the ball
remains far from its intended destination. The one-patch centroid gate above
calibrates the genuine-target readout, not the edited result.

Report actual edited centroid errors in pixels, missing-centroid counts,
selected/distractor IoU, source-hole ghost occupancy, and temporal centroid
velocity errors for every method, alongside genuine-target and no-op references.
Report appearance accuracy with eligible/skipped counts and RGB core error.
These absolute values remain necessary even when a relative-improvement flag
passes. No separate absolute-arrival or ghost-free gate is implemented in this
version; do not call the combined flag exact placement, complete removal of the
old object, or consistent video generation. Use the saved per-tubelet values to
show whether errors persist or fluctuate through the sequence.

Report the mean over three learned seeds per scene before bootstrapping
independent scenes, with seen/unseen magnitudes separately. Any learned-method
contribution claim must be supported by the predeclared paired comparisons
against BOTH naive and residual transport; a gain over no-op alone is
insufficient. A claim that learned correction improves the primary latent
objective requires a paired95% interval below zero against both baselines, and
must be limited to that objective unless the semantic/appearance comparisons
also support the claim. Compare learned residual correction against
geometry-only correction before attributing an advantage to source-content or
shared-identity inputs. The geometry-only arm retains a content-carrying
transport base, so this comparison isolates correction-network content access,
not all use of appearance. It does not isolate the sequence anchor from the
other content inputs, so it cannot establish an anchor-specific contribution.
If the paired intervals include zero, describe the
additional contribution as unresolved rather than established. The combined
flag contains none of these learned-versus-baseline requirements.

With16test scenes this is a controlled proof-of-concept, not broad
generalization evidence. Confidence intervals are exploratory and have no
multiple-comparison correction.

Train/dev may inform implementation corrections. Commit complete trained
checkpoint hashes and selection records before extracting test videos. Do not
tune on test. Any post-test diagnosis is labeled exploratory. A latent-stage pass
still leaves intervention-aware prediction and full pixel-video rendering open.
