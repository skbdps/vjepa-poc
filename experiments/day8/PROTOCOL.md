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
Encoder attention is offline within each block. CPU float32 or GPU float16
inference is recorded; a run must use one consistent extraction precision.

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
R=||edited-target||²/||source-target||². No-op is1. Report destination and vacated
source-hole separately, plus their equal-weight mean, halo, unchanged distractor,
and background. Scoring regions are fixed from the known intervention and shared
by every method. Incorrect control edits cannot choose their own scoring region.
Also report errors over a common region containing source and both shift signs.
Full-frame averages alone cannot establish success. Preserve numerator,
denominator and counts, including degenerate regions.

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
without degrading distractor output. For a learned addition, compare against
naive and residual transport too; report mean over three seeds before bootstrapping
independent scenes. Report seen/unseen magnitudes separately. An apparent gain
over no-op does not establish that learning was necessary. With16test scenes this
is a controlled proof-of-concept, not broad generalization evidence.

Train/dev may inform implementation corrections. Commit complete trained
checkpoint hashes and selection records before extracting test videos. Do not
tune on test. Any post-test diagnosis is labeled exploratory. A latent-stage pass
still leaves intervention-aware prediction and full pixel-video rendering open.
