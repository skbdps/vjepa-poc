# Day 10: one image, one selection, prescribed latent animation

## Question and scope

Can one still image and one selected-object mask produce a sequence of direct
JEPA edits whose representations approach independently encoded images of the
requested object positions? This is a bounded four-image proof of concept, not
a comparison against video generators or evidence of learned motion prediction.

Use the existing official V-JEPA 2.1 ViT-L/384 checkpoint and pinned upstream
revision from Day8. Its native image path receives `[1,3,1,384,384]` and returns
576 1024-dimensional tokens. All encoding, operations and probe evaluation use
CPU float32. The encoder and existing Day8 probe remain frozen. No repetition
of the image as a video, temporal donor frames, new neural head, SAM, diffusion
generator, or external image is needed. The calibration scene is seed13000,
frame15, shifts0/48; its results cannot be counted as test evidence.

## Inputs and editing

The editor receives one RGB image, one supplied selected-object mask, and a
prescribed horizontal movement path. It does not receive a distractor mask,
ground-truth target, empty background, renderer state, or other source frames.
The static selected mask is mapped to the16-pixel feature grid. Every token
intersecting it is selected; these features can include surrounding context.

Encode the source once. Estimate the masked background once using the existing
Day8 local-background rule, excluding the selected object. For each displacement,
start from the immutable source, fill its selected region with that estimate,
and copy original selected tokens into the requested destination. Destination
writes take precedence in overlaps. Zero displacement returns the exact source.
This is mathematically the existing Day8 naive-copy operator on a single grid;
precomputing its background is a consistency property, not a novel repair method.
Hidden background cannot be recovered exactly from one image in general.

## Fixed examples and controls

Use four new scene seeds13200–13203 from the unchanged Day8 renderer, selecting
only frame15 as a still. No other rendered source frame enters the encoder or
editor. Seeds13200–13201 use shifts[0,16,32,48,64,80]pixels;13202–13203 use their
negative counterparts. The source is independent of the requested shift. At
each shift, render the correct static target separately and encode it through
the same native image path for scoring only. Renderer distractor masks and
pixel truth are restricted to scoring and preservation checks.

Arms: no edit, copy-and-repair, wrong-direction copy-and-repair, genuine target.
Display positions0,1,2,3,4,5,4,3,2,1,0 as a return loop. Repeated display states
are not extra observations. Show all four images; image13200 is the fixed primary
preview. Visuals must identify24x24probe output as a coarse diagnostic animation.
Never present target RGB as model-generated animation.

## Checks and interpretation

For every nonzero shift report source-hole and destination feature MSE, each
normalized by no-edit MSE on that same region, and their equal-weight mean.
Keep raw numerators/denominators and all failures. Empty or degenerate regions
must be explicit. Exclude shift0 from normalized edit aggregates. Average the
five shifts within each image first, then weight all four images equally.
No confidence interval or broad generalization claim is justified by this
small descriptive experiment.

Use the unchanged frozen probe to report selected centroid error, occupancy IoU,
coarse color identity, hole ghost occupancy and hole RGB MSE. Genuine-target
readout validity requires mean selected centroid error<16px, all selected
centroids present, and>=90%correct coarse color identity among eligible cases;
report eligibility/skips. This checks localization, not pixel reconstruction.
Copy-and-repair supports the bounded proof only if that reference gate passes,
all edited selected centroids are present, mean edited centroid error<16px,
each image's mean source-hole, destination and balanced latent ratios<1,
no primary-region denominator is degenerate, and every exact invariant holds.
Report wrong-direction controls even if they are unexpectedly competitive.

Required exact invariants: zero-shift identity; identical repeated positions;
unchanged source tokens outside source/destination union; copied destination
equal to translated original tokens; source immutability; fixed filled
background wherever uncovered; and unchanged distractor features where the
prescribed edit does not overlap them. These are code guarantees, not learned
object identity. A failed readout gate makes semantic visualization inconclusive
but does not invalidate the separately reported feature-distance arithmetic.

Publish and byte-verify the protocol, numerical source dependencies, probe hash,
calibration report and fixed specifications before fresh test encoding. Changes
after seeing these results must be described as follow-up development, with
new scenes required for a new validation claim.
