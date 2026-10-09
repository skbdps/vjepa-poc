# Day 9: source-only temporal memory for latent source-hole repair

## Question and scope

Can observations of the same background location elsewhere in the original
video repair the vacated source of a JEPA-space object move more accurately
than Day 8's local spatial fill, while preserving its successful destination
copy exactly?

This is a direct latent intervention experiment. It adds a deterministic
temporal memory and tests a context-alignment hypothesis; it does not introduce
a new neural architecture, train a new head, or demonstrate propagation from a
single edited frame. The source object's complete mask trajectory is supplied.
Retrieval may use earlier or later tubelets, including another encoder block.
The procedure is offline and is not a causal predictor.

The renderer, frozen encoder, and frozen coarse semantic readout are inherited
from Day 8. Videos have 32 frames at 384 by 384 pixels, encoded as two independent
16-frame blocks. Features have shape [16, 24, 24, 1024]: each token represents
two frames and one 16 by 16 spatial patch. Scenes contain two textured moving
balls in separate lanes over static textured backgrounds. There is no camera
motion, object occlusion, depth, face, car-part, or photorealistic rendering
claim. This static-background setting is deliberately favorable to temporal
retrieval; success would not establish the same mechanism for moving or
previously unseen backgrounds.

## Data and information budget

The 24 existing Day 8 training scenes (seeds 11000 through 11023) may be used for
implementation and diagnostics. The eight existing Day 8 development scenes
(11100 through 11107) select one blend as specified below. The previously
published Day 8 outcome motivates the question, but Day 8 test caches and
per-scene predictions are not used for Day 9 development, selection, or scoring.

The new sealed test consists of 16 independent scene seeds 12200 through 12215,
using the unchanged Day 8 renderer. For index i from 0 through 15, the first
eight scenes use offsets (32, -32, 64, -64)[i mod 4] pixels; the last eight use
(48, -48, 80, -80)[i mod 4]. Report all scenes and the two eight-scene groups
separately. The latter are magnitudes absent from the reused training and
development splits. All offsets are whole spatial patches. Scene generation
must remain independent of the requested displacement except for the target
object's translation.

An operator receives only original-video JEPA features, full source selected
and distractor fractional masks, and the requested signed displacement. Source
object support means any positive coverage in either frame of a tubelet; it is
not a majority-coverage threshold. Destination support is derived by shifting
source support. True target encodings, target RGB, renderer background, empty
scene features, exact trajectories, scene seed, and target masks are not
operator inputs. Genuine source and target encodings are used for evaluation;
the renderer's optional empty scene is never encoded for these operators.

## Fixed operators

Let Z be the immutable original feature array, S the source support, D its
zero-padded horizontal shift, and H = S AND NOT D the vacated source hole.
Let N be the exact Day 8 naive edit: local background filling followed by full
token copying to D. All new operators start from N and write only H. A copied
destination therefore wins every source/destination overlap. For dx = 0 all
operators return exact copies of Z.

At each tubelet, form C by spatially dilating the union of source and distractor
support by one patch using a square neighborhood. A temporal donor for hole
cell (t, y, x) is any original tubelet u where C[u, y, x] is false. The query
tubelet cannot be a donor because the hole is occupied in the original video.
No cyclic wrapping, target-derived donor scoring, or displacement-conditioned
background lookup is allowed.

1. **Naive:** the unchanged Day 8 spatial-fill-and-copy baseline N.
2. **Temporal mean:** at each hole cell, use the float32 arithmetic mean of
   Z[u, y, x] over all eligible donors, with uniform weights. With no eligible
   donor, retain N[t, y, x] exactly.
3. **Aligned temporal mean:** for each eligible donor u, find positions r clear
   of C at both t and u. First restrict r to the boundary-clipped square of
   radius four patches around (y, x). When at least four mutually clear
   positions exist locally, set the alignment offset to the per-channel mean
   of Z[t, r] minus Z[u, r] over those positions. Otherwise use the mean over
   all mutually clear positions in the spatial grid, provided at least one
   exists. If none exists globally, reject that donor for this arm. Add the
   offset to Z[u, y, x], and uniformly average the accepted aligned donors.
   With no accepted donor, retain the exact naive hole fill.
4. **Development-selected blend:** within H, form
   B = (1 - alpha) N + alpha M, where M is either temporal arm and alpha is
   selected below. Outside H retain N exactly. Implement alpha = 0 and alpha = 1
   by direct copying of the corresponding endpoint to avoid unnecessary
   floating-point differences.

An object-free donor pixel is not an object-independent JEPA vector. The encoder
attends across each complete 16-frame block; contextual information and temporal
position can remain in donor features. Context alignment assumes an additive
offset estimated from mutually visible background can reduce this mismatch.
That is a testable approximation, not a claimed factorization of JEPA features.
It also aligns to the original query context, which still contains the object
that the requested edit moves. Record failure as well as success.

## Development selection

Evaluate the joint grid of both temporal variants and alpha in
{0, 0.25, 0.5, 0.75, 1} on all eight development scenes. The score is the
unweighted scene mean of source-hole feature MSE to the genuine target divided
by that scene's original-to-target source-hole MSE, with only a 1e-12 numerical
denominator floor. Record numerators, denominators, token counts, and any
degenerate region. All 1024 feature dimensions are scored.

Select the candidate with the smallest score. Ties use smaller alpha, then
plain temporal mean before aligned temporal mean. Use a fixed absolute score
tolerance of 1e-12 to identify ties, without sequential approximate comparisons:
find the numerical minimum, retain candidates within that tolerance of it,
then apply the tie order. Preserve the full candidate table, selected variant,
alpha, score, and exact input hashes. No additional hyperparameter is selected.

Selection of alpha = 0 means development evidence does not support adopting
temporal memory under this rule. The fixed temporal variants are still scored
on the new test to evaluate the mechanism, but a favorable secondary result
must not be relabeled as success of the selected method or used to change the
reported selection.

## Frozen model, precision, and pretest publication

Use the same official V-JEPA 2.1 ViT-L/384 encoder and checkpoint as Day 8:

- Upstream commit: 204698b45b3712590f06245fbfba32d3be539812.
- Weight SHA256:
  7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6.

Reused Day 8 production caches were generated on CPU with bfloat16 autocast and
stored as float16. Before new test extraction after environment restoration,
repeat the established precision check on training fixture train_11000_dx+32
against its archived Colab float32 reference using those exact model weights
and videos. Source and target are each checked over the full grid, fixed source
hole, and destination: precision MSE divided by the float32 source-to-target
edit MSE must be below 0.01 in all six cases. A zero reference edit denominator
cannot qualify. Save raw results and reference hashes. This one-fixture check
is an engineering gate, not a semantic result or a guarantee for every scene.

Use the validated CPU bfloat16 path consistently for new test encodings; cache
as float16 and evaluate in float32, as in Day 8. If the gate fails, stop before
test access and revise the precision/data plan transparently; do not silently
mix a new backend with the reused selection caches. Record the restored
environment, encoder configuration, numerical precision, and weight hash.

Reuse the exact frozen Day 8 token-only occupancy/RGB readout. Do not fit or
select another readout using any Day 9 test output. Record its file hash and
the original genuine-training/development provenance. It predicts coarse
patch occupancy and mean RGB, not detailed texture or photorealistic pixels.

Before rendering or encoding any new test scene, publish the complete source,
this protocol, precision validation, development candidate table, selection,
model/readout hashes, and input/source hashes in the GitHub experiment branch.
Fetch the committed freeze record and verify its bytes against the local
record. Save the verification time and a check that no fresh test cache existed
before that verification. Run the test only after this succeeds. Do not tune
operators, alignment, blending, or scoring on fresh test outputs. Any necessary
implementation correction after unsealing must be documented and cannot retain
an unqualified untouched-holdout claim.

## Measurements and predeclared decision

For every scene and arm, retain raw feature error, no-op denominator, region
counts, and normalized ratio R. Report source hole and destination separately
and their equal-weight mean to connect with Day 8. The Day 9 primary comparison
is the selected blend's source-hole ratio minus naive's source-hole ratio.
Average at the scene level and use 5,000 paired scene bootstrap resamples with
fixed random seed 1909; report the percentile 95% interval. Scenes, not tokens
or tubelets, are independent resampling units. Report the all-scene primary
result and the two displacement groups. Retain all scenes, including fallback
and degenerate cases, with explicit counts rather than silent exclusion.

The primary latent improvement condition is an upper 95% confidence bound below
zero for that paired difference. Report the same paired comparison for both
fixed temporal variants as secondary mechanism diagnostics. Report raw MSE
alongside ratios, so a small no-op denominator cannot hide the error scale.
The intervals are exploratory and have no multiple-comparison correction.

For semantic interpretation, first report the frozen readout on genuine new
targets. The inherited validity gate requires mean selected-object centroid
error below one 16-pixel patch and coarse color-identity accuracy at least 90%,
with eligible/skipped counts. If these fail, latent comparisons remain
measurable but semantic interpretation is inconclusive. Apply the same fixed
readout to every edit and report coarse source-hole RGB MSE to genuine target
RGB, source-hole mean predicted occupancy (ghosts), selected and distractor
centroids/IoU, color identity, missing detections, and temporal centroid-velocity
error. Ghost occupancy is an absolute readout value; also show its genuine
target reference rather than calling every nonzero prediction a real object.

For a qualified adoption recommendation, require the primary latent condition,
exact preservation assertions below, and both of these additional descriptive
checks: the selected blend's mean source-hole RGB MSE difference from naive is
at most zero, and its mean source-hole ghost-occupancy difference is at most
zero. Report paired confidence intervals for both differences as well. These
sample-mean checks are not statistical proof of population noninferiority;
zero-margin intervals including positive differences leave that uncertainty
visible. Failure of either check prevents a blanket claim that the repair is
better. A zero-alpha selection remains no adoption regardless of secondary
test outcomes.

Report per-scene and per-tubelet donor coverage, count distributions, fallback
fractions, same-block and cross-block donor counts, local/global alignment
counts, and rejected aligned donors. A block comprises eight consecutive
tubelets. Distinguish plain visible donors from accepted aligned donors.
Break down hole errors into covered and fallback positions using declared
availability masks, alongside the all-hole primary metric. Such conditional
breakdowns are diagnostic and must not replace the full-scene comparison.

Verify bitwise equality of every new arm to naive on the destination and all
positions outside H. Verify its destination agrees with the original source
snapshot shifted by the requested displacement, and that source/destination
overlap obeys destination priority. Verify dx = 0 is exact identity. Unchanged
destination/readout values and distractor tokens are consequences of this
construction, not independent evidence of learned consistency. The latent
source-hole comparison to separately encoded genuine targets is the new
evidence being tested.

## Reporting boundary and next interpretation

Save source, freeze/publication evidence, candidate table, per-scene/tubelet
measurements, summary, audit, and fixed-scene visualizations with the repository
progress. Preserve expensive raw feature and prediction evidence separately.
Label probe pictures as coarse occupancy/RGB readouts rather than decoded
edited videos. Report selection failure and fixed-variant failures candidly.

If retrieval has little donor coverage, the experiment tests a memory-availability
limit. If donors are plentiful but latent target error or ghosting worsens,
contextual/temporal mismatch is a possible explanation, supported only to the
extent the alignment control distinguishes it. Neither failure establishes
that all JEPA latent editing is impossible. Success would establish a limited
source-only repair improvement for static-background synthetic scenes and
leave single-edit propagation, realistic backgrounds, object-part identity,
and pixel-video generation open.
