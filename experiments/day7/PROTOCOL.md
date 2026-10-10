# Day 7: learned predictive JEPA part identity

This experiment returns to the central project hypothesis: a persistent latent
representation of an object and its independently addressable parts can improve
identity through a video. It trains a new small predictor and recurrent part
representation over frozen V-JEPA 2.1 features. SAM2 is not used for features,
labels, predictions or losses in this experiment.

This is a **supervised predictive-latent prototype**, not foundation-model
pretraining or automatic part discovery. Synthetic training masks supply part
correspondences and visibility. Test inference gets RGB-derived tokens and four
frame-zero part prompts only. Existing Day3–6 results remain unchanged.

## Mechanism and decisive comparison

Each prompted part has an immutable initial appearance anchor and a learned
128-dimensional recurrent state. Sibling parts share an owner-context vector;
the owner grouping is supplied with the initial tags, not by a parent segmenter.
A learned retrieval query selects matching patch features. A learned visibility
gate controls state updates. A new learned predictor forecasts the same part's
256-dimensional teacher representation eight feature steps later. That forecast
is queued and consumed in retrieval at exactly its corresponding future step.

The frozen encoder is official `vjepa2_1_vit_large_384` at upstream revision
`204698b45b3712590f06245fbfba32d3be539812`, using FP16 inference on four disjoint
16-frame windows. A fixed orthogonal 1024-to-256 projection uses seed 77001.
Centering statistics are fitted exclusively on training tokens, then frozen.
Tokens and pooled prediction targets are normalized; the target branch has no
trainable parameters. The original pretrained predictor is not used: the new
part-conditioned predictor is the component being trained and evaluated here.

The arms are:

1. `retrieval_only`: train the entire recurrent architecture, including the
   active forecast path, using localization and absence losses.
2. `predictive`: identical architecture, initialization and data order, plus
   a future-latent objective with coefficient 1. This isolates what predictive
   supervision adds, rather than attributing all trained retrieval gains to JEPA.
3. `coordinate_only`: the same architecture and retrieval loss, but only fixed
   two-dimensional positional features. It receives no RGB/V-JEPA information.
   This diagnoses exploitation of the renderer's repeated motion schedules.

Frozen full-feature cosine matching, the previous fixed context/motion V-JEPA
tracker with appearance updates disabled, and centered/projected cosine matching
are additional baselines. They share the same fresh clips and initial labels.

## Data and access rules

The dataset is the existing two-dimensional car renderer with fresh seeds:

| Split | Long occlusion | Crossing | Scale/camera | Total |
|---|---|---|---|---:|
| Training | 7100–7111 | 7200–7211 | 7300–7311 | 36 |
| Development | 9100–9102 | 9200–9202 | 9300–9302 | 9 |
| Held-out test | 10100–10105 | 10200–10205 | 10300–10305 | 18 |

Every clip contains 64 frames at 384×384 and four tagged parts. Independent
seed-hashed horizontal/vertical flips and coherent A/B label-group swaps apply
to RGB/masks before encoding. Model inputs exclude the seed, condition, transform
metadata, later masks and trajectories. These transforms reduce simple shortcuts
but do not make this a new-domain or real-video benchmark.

Train/dev feature extraction and fitting precede test extraction. Before test
features are opened, source hashes, chosen development checkpoints, all three
training seeds, centering/projection hashes and baseline thresholds are frozen
and committed to GitHub. All test clips and all model seeds are reported; test
outcomes do not choose a model, epoch, seed or threshold.

The encoder can see later frames **within its 16-frame block**. Therefore this
is offline block-based tracking, not frame-causal tracking. The +8-step latent
target is in the next independently encoded block, so its frames are not in the
source block's visual context. The head itself must pass a future-prefix
isolation test. Later training labels are used solely in losses, never to update
recurrent state or choose model observations.

## Training and measurement

Initial training budget: 30 epochs, AdamW learning rate 3e-4 and weight decay
1e-4, batch size 2, paired initialization/data-order seeds 1701, 1702 and 1703.
An out-of-memory fallback must be logged and applied uniformly across arms.
Any further training adjustment must use development evidence only, be recorded
as a separate development round, and occur before the held-out freeze.

Localization loss is the negative log probability of any valid target patch;
fully absent parts target the null class. Within each batch, the available
visible/absent class means receive equal weight; a single present class gets
weight one. This is batchwise available-class balancing, not a fixed global
class reweighting. Thin/ambiguous targets with no qualifying patch are
ignored, and the initialization tubelet is excluded from retrieval loss/scoring.
The auxiliary objective combines future cosine error with 0.2 times a
same-future-time, visible-part contrastive loss at temperature 0.1. Future targets
are detached normalized pools of fixed JEPA tokens, not learned projections.

Presence and spatial position are factorized: a learned part is emitted when
predicted visibility is at least 0.5, then its maximum-scoring spatial patch is
used. This avoids comparing one patch's probability with the total absence
probability. The decision rule is identical in all learned arms. Frozen-baseline
presence thresholds are selected on development only.

Select each seed/arm's earliest best development epoch by the mean of visible
localization accuracy and absent specificity. Preserve full curves, final-epoch
results and checkpoint hashes to distinguish optimization failure from lack of
generalization. Report all seeds, paired clip comparisons and intervals; three
models tested on the same 18 clips do not constitute 54 independent videos.

The primary contrast is predictive versus retrieval-only balanced utility.
Evidence for the predictive addition requires a positive mean paired improvement
with a clip-bootstrap interval excluding zero, without more than 1 percentage
point loss in visible localization or absent specificity. Report per-seed effects
and between-seed variation separately. Also report wrong-car assignments,
identity switches, first-visible recovery after absence, and condition breakdowns.
Reduced latent loss alone is not success. Compare future predictions with copying
the immutable anchor and the last retrieved representation, where applicable.

After the primary test is complete, descriptive checkpoint interventions replace
queued forecasts with the immutable anchor or remove sibling context. These use
every visual arm/seed, preserve the primary predictions, and first require normal
forward parity. They measure reliance on these inputs, not a better selected
method or proof that predictive training itself helped. Small task-metric changes
can mask numerical effects, so prediction/logit/state sensitivity is also reported.

## Failure diagnosis and scope

- Both learned visual arms beat frozen retrieval but do not differ: trained
  retrieval is useful; predictive supervision is not established as useful.
- Prediction error improves but identity does not: the target/objective can be
  satisfied without solving the identity problem.
- Position-only performs similarly: trajectory regularities may explain gains.
- Training improves while development fails: evidence of overfitting or mismatch.
- Full features beat projected features: quantify representation compression cost.
- Wrong-car errors concentrate at crossings: inspect whether owner information
  or state updates fail, instead of replacing the system with a segmenter.

No conclusion here establishes real-video/face consistency, 3-D geometry,
generative control, self-supervised discovery or literature novelty. See
[RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md) for the relationship to prior work.
