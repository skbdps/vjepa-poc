# Day 2: Validating V-JEPA 2.1 Dense Features on Bouncing Ball

## Context

Day 1 produced (1, 4608, 1024) patch embeddings from V-JEPA 2.1 ViT-L on
a synthetic bouncing ball video. Before we attempt to edit these embeddings
(day 6-7 editability test), we need to verify that the features actually
carry the dense, spatially-grounded information the paper promises.

Our bouncing ball is out-of-distribution for V-JEPA 2.1 (which was trained
on natural video), so we cannot assume dense features just because the paper
demonstrates them on Kinetics / DAVIS / etc.

This plan replicates the paper's own validation methodology (Sections 2.2,
3.5, 3.6, 3.9) on our specific input, then adds one step beyond.

## Tensor layout reference

Encoder output: (1, 4608, 1024) where 4608 = 8 temporal x 24 x 24 spatial.

- Tubelet size 2 means each temporal token covers 2 adjacent input frames
- Patch size 16 means each spatial token covers a 16x16 pixel region in a 384x384 frame
- Our 16-frame input gives 8 temporal tokens at frames 0,2,4,...,14 conceptually

When we need to associate a temporal token to input frames, temporal token t
corresponds to input frames 2t and 2t+1.

## Steps

### Step A. PCA to RGB visualization (paper replication of Fig 1, Fig 15)

Paper claim: "similar semantic parts are mapped to the same PCA components"
(Section 2.2, validated in 3.9).

Procedure:

1. Load bounce-embeddings.pt and reshape to (4608, 1024).
2. Fit sklearn PCA with 3 components.
3. Reshape to (T=8, H=24, W=24, 3).
4. Normalize each component to [0, 1] via per-component min-max.
5. Optionally try the 6 channel permutations and pick the clearest.
6. Upsample spatially to 384x384 via nearest-neighbor.
7. Expand temporally so each temporal token is shown for 2 consecutive frames.
8. Render side-by-side with original 16 sampled frames, write to mp4.

Success criterion: ball region has a distinct color consistent across time;
background is another color. Even if the ball blurs across 2-3 patches that's
fine — the patch grid is coarse.

Failure modes to watch:

- Noise everywhere, no spatial coherence: OOD failure, need different input
- Strong PC1 pattern but ball and background are same color: feature variance
  is not dominantly about the ball
- Color changes frame-to-frame: temporal inconsistency (unexpected)

Quantitative sanity check: print the explained variance ratio of the top 3
PCs. Sum above 0.5 suggests clean structure (our own bar, not in paper).

### Step B. Non-parametric label propagation (paper Section 3.6 protocol)

Paper claim: patch features support nearest-neighbor object tracking across
frames via cosine similarity (used for video object segmentation on DAVIS).

Procedure:

1. In the original pixel space, identify which spatial patches in frame 0
   contain the ball. Use a color threshold (red > 150, others < 100). Map
   to a mask of (24, 24) patch indices.
2. Take the corresponding tokens from temporal slice 0: that's our "ball
   query set" — a small set of 1024-dim vectors.
3. For each subsequent temporal slice t in 1..7:
   a. Compute cosine similarity between each of the 576 spatial tokens in
      slice t and each query vector.
   b. For each spatial position in slice t, take max similarity across
      query vectors.
   c. Mark patches above a threshold as "ball" (top-k or similarity > 0.7).
4. Upsample the ball-map to 384x384 and overlay on each frame.
5. Write as side-by-side video: original mid ball-map overlay.

Success criterion: the highlighted region tracks the ball's trajectory
across all 8 temporal slices, approximately following the parabolic path.

Failure modes:

- Ball-map stays in frame 0 position: features are positional, not object-aware
- Ball-map scatters randomly: no temporal consistency in features
- Ball-map covers everything: threshold too low, features don't discriminate

### Step C. PCA histogram analysis (our addition)

Goal: before running K-means, check whether the feature distribution is
actually bimodal along informative axes.

Procedure:

1. Take PCA result from Step A.
2. For each of PC1, PC2, PC3:
   a. Plot histogram of the 4608 token values.
   b. Identify if there are two clear peaks.
3. If bimodal on any PC, check if a simple threshold on that PC separates
   ball patches from background (compare to Step B's pixel-space mask).

Success criterion: at least one PC shows clear bimodality, and a threshold
on that PC produces a mask approximately matching the pixel-space ball mask.

This is the weakest-but-most-informative step. If PCs are bimodal, K-means
is overkill. If PCs are unimodal, K-means may still find clusters but the
structure is more subtle than "ball vs background".

### Step D. K-means clustering (our original idea)

Only run if Steps A-C suggest discrete object structure exists.

Procedure:

1. Flatten (4608, 1024) -> 4608 samples in 1024-d.
2. Run sklearn KMeans with K=2 and K=3.
3. Map cluster assignments back to (T=8, H=24, W=24).
4. Visualize: assign each cluster a color, upsample, overlay on video.
5. Compare cluster assignments to the pixel-space ball mask from Step B
   on frame 0 — quantify via IoU.

Success criterion: one cluster has IoU above 0.5 with the ball region, and
this cluster tracks the ball across time.

### Step E. Writeup

Summarize findings in experiments/day2/RESULTS.md:

- What worked, what didn't, with numbers.
- Embedded images or thumbnails of outputs.
- Verdict: do V-JEPA 2.1 features support our editability hypothesis on
  this video?

## Out of scope for day 2

- No probe training (takes too long, no labels available)
- No diffusion/decoder experiments (day 8+)
- No editing experiments (day 6-7)
- No quantitative comparison vs V-JEPA 2 (not critical for go/no-go)

## Code layout

    experiments/day2/
      PLAN.md              - this file
      pca_visualize.py     - Step A
      label_propagation.py - Step B
      pca_histogram.py     - Step C
      kmeans_cluster.py    - Step D
      outputs/             - rendered mp4s, png plots (gitignored)
      RESULTS.md           - Step E

## Reproducibility

All scripts load from experiments/day1/outputs/bounce-embeddings.pt on TPU,
or regenerate it via encode_bounce.py if missing. Scripts should be
CPU-runnable (sklearn PCA and KMeans don't need TPU) and should work locally
on Mac if the .pt file is scp'd over.
