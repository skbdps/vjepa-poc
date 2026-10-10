# Day8 results: direct JEPA latent editing

**The original copying idea passed a substantially stronger test than the Day2
PCA demonstration. Our learned addition did not improve the best copying
baseline.** Across 16 held-out paired scenes, direct token copying reduced the
predeclared regional latent-error ratio from **1.000 to 0.1187**. A frozen coarse
readout placed the edited object within **1.33 pixels** of its intended centroid
on average, compared with **1.06 pixels** when reading genuine target embeddings.

This is a positive result for controlled translation of JEPA representations in
this restricted setting. It is not a demonstration of generated RGB video,
faces, car-part control, automatic selection, or a new JEPA architecture.

## What this test establishes

Day2 moved selected embeddings and inspected a separately fitted PCA display.
Here, a simulator creates matched source and target videos differing only in
one textured ball's horizontal displacement. The target is independently encoded
by the same frozen V-JEPA2.1 ViT-L/384 model. We ask whether an edit to the source
representation approaches that independently observed target representation.
Scoring uses the original 1024 feature dimensions, not PCA coordinates.

There are 24 training, 8 development and 16 test scenes. The learned correction
and geometry-only ablation each use three paired seeds; checkpoints are selected
on development data. Test scenes remain **16 independent observations**, not 48.
Each video has 32 frames, encoded in two separate 16-frame blocks. The encoder
can attend to future frames within a block.

Operators receive source-object and distractor masks for the entire sequence,
plus the requested displacement. Destination masks follow from that request;
target RGB and embeddings enter supervision or scoring only. This generous
oracle isolates editing from tracking. Applying an edit throughout a supplied
trajectory does not establish propagation from a single edited frame.

The primary ratio averages the source-hole and destination error ratios equally.
Each region's error is divided by its own no-op source-to-target error. Thus an
88.13% reduction refers to this regional objective, not full-frame RGB error.
No test region had a degenerate denominator. Intervals use 5,000 scene-bootstrap
draws after averaging learned seeds within scenes; they are exploratory and
uncorrected for multiple comparisons.

## Held-out outcomes

Lower ratios and centroid errors are better. IoU is the coarse occupancy
readout's overlap with true patch coverage, not dense pixel segmentation.
Centroids derive from a 16-pixel patch grid; small centroid errors do not imply
equally precise object boundaries.

| Method | Primary ratio | Source-hole ratio | Destination ratio | Centroid error, px | Occupancy IoU |
|---|---:|---:|---:|---:|---:|
| No edit | 1.0000 | 1.0000 | 1.0000 | 56.06 | 4.36% |
| **Direct copying** | **0.1187** | 0.2088 | **0.0286** | **1.33** | **93.99%** |
| Residual transport | 0.2245 | 0.2088 | 0.2403 | 3.23 | 75.61% |
| Learned residual correction | 0.1499 | **0.2001** | 0.0998 | 1.82 | 89.83% |
| Geometry-only correction | 0.2244 | 0.2087 | 0.2401 | 3.16 | 75.83% |
| Wrong direction | 0.7687 | 0.4816 | 1.0557 | 111.80 | 0.00% |
| Wrong object | 1.0000 | 1.0000 | 1.0000 | 56.06 | 4.36% |
| Genuine target reference | 0.0000 | 0.0000 | 0.0000 | 1.06 | 94.45% |

Direct copying's primary-ratio 95% interval is **[0.1123, 0.1244]**. Both edited
regions improve. Wrong-direction transport illustrates why their separation
matters: removing some source content improves its average ratio, even while
its destination error exceeds no-op and its object moves the wrong way.

The genuine-target probe passes its predeclared calibration gates: 1.06-pixel
mean centroid error and 100% coarse color identity. Direct copying and learned
correction also attain 100% color identity; all 256 target tubelets are eligible,
with none skipped. This distinguishes the two objects' mean colors, not detailed
texture or individual face identity. Selected-core RGB MSE is 0.001956 for
copying, 0.005069 for learned correction, and 0.002007 for genuine targets.

## What learning added—and failed to add

Negative differences below favor the first method on the primary latent ratio.

| Paired comparison | Mean difference | 95% scene-bootstrap interval |
|---|---:|---:|
| Learned correction − direct copying | **+0.03123** | **[+0.02641, +0.03603]** |
| Learned correction − residual transport | −0.07462 | [−0.08483, −0.06342] |
| Learned correction − geometry-only | −0.07446 | [−0.08453, −0.06349] |
| Geometry-only − residual transport | −0.00016 | [−0.00045, +0.00017] |
| Direct copying − residual transport | −0.10584 | [−0.11831, −0.09256] |

Learning improves the residual base and benefits from content inputs, but
**fails the required added-value comparison against both transport baselines**.
Against direct copying it worsens centroid error by 0.490 pixels
[0.343, 0.654], IoU by 4.16 percentage points, and core RGB MSE by 0.003113.
Its modest source-hole ratio improvement, −0.00873 [−0.01417, −0.00339], is
outweighed by destination degradation of +0.07119 [0.06393, 0.07829].

The measured residual-transport failure is primarily at the destination: its
ratio rises from copying's 0.0286 to 0.2403 while source-hole handling is
identical. Subtracting estimated background and adding that residual elsewhere
may be an inappropriate additive model of contextual JEPA features. That is a
hypothesis suggested by the comparison, not an isolated causal explanation.

The geometry-only arm retains the content-bearing residual base; it removes
content only from the correction network. Its comparison cannot isolate the
sequence anchor from other content inputs or establish anchor-specific novelty.

| Training seed | Learned primary ratio | Learned centroid, px | Geometry-only primary ratio |
|---|---:|---:|---:|
| 1801 | 0.14918 | 1.72 | 0.22442 |
| 1802 | 0.15038 | 2.02 | 0.22434 |
| 1803 | 0.15020 | 1.73 | 0.22438 |

Every learned seed's mean ratio is worse than direct copying's 0.11870. Although
all four correct-edit methods pass `combined_exploratory_pass`, that flag tests
improvement over no-op and does not include learned-versus-copying superiority.

## Displacements, removal and temporal behavior

| Test displacement group | Scenes | Copying ratio | Learned ratio | Copying centroid, px |
|---|---:|---:|---:|---:|
| Previously trained magnitudes, ±32/64 px | 8 | 0.11830 | 0.14963 | 1.42 |
| Held-out magnitudes, ±48/80 px | 8 | 0.11909 | 0.15021 | 1.24 |

The copying result persists across these modest interpolation/extrapolation
tests. Learned correction remains worse in both groups; its held-out-magnitude
primary difference versus copying is +0.03112 [0.02584, 0.03691].

Copying has no missing centroids across 256 tubelets; maximum error is 10.86
pixels and the descriptive 95th percentile is 3.44 pixels. However, mean
source-hole probe occupancy remains **0.06551**, versus **0.00227** on genuine
targets. No source-hole value exceeds the 0.5 occupancy threshold, but soft
residual occupancy and latent mismatch remain. This is not perfect erasure.
Learned correction's corresponding mean occupancy is 0.06739.

Centroid velocity error is 1.71 pixels per tubelet step for copying, 2.18 for
learned correction and 1.43 for genuine targets. Learning worsens copying by
0.474 [0.269, 0.668]. No-op also has low velocity error because a constant
positional offset leaves trajectory velocity unchanged; that metric alone
cannot demonstrate the requested edit.

## Interpretation and next scoped test

A pointwise probe follows copied tokens by construction. Its successful
relocation is a sanity check; independent target-encoding agreement is the new
evidence beyond Day2. Exact preservation outside edit support, and unchanged
distractor probe outputs, are also enforced rather than learned. Contextual
target embeddings can change outside the edited pixels.

Keep the successful copied destination. The next scoped experiment should
target source-hole repair using temporal background/context from the original
video, with explicit input restrictions and fresh evaluation scenes. Then test
one-anchor propagation separately, pinning its oracle budget before evaluation.
Do not tune these methods on this test set or substitute another tracking
benchmark. Full RGB reconstruction remains a separate untested capability.

## Execution and evidence

After Colab accelerator quota blocked an accelerator run, production feature
extraction used local CPU BF16, with float16 caches. The existing training-fixture
comparison against Colab FP32 passed all six predeclared precision checks; its
largest precision-error/edit-error ratio was 0.008084, below 0.01. This validates
one fixture, not identical numerics on every scene. Edit heads and probe use
float32. No SAM inputs, labels or predictions enter this experiment.

Pipeline revision: `9ebd665f1be56535199af13d2f6deb8c27af92b1`.
Completed checkpoint freeze, committed before test extraction:
`598f5bc1770c72b4af437a44f834cef41f0aeb08`.

See [test summary](run_2026-10-09/test/summary.json),
[per-clip rows](run_2026-10-09/test/per_clip.csv),
[seed-averaged rows](run_2026-10-09/test/seed_averaged_per_clip.csv),
[per-tubelet rows](run_2026-10-09/test/per_tubelet.csv), and
[analysis artifacts](run_2026-10-09/analysis/).

The [independent audit](run_2026-10-09/analysis/independent_audit.json)
completed 332,228 checks with zero numerical or provenance mismatches. The
maximum numerical difference was 4.44×10⁻¹⁵. It independently recomputed saved
error-map aggregation, semantic metrics, seed averaging, bootstrap intervals
and decision gates, and verified source, checkpoint, feature-cache and prediction
hashes. It recomputed full-dimensional no-op error from cached features; it did
not rerun learned models to independently regenerate their saved error maps.

Two artifact-completeness warnings remain: the learned seed 1801 curve CSV lacks
epochs 29–30, and seed 1803 lacks epoch 30. The retained
[training log](run_2026-10-09/training.log) records all 30 epochs and corroborates
the frozen best selections, epochs 27 and 29. Original CSVs are preserved;
missing fields were not fabricated. The separate
[visual audit](run_2026-10-09/analysis/visual_audit.json) verifies fixed scene/seed
selection, artifact hashes and complete decoding of both four-second previews.
